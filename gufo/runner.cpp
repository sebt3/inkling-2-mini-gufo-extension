#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../core/gguf.hpp"
#include "../host/chat_template.hpp"
#include "../host/tokenizer.hpp"
#include <gufo/ext/sha256.hpp>
#include "src/core/image.hpp"
#include "continuation_adapter.hpp"
#include "image_prep.hpp"
#include "probe.hpp"
#include "session.hpp"

namespace gufo::models::inkling {
namespace {

using ext::TextRunnerToken;
constexpr std::string_view kStateAbi = "inkling2-rocm-session-v1";
constexpr std::uint32_t kDefaultContext = 131072;

void Log(const std::string& line) { std::fprintf(stderr, "[inkling] %s\n", line.c_str()); }

std::vector<std::uint8_t> CompatibilityIdentityBytes(const std::string& fingerprint, std::uint32_t max_context) {
  std::ostringstream id;
  id << "schema=gufo-text-continuation-v1\nartifact_sha256=" << fingerprint << "\nmodel_kind=inkling2_loop\ncontext_tokens=" << max_context
     << "\nchat_template=inkling2-compiled-v1\nstate_abi=" << kStateAbi << "\npayload_layout=inkling-row-components-v" << ContinuationLayout::kLayoutVersion << "\n";
  const auto s = id.str(); return {s.begin(), s.end()};
}
std::vector<std::uint8_t> ProcessIdentityBytes(const void* engine, std::uint32_t max_context) {
  std::ostringstream id;
  id << "schema=gufo-process-text-continuation-v1\nmodel_kind=inkling2_loop\nmodel_instance=" << engine << "\ncontext_tokens=" << max_context
     << "\nstate_abi=" << kStateAbi << "\npayload_layout=inkling-row-components-v" << ContinuationLayout::kLayoutVersion << "\n";
  const auto s = id.str(); return {s.begin(), s.end()};
}

// Image embeddings of a prompt: rows[k] holds the embeddings of the image whose placeholder tokens start at starts[k].
struct ImageEmbedding { std::vector<float> rows; int n_tokens{0}; std::string hash; };
struct PromptContext final : ext::TextPromptContext {
  std::vector<std::size_t> starts;
  std::vector<std::shared_ptr<const ImageEmbedding>> images;
};

class InklingState final : public ext::TextRunnerState {
 public:
  InklingState(Session& session, cache::Adapter& adapter, cache::Slot& slot) : session_(&session), adapter_(&adapter), slot_(&slot) {}
  void Invalidate() noexcept override { position_ = 0; thinking_ = false; pending_.reset(); (void)adapter_->Invalidate(*slot_); }
  Session& session() const { return *session_; }
  std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t p) noexcept { position_ = p; }
  bool thinking() const noexcept { return thinking_; }
  void set_thinking(bool v) noexcept { thinking_ = v; }
  std::optional<std::int32_t>& pending() noexcept { return pending_; }
  const PromptContext* context() const noexcept { return context_.get(); }
  void set_context(std::shared_ptr<const ext::TextPromptContext> c) { context_ = std::static_pointer_cast<const PromptContext>(c); }
 private:
  Session* session_; cache::Adapter* adapter_; cache::Slot* slot_; std::size_t position_{0}; bool thinking_{false};
  std::shared_ptr<const PromptContext> context_;
  std::optional<std::int32_t> pending_;
};

InklingState& Require(ext::TextRunnerState& s) { auto* p = dynamic_cast<InklingState*>(&s); if (!p) throw std::logic_error("text runner state is not Inkling"); return *p; }
const InklingState& Require(const ext::TextRunnerState& s) { auto* p = dynamic_cast<const InklingState*>(&s); if (!p) throw std::logic_error("text runner state is not Inkling"); return *p; }

std::string RoleName(ext::ChatRole r) {
  switch (r) {
    case ext::ChatRole::kSystem: case ext::ChatRole::kDeveloper: return "system";
    case ext::ChatRole::kUser: return "user";
    case ext::ChatRole::kAssistant: return "assistant";
    case ext::ChatRole::kTool: return "tool";
  }
  throw std::invalid_argument("unknown chat role");
}
const char* EffortName(ReasoningEffort e) {
  switch (e) { case ReasoningEffort::kMinimal: return "minimal"; case ReasoningEffort::kLow: return "low"; case ReasoningEffort::kMedium: return "medium";
    case ReasoningEffort::kHigh: return "high"; case ReasoningEffort::kXHigh: return "xhigh"; case ReasoningEffort::kMax: return "max"; }
  return "high";
}

class InklingTextRunner final : public ext::HipTextModelRunner {
 public:
  static constexpr std::size_t kPrefillChunk = 1024;

  InklingTextRunner(std::shared_ptr<::inkling::GpuEngine> engine, std::unique_ptr<::inkling::Tokenizer> tokenizer, std::uint32_t max_context, std::string fingerprint)
      : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), max_context_(max_context) {
    identity_ = fingerprint.empty() ? ProcessIdentityBytes(engine_.get(), max_context_) : CompatibilityIdentityBytes(fingerprint, max_context_);
    stop_ = tokenizer_->token_id("<|content_model_end_sampling|>");
    think_open_ = tokenizer_->token_id("<|content_thinking|>");
    end_message_ = tokenizer_->token_id("<|end_message|>");
    image_pad_ = tokenizer_->token_id("<|unused_200054|>");
    hidden_ = {tokenizer_->token_id("<|message_model|>"), tokenizer_->token_id("<|content_text|>"), stop_};
  }

  ext::TextRunnerDescriptor Descriptor() const override {
    sampling::TextSamplingValues v; v.temperature = 1.0F; v.top_k = 0; v.top_p = 1.0F; v.min_p = 0.0F;
    return {
        .model_id = "Inkling-2-Mini",
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities = ext::TextRunnerCapabilities{.supports_vision = engine_->has_vision(), .incremental_prefill = true, .final_token_advance_required = false, .incremental_text_is_exact = true, .multi_token_decode = engine_->has_dspark(), .prefix_reuse = true},
        .sampling_defaults = sampling::TextSamplingDefaults{v, v, v},
    };
  }

  ext::TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0, total_bytes = 0; std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) capacity = free_bytes;
    return {
        .resident_weights_bytes = engine_->info().weight_bytes,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = engine_->seq_bytes(),
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = ext::HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = ext::HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  std::vector<ext::TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}, {.kind = ext::TextExecutionPlanKind::kBatched, .physical_width = 8}};
  }

  std::unique_ptr<ext::ComponentCacheResources> CreateComponentCacheResources(cache::ResourceLedger& ledger, std::size_t) const override {
    return hip::CreateHipComponentCacheResources(std::make_unique<InklingContinuationAdapter>(engine_, max_context_, identity_), ledger,
                                                 hip::ComponentCacheHooks{[] { return ext::HostSnapshotCeilingBytes(); }, [](std::string_view) {}, [](std::string_view) {}});
  }
  std::unique_ptr<ext::TextRunnerState> BindComponentState(cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& a = dynamic_cast<InklingContinuationAdapter&>(adapter);
    return std::make_unique<InklingState>(a.GetSession(slot), adapter, slot);
  }
  void ReconcileComponentState(ext::TextRunnerState& state, std::size_t position) const override { Require(state).set_position(position); }

  std::shared_ptr<const sampling::ConstraintVocabulary> BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(tokenizer_->size(), [this](std::uint32_t id) {
      return sampling::ConstraintVocabulary::Piece{tokenizer_->decode({(int)id}), (int)id == stop_};
    });
  }
  std::uint32_t VocabularySize() const override { return (std::uint32_t)tokenizer_->size(); }

  std::vector<TextRunnerToken> Tokenize(std::string_view text) const override { return ToTokens(tokenizer_->encode(std::string(text), false)); }
  std::vector<TextRunnerToken> TokenizeControl(std::string_view text) const override { return ToTokens(tokenizer_->encode(std::string(text), true)); }

  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(const ext::ChatRequest& request) const override {
    for (const auto& m : request.messages) if (!m.images.empty()) throw std::invalid_argument("image input requires PreparePrompt");
    auto p = PreparePrompt(request); if (!p) return std::nullopt; return std::move(p->tokens);
  }

  std::optional<ext::TextPreparedPrompt> PreparePrompt(const ext::ChatRequest& request) const override {
    if (!request.tools.empty() && request.tool_choice != ext::ChatRequest::ToolChoice::kNone) throw std::invalid_argument("Inkling tool calls are not supported yet");
    std::vector<::inkling::Message> messages; std::vector<std::shared_ptr<const ImageEmbedding>> images;
    for (const auto& m : request.messages) {
      if (!m.tool_calls.empty()) throw std::invalid_argument("Inkling tool calls are not supported yet");
      const std::string text = m.content + m.framing_suffix;
      ::inkling::Message out;
      if (m.images.empty()) out = ::inkling::Message::make(RoleName(m.role), text);
      else {
        if (!engine_->has_vision()) throw std::invalid_argument("this Inkling server was started without a vision projector (--mmproj)");
        out.role = RoleName(m.role); out.content = ::inkling::Message::Content::Parts;
        std::vector<const ext::ChatMessage::ImagePart*> order; for (const auto& im : m.images) order.push_back(&im);
        std::stable_sort(order.begin(), order.end(), [](auto* a, auto* b) { return a->offset < b->offset; });
        std::size_t pos = 0;
        for (const auto* im : order) {
          const std::size_t off = std::min(im->offset, m.content.size());
          if (off > pos) { out.parts.push_back({"text", text.substr(pos, off - pos)}); pos = off; }
          out.parts.push_back({"image", {}}); images.push_back(Embed(*im));
        }
        if (pos < text.size()) out.parts.push_back({"text", text.substr(pos)});
      }
      out.reasoning_content = m.thought; out.name = m.name; messages.push_back(std::move(out));
    }
    ::inkling::Options options;
    if (request.reasoning.enabled.has_value() && !*request.reasoning.enabled) options.reasoning_effort = "none";
    else if (request.reasoning.effort.has_value()) options.reasoning_effort = EffortName(*request.reasoning.effort);
    const std::vector<int> ids = tokenizer_->encode(::inkling::render_prompt(messages, options), true);
    ext::TextPreparedPrompt prepared; auto ctx = std::make_shared<PromptContext>(); std::size_t k = 0;
    for (int id : ids) {
      if (id == image_pad_ && k < images.size()) {   // the placeholder of image k becomes one token per embedding row
        ctx->starts.push_back(prepared.tokens.size()); ctx->images.push_back(images[k]);
        for (int r = 0; r < images[k]->n_tokens; ++r) prepared.tokens.push_back((TextRunnerToken)id);
        ++k;
      } else prepared.tokens.push_back((TextRunnerToken)id);
    }
    if (k != images.size()) throw std::invalid_argument("image placeholders do not match the images of the request");
    if (!images.empty()) {
      std::string all;
      for (std::size_t i = 0; i < images.size(); ++i) {
        ctx->cache_prefixes.push_back({ctx->starts[i], std::vector<std::uint8_t>(all.begin(), all.end())});
        all += images[i]->hash;
      }
      ctx->cache_identity.assign(all.begin(), all.end()); prepared.context = ctx;
    }
    return prepared;
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::vector<int> ids(tokens.begin(), tokens.end()); return tokenizer_->decode(ids);
  }

  // Reasoning arrives as <|message_model|><|content_thinking|>...<|end_message|>; the role and content-kind markers of the
  // answer stay out of the text, the two thinking markers are kept so the host can split reasoning from the answer.
  ReasoningTags ReasoningBudgetTags() const override { return {"<|content_thinking|>", "<|end_message|>", {"<|end_message|>"}}; }
  ReasoningMarkup ReasoningDelimiters() const override { return {"<|content_thinking|>", "<|end_message|>"}; }

  void SetPromptContext(ext::TextRunnerState& state, std::shared_ptr<const ext::TextPromptContext> context) const override { Require(state).set_context(std::move(context)); }

  void PreparePrefixReuse(ext::TextRunnerState& state, std::span<const TextRunnerToken> prefix) const override {
    if (Require(state).position() != prefix.size()) throw std::logic_error("Inkling reused prefix does not match checkpoint");
  }

  ext::TextPrefillStep Prefill(ext::TextRunnerState& state, std::span<const TextRunnerToken> prompt, std::size_t offset, std::size_t max_input_tokens) const override {
    auto& s = Require(state);
    if (offset != s.position()) throw std::logic_error("Inkling prefill offset does not match state");
    if (offset >= prompt.size()) throw std::logic_error("Inkling prefill has no remaining input");
    const std::size_t consumed = std::min<std::size_t>({max_input_tokens, prompt.size() - offset, kPrefillChunk});
    std::vector<std::int32_t> chunk; chunk.reserve(consumed);
    for (std::size_t i = 0; i < consumed; ++i) chunk.push_back((std::int32_t)prompt[offset + i]);
    std::string error; std::vector<int> mask; std::vector<float> embd;
    if (const PromptContext* pc = s.context()) {   // rows of the image embeddings that fall in this chunk
      const std::size_t dim = (std::size_t)engine_->info().n_embd;
      for (std::size_t k = 0; k < pc->starts.size(); ++k) {
        const std::size_t a = pc->starts[k], b = a + (std::size_t)pc->images[k]->n_tokens; const std::size_t lo = std::max(a, offset), hi = std::min(b, offset + consumed);
        if (lo >= hi) continue;
        if (mask.empty()) { mask.assign(consumed, 0); embd.assign(consumed * dim, 0.f); }
        for (std::size_t p = lo; p < hi; ++p) { mask[p - offset] = 1; std::memcpy(&embd[(p - offset) * dim], &pc->images[k]->rows[(p - a) * dim], dim * 4); }
      }
    }
    if (!s.session().Append(chunk, &error, mask.empty() ? nullptr : mask.data(), embd.empty() ? nullptr : embd.data())) { s.Invalidate(); throw std::runtime_error("Inkling prefill failed: " + error); }
    s.set_position(offset + consumed);
    return {.consumed_tokens = consumed, .decode_ready = offset + consumed == prompt.size()};
  }

  ext::TextDecodeSelection SelectNext(ext::TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& s = Require(state);
    if (s.position() >= max_context_) return {.stop = true};
    const auto logits = s.session().Logits();
    if (logits.empty()) throw std::runtime_error("Inkling token selection has no logits");
    const auto token = static_cast<int>(sampler.Sample(logits));
    if (s.stop_at_eos() && token == stop_) return {.stop = true, .token = 0};
    return {.stop = false, .token = (TextRunnerToken)token, .piece = Piece(token, s.thinking()), .logprobs = std::nullopt};
  }

  void Advance(ext::TextRunnerState& state, TextRunnerToken token) const override {
    auto& s = Require(state); FlushPending(s);
    const std::int32_t input = (std::int32_t)token; std::string error;
    if (!s.session().Append({&input, 1}, &error)) { s.Invalidate(); throw std::runtime_error("Inkling decode failed: " + error); }
    if ((int)token == think_open_) s.set_thinking(true); else if ((int)token == end_message_) s.set_thinking(false);
    s.set_position(s.position() + 1);
  }

  // One token for each of several sessions in a single engine pass. Speculative sessions and sessions with a pending token go one by one.
  void AdvanceBatch(std::span<const ext::TextRunnerAdvance> advances) const override {
    std::vector<int> seqs, tokens, pos; std::vector<InklingState*> states; std::vector<const ext::TextRunnerAdvance*> rest;
    for (const auto& a : advances) {
      auto& s = Require(a.state.get());
      if (s.session().speculative() || s.pending().has_value() || s.position() >= max_context_) { rest.push_back(&a); continue; }
      seqs.push_back(s.session().seq()); tokens.push_back((int)a.token); pos.push_back((int)s.position()); states.push_back(&s);
    }
    if (!seqs.empty()) {
      std::vector<std::vector<float>> logits; std::string error;
      if (!engine_->DecodeBatch(seqs, tokens, pos, &logits, &error)) {
        for (auto* s : states) s->Invalidate();
        throw std::runtime_error("Inkling batched decode failed: " + error);
      }
      for (std::size_t i = 0; i < states.size(); ++i) {
        states[i]->session().CommitDecoded(std::move(logits[i]));
        if (tokens[i] == think_open_) states[i]->set_thinking(true); else if (tokens[i] == end_message_) states[i]->set_thinking(false);
        states[i]->set_position(states[i]->position() + 1);
      }
    }
    for (const auto* a : rest) Advance(a->state.get(), a->token);
  }

  // Plain sessions: sample each one's next token, then advance all of them together; the rest use their own DecodeStep.
  std::vector<ext::TextDecodeStep> DecodeBatch(std::span<const ext::TextRunnerDecode> decodes) const override {
    std::vector<ext::TextDecodeStep> out(decodes.size()); std::vector<ext::TextRunnerAdvance> adv; std::vector<std::size_t> adv_idx;
    for (std::size_t i = 0; i < decodes.size(); ++i) {
      auto& d = decodes[i]; auto& s = Require(d.state.get());
      if (s.session().speculative() || d.max_tokens != 1 || s.pending().has_value()) { try { out[i] = DecodeStep(d.state.get(), d.max_tokens, d.sampler.get()); } catch (...) { out[i].failure = std::current_exception(); } continue; }
      try {
        if (s.position() >= max_context_) { out[i].stop = true; continue; }
        auto sel = SelectNext(d.state.get(), d.sampler.get());
        if (sel.stop) { out[i].stop = true; out[i].terminal_logprobs = std::move(sel.logprobs); continue; }
        adv.push_back({d.state, sel.token, &out[i].failure}); adv_idx.push_back(i); out[i].selections.push_back(std::move(sel));
        out[i].execution_plan = {ext::TextExecutionPlanKind::kBatched, 8};
      } catch (...) { out[i].failure = std::current_exception(); }
    }
    if (!adv.empty()) {
      try { AdvanceBatch(adv); } catch (...) { for (auto i : adv_idx) out[i].failure = std::current_exception(); }
    }
    return out;
  }

  // Consume a token that was sampled and emitted but not yet fed to the session, so the state can move one token at a time again.
  void FlushPending(InklingState& s) const {
    if (!s.pending().has_value()) return;
    const std::int32_t t = *s.pending(); s.pending().reset(); std::string error;
    if (!s.session().Append({&t, 1}, &error)) { s.Invalidate(); throw std::runtime_error("Inkling decode failed: " + error); }
    s.set_position(s.position() + 1);
  }

  // Speculative decoding with the DSpark drafter. Each call is one block: the drafter proposes up to 7 tokens after the anchor (the
  // pending token), one verification pass runs the anchor and the proposals through the target, and a proposal is kept when the target's
  // own sample at its row equals it, otherwise that sample is the next token: every emitted token is a draw from the target distribution.
  ext::TextDecodeStep DecodeStep(ext::TextRunnerState& state, std::size_t max_tokens, sampling::SamplerState& sampler) const override {
    auto& s = Require(state);
    if (!s.session().speculative()) return ext::TextModelRunner::DecodeStep(state, max_tokens, sampler);
    ext::TextDecodeStep step; if (max_tokens == 0) return step;
    if (s.position() >= max_context_) { step.stop = true; return step; }
    bool phase = s.thinking();
    const auto emit = [&](int token) {
      step.selections.push_back(ext::TextDecodeSelection{.stop = false, .token = (TextRunnerToken)token, .piece = Piece(token, phase), .logprobs = std::nullopt});
      if (token == think_open_) phase = true; else if (token == end_message_) phase = false;
    };
    int anchor;
    if (s.pending().has_value()) anchor = *s.pending();
    else {
      auto sel = SelectNext(state, sampler);
      if (sel.stop) { step.stop = true; step.terminal_logprobs = std::move(sel.logprobs); return step; }
      anchor = (int)sel.token; step.selections.push_back(std::move(sel)); s.pending() = anchor;
      if (anchor == think_open_) phase = true; else if (anchor == end_message_) phase = false;
      --max_tokens;
    }
    s.set_thinking(phase);
    if (max_tokens == 0) return step;
    sampling::SamplerState tentative = sampler;
    const std::size_t room = max_context_ - s.position();
    std::uint32_t k = std::min<std::uint32_t>({7U, (std::uint32_t)(max_tokens - 1), room > 2 ? (std::uint32_t)(room - 2) : 0U});
    std::vector<int> drafted;
    if (k > 0) drafted = engine_->DSparkDraft(s.session().dseq(), anchor, (long)s.position(), (int)k);
    k = (std::uint32_t)drafted.size();
    std::vector<std::int32_t> run{(std::int32_t)anchor}; run.insert(run.end(), drafted.begin(), drafted.end());
    std::string error;
    if (!s.session().AppendAll(run, &error)) { s.Invalidate(); throw std::runtime_error("Inkling decode failed: " + error); }
    std::uint32_t consumed = 0; bool stop = false;
    for (std::uint32_t i = 0; i <= k; ++i) {
      const auto logits = s.session().Logits(i); consumed = i;
      const int token = (int)tentative.Sample(logits);
      const bool accepted = i < k && token == drafted[i];
      if (s.stop_at_eos() && token == stop_) { stop = true; s.pending().reset(); break; }
      tentative.Accept((sampling::TokenId)token); emit(token);
      if (accepted) { ++step.draft_accepted_tokens; continue; }
      s.pending() = token; break;
    }
    sampler.CopyDrawStateFrom(tentative);
    const std::size_t kept = consumed + 1;
    s.session().Rewind(kept); s.session().KeepLogitRow(consumed);
    s.set_position(s.session().Position()); s.set_thinking(phase);
    step.draft_rounds = k > 0 ? 1 : 0; step.draft_tokens = k; step.stop = stop;
    return step;
  }

  std::size_t CheckpointPosition(const ext::TextRunnerState& state) const override { return Require(state).position(); }

 private:
  // Encodes an image (cached by content hash: the host may prepare the same request more than once).
  std::shared_ptr<const ImageEmbedding> Embed(const ext::ChatMessage::ImagePart& part) const {
    const std::string hash = crypto::Sha256Hex(*part.bytes);
    { const std::lock_guard lock(image_mutex_); for (const auto& e : image_cache_) if (e->hash == hash) return e; }
    const core::Image img = core::DecodeImage(*part.bytes);
    const PreparedImage prepared = PrepareImage(img.pixels.data(), (int)img.width, (int)img.height);
    auto e = std::make_shared<ImageEmbedding>(); e->hash = hash; std::string error;
    if (!engine_->EncodeImage(prepared.planar, prepared.width, prepared.height, &e->rows, &e->n_tokens, &error)) throw std::runtime_error("Inkling image encoding failed: " + error);
    const std::lock_guard lock(image_mutex_); image_cache_.push_back(e); if (image_cache_.size() > 8) image_cache_.erase(image_cache_.begin()); return e;
  }
  static std::vector<TextRunnerToken> ToTokens(const std::vector<int>& ids) { return std::vector<TextRunnerToken>(ids.begin(), ids.end()); }
  // Text of a generated token: structural markers of the answer are hidden, the thinking open/close markers are shown.
  std::string Piece(int token, bool thinking) const {
    if (token == think_open_) return tokenizer_->decode({token});
    if (token == end_message_) return thinking ? tokenizer_->decode({token}) : std::string();
    if (std::find(hidden_.begin(), hidden_.end(), token) != hidden_.end()) return {};
    return tokenizer_->decode({token});
  }

  std::shared_ptr<::inkling::GpuEngine> engine_;
  std::unique_ptr<::inkling::Tokenizer> tokenizer_;
  std::uint32_t max_context_;
  cache::Identity identity_;
  int stop_{-1}, think_open_{-1}, end_message_{-1};
  std::vector<int> hidden_; int image_pad_{-1};
  mutable std::mutex image_mutex_; mutable std::vector<std::shared_ptr<ImageEmbedding>> image_cache_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateInklingTextRunner(const ext::TextRunnerOptions& options, std::string* error) {
  try {
    if (options.model_path.empty()) { if (error) *error = "Inkling requires a model path"; return nullptr; }
    const bool dspark = options.speculative.backend == ext::TextSpeculativeBackend::kDSpark;
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled && !dspark) { if (error) *error = "Inkling supports dspark speculation only"; return nullptr; }
    if (dspark && options.speculative.draft_model_path.empty()) { if (error) *error = "Inkling dspark speculation requires --dspark-model"; return nullptr; }
    std::uint32_t max_context = options.max_context ? options.max_context : kDefaultContext;
    max_context = std::min<std::uint32_t>(max_context, 1048576);
    std::string load_error;
    auto engine = ::inkling::GpuEngine::Load(options.model_path, (int)max_context, &load_error);
    if (!engine) { if (error) *error = "Failed to load Inkling: " + load_error; return nullptr; }
    if (dspark && !engine->LoadDSpark(options.speculative.draft_model_path, &load_error)) { if (error) *error = "Failed to load the Inkling DSpark drafter: " + load_error; return nullptr; }
    if (!options.vision_model_path.empty() && !engine->LoadVision(options.vision_model_path, &load_error)) { if (error) *error = "Failed to load the Inkling vision projector: " + load_error; return nullptr; }
    ::inkling::Gguf gguf(options.model_path);
    auto tokenizer = std::make_unique<::inkling::Tokenizer>(gguf);
    std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
    if (!options.disk_cache.directory.empty() && fingerprint.empty() && options.reader) fingerprint = core::GgufIdentityHex(*options.reader);
    Log("event=load max_context=" + std::to_string(max_context) + " weights_gib=" + std::to_string(engine->info().weight_bytes >> 30) + " seq_mib=" + std::to_string(engine->seq_bytes() >> 20));
    return std::make_shared<InklingTextRunner>(std::move(engine), std::move(tokenizer), max_context, std::move(fingerprint));
  } catch (const std::exception& e) {
    if (error) *error = std::string("Inkling runner creation failed: ") + e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::inkling
