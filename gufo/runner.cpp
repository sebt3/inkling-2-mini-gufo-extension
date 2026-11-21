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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../core/gguf.hpp"
#include "../host/chat_template.hpp"
#include "../host/tokenizer.hpp"
#include "continuation_adapter.hpp"
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

class InklingState final : public ext::TextRunnerState {
 public:
  InklingState(Session& session, cache::Adapter& adapter, cache::Slot& slot) : session_(&session), adapter_(&adapter), slot_(&slot) {}
  void Invalidate() noexcept override { position_ = 0; thinking_ = false; (void)adapter_->Invalidate(*slot_); }
  Session& session() const { return *session_; }
  std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t p) noexcept { position_ = p; }
  bool thinking() const noexcept { return thinking_; }
  void set_thinking(bool v) noexcept { thinking_ = v; }
 private:
  Session* session_; cache::Adapter* adapter_; cache::Slot* slot_; std::size_t position_{0}; bool thinking_{false};
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
    hidden_ = {tokenizer_->token_id("<|message_model|>"), tokenizer_->token_id("<|content_text|>"), stop_};
  }

  ext::TextRunnerDescriptor Descriptor() const override {
    sampling::TextSamplingValues v; v.temperature = 1.0F; v.top_k = 0; v.top_p = 1.0F; v.min_p = 0.0F;
    return {
        .model_id = "Inkling-2-Mini",
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities = ext::TextRunnerCapabilities{.incremental_prefill = true, .final_token_advance_required = false, .incremental_text_is_exact = true, .prefix_reuse = true},
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

  std::vector<ext::TextExecutionPlan> SupportedPlans() const override { return {{.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}}; }

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
    if (!request.tools.empty() && request.tool_choice != ext::ChatRequest::ToolChoice::kNone) throw std::invalid_argument("Inkling tool calls are not supported yet");
    std::vector<::inkling::Message> messages;
    for (const auto& m : request.messages) {
      if (!m.images.empty()) throw std::invalid_argument("Inkling image input is not supported yet");
      if (!m.tool_calls.empty()) throw std::invalid_argument("Inkling tool calls are not supported yet");
      ::inkling::Message out = ::inkling::Message::make(RoleName(m.role), m.content + m.framing_suffix);
      out.reasoning_content = m.thought; out.name = m.name; messages.push_back(std::move(out));
    }
    ::inkling::Options options;
    if (request.reasoning.enabled.has_value() && !*request.reasoning.enabled) options.reasoning_effort = "none";
    else if (request.reasoning.effort.has_value()) options.reasoning_effort = EffortName(*request.reasoning.effort);
    return ToTokens(tokenizer_->encode(::inkling::render_prompt(messages, options), true));
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::vector<int> ids(tokens.begin(), tokens.end()); return tokenizer_->decode(ids);
  }

  // Reasoning arrives as <|message_model|><|content_thinking|>...<|end_message|>; the role and content-kind markers of the
  // answer stay out of the text, the two thinking markers are kept so the host can split reasoning from the answer.
  ReasoningTags ReasoningBudgetTags() const override { return {"<|content_thinking|>", "<|end_message|>", {"<|end_message|>"}}; }
  ReasoningMarkup ReasoningDelimiters() const override { return {"<|content_thinking|>", "<|end_message|>"}; }

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
    std::string error;
    if (!s.session().Append(chunk, &error)) { s.Invalidate(); throw std::runtime_error("Inkling prefill failed: " + error); }
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
    auto& s = Require(state);
    const std::int32_t input = (std::int32_t)token; std::string error;
    if (!s.session().Append({&input, 1}, &error)) { s.Invalidate(); throw std::runtime_error("Inkling decode failed: " + error); }
    if ((int)token == think_open_) s.set_thinking(true); else if ((int)token == end_message_) s.set_thinking(false);
    s.set_position(s.position() + 1);
  }

  std::size_t CheckpointPosition(const ext::TextRunnerState& state) const override { return Require(state).position(); }

 private:
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
  std::vector<int> hidden_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateInklingTextRunner(const ext::TextRunnerOptions& options, std::string* error) {
  try {
    if (options.model_path.empty()) { if (error) *error = "Inkling requires a model path"; return nullptr; }
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled) { if (error) *error = "Inkling speculation is not supported yet"; return nullptr; }
    if (!options.vision_model_path.empty()) { if (error) *error = "Inkling vision is not supported yet"; return nullptr; }
    std::uint32_t max_context = options.max_context ? options.max_context : kDefaultContext;
    max_context = std::min<std::uint32_t>(max_context, 1048576);
    std::string load_error;
    auto engine = ::inkling::GpuEngine::Load(options.model_path, (int)max_context, &load_error);
    if (!engine) { if (error) *error = "Failed to load Inkling: " + load_error; return nullptr; }
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
