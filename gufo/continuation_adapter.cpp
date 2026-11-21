#include "continuation_adapter.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace gufo::models::inkling {
namespace {
std::vector<bool> LocalSlots(const ::inkling::GpuEngine& e) {
  std::vector<bool> v; for (int s = 0; s < e.info().n_slot; ++s) v.push_back(e.slot_is_local(s)); return v;
}
}  // namespace

class InklingContinuationAdapter::State final : public cache::Slot {
 public:
  State(InklingContinuationAdapter& adapter, cache::MutationGuard& guard, std::unique_ptr<Session> session) : adapter(adapter), guard(guard), session(std::move(session)) {}

  struct Job { std::mutex mutex; std::optional<cache::TransferResult> result; };
  class Signal final : public cache::CompletionSignal {
   public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return true; }
    cache::TransferResult Wait() noexcept override { return *job_->result; }
   private:
    std::shared_ptr<Job> job_;
  };
  static cache::Completion Settled(bool success) {
    auto job = std::make_shared<Job>(); job->result = success ? cache::TransferResult::kSucceeded : cache::TransferResult::kFailed;
    return cache::Completion(std::make_unique<Signal>(std::move(job)));
  }

  bool IsValid() const noexcept override { return !restoring && !failed; }
  void Cover(std::size_t component, cache::Rows first, cache::Rows end) { const std::lock_guard lock(mutex); coverage[component].emplace_back(first, end); }
  bool Covers(std::size_t component, cache::Rows first, cache::Rows end) {
    const std::lock_guard lock(mutex); auto ranges = coverage[component]; std::sort(ranges.begin(), ranges.end()); cache::Rows next = first;
    for (const auto& [from, to] : ranges) { if (from != next) return false; next = to; }
    return next == end;
  }
  void ResetRestore() { const std::lock_guard lock(mutex); coverage.clear(); blob.clear(); blob_loaded.clear(); expected_position = 0; }

  InklingContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false}, failed{false};
  std::mutex mutex;
  std::vector<std::vector<std::pair<cache::Rows, cache::Rows>>> coverage;
  std::vector<std::uint8_t> blob;                                  // private state assembled from pieces
  std::vector<std::pair<std::size_t, std::size_t>> blob_loaded;
  cache::Rows expected_position{0};
};

InklingContinuationAdapter::InklingContinuationAdapter(std::shared_ptr<::inkling::GpuEngine> engine, std::uint32_t context, cache::Identity identity)
    : engine_(std::move(engine)), context_(context), identity_(std::move(identity)),
      layout_(LocalSlots(*engine_), engine_->info().row_bytes, (cache::Rows)engine_->info().n_swa,
              engine_->info().private_bytes + (std::size_t)engine_->info().n_vocab * 4, context) {
  if (identity_.empty()) throw std::invalid_argument("Inkling adapter requires an identity");
  const auto put = [&](std::uint32_t v) { for (unsigned i = 0; i < 4; ++i) identity_.push_back(static_cast<std::uint8_t>(v >> (8 * i))); };
  put(ContinuationLayout::kLayoutVersion); put(context_);
}
InklingContinuationAdapter::~InklingContinuationAdapter() = default;

InklingContinuationAdapter::State& InklingContinuationAdapter::As(cache::Slot& slot) const {
  auto& s = dynamic_cast<State&>(slot);
  if (s.adapter.engine_.get() != engine_.get()) throw std::invalid_argument("slot belongs to another Inkling adapter");
  return s;
}
Session& InklingContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& s = As(slot); if (!s.IsValid()) throw std::logic_error("Inkling slot is not valid"); return *s.session;
}

std::vector<cache::BackingClass> InklingContinuationAdapter::BackingPlan(std::size_t ram_bytes) const {
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram_bytes > (4ULL << 20) ? (ram_bytes - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header) return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks) return {};
  return {{page, blocks, header + blocks * per_block}};
}
cache::Capabilities InklingContinuationAdapter::GetCapabilities() const { return {.continuation = true, .persistent_encoding = true}; }
std::span<const cache::ComponentDescriptor> InklingContinuationAdapter::Components() const { return layout_.descriptors(); }
cache::Identity InklingContinuationAdapter::CompatibilityIdentity() const { return identity_; }

std::unique_ptr<cache::Slot> InklingContinuationAdapter::CreateSlot(cache::MutationGuard& guard) {
  std::string error; auto session = Session::Create(engine_, &error);
  if (!session) throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> InklingContinuationAdapter::Positions(const cache::Slot& slot) const {
  const auto& s = As(const_cast<cache::Slot&>(slot)); if (!s.IsValid()) throw std::logic_error("cannot query an invalid Inkling slot");
  return layout_.Positions(s.session->Position());
}
std::vector<cache::Rows> InklingContinuationAdapter::PlanPrefill(const cache::Slot& slot, cache::Rows first, cache::Rows end) const {
  const auto& s = As(const_cast<cache::Slot&>(slot));
  if (!s.IsValid() || first > end || end > context_) throw std::invalid_argument("invalid Inkling prefill plan");
  std::vector<cache::Rows> result; const cache::Rows capacity = (cache::Rows)engine_->info().max_batch;  // multiple of the 256-row chunk
  while (first < end) { first += std::min(end - first, capacity); result.push_back(first); }
  return result;
}

void InklingContinuationAdapter::BeginRestore(cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& s = As(slot); if (!s.IsValid()) throw std::logic_error("Inkling restore requires a valid slot");
  const auto& components = layout_.components();
  if (positions.size() != components.size()) throw std::invalid_argument("incomplete Inkling restore positions");
  const cache::Rows position = positions.front().valid_rows;
  for (std::size_t i = 0; i < positions.size(); ++i)
    if (positions[i].id.value != i + 1 || positions[i].valid_rows != position) throw std::invalid_argument("inconsistent Inkling restore positions");
  if (position == 0 || position > context_) throw std::invalid_argument("Inkling frontier is out of range");
  for (const auto& c : components) if (cache::HasRows(c.descriptor.kind)) s.guard.BeforeOverwrite(c.descriptor.id, layout_.FirstRetainedRow(c, position), position);
  s.ResetRestore(); s.coverage.resize(components.size()); s.expected_position = position; s.failed = false; s.restoring = true;
}

cache::Completion InklingContinuationAdapter::CapturePrivate(const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes, cache::Stream& stream) { return CapturePrivatePiece(slot, id, 0, bytes, stream); }
cache::Completion InklingContinuationAdapter::CapturePrivatePiece(const cache::Slot& slot, cache::ComponentId id, std::size_t offset, std::span<std::byte> bytes, cache::Stream&) {
  if (id.value != layout_.private_id().value) throw std::invalid_argument("Inkling private state id");
  auto& s = As(const_cast<cache::Slot&>(slot));
  if (!s.IsValid() || s.session->Position() == 0) throw std::logic_error("cannot capture an empty or invalid Inkling slot");
  const auto total = layout_.at(id).descriptor.state_bytes;
  if (offset > total || bytes.size() > total - offset) throw std::invalid_argument("invalid Inkling private piece");
  std::vector<std::uint8_t> blob; s.session->CapturePrivate(&blob);
  std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(blob.data()) + offset, bytes.size());
  return State::Settled(true);
}
cache::Completion InklingContinuationAdapter::LoadPrivate(cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes, cache::Stream& stream) { return LoadPrivatePiece(slot, id, 0, bytes, stream); }
cache::Completion InklingContinuationAdapter::LoadPrivatePiece(cache::Slot& slot, cache::ComponentId id, std::size_t offset, std::span<const std::byte> bytes, cache::Stream&) {
  if (id.value != layout_.private_id().value) throw std::invalid_argument("Inkling private state id");
  auto& s = As(slot); if (!s.restoring || s.failed) throw std::logic_error("Inkling load requires an unfailed restore");
  const auto total = layout_.at(id).descriptor.state_bytes;
  if (offset > total || bytes.size() > total - offset) throw std::invalid_argument("invalid Inkling private piece");
  const std::lock_guard lock(s.mutex);
  if (s.blob.empty()) s.blob.resize(total);
  std::memcpy(s.blob.data() + offset, bytes.data(), bytes.size()); s.blob_loaded.emplace_back(offset, offset + bytes.size());
  return State::Settled(true);
}

cache::Completion InklingContinuationAdapter::CopyRowsOut(const cache::Slot& slot, cache::ComponentId id, cache::Rows first, cache::Rows end, std::span<std::byte> bytes, cache::Stream&) {
  auto& s = As(const_cast<cache::Slot&>(slot)); const auto& c = layout_.at(id);
  if (!cache::HasRows(c.descriptor.kind)) throw std::invalid_argument("Inkling component has no rows");
  if (!s.IsValid()) throw std::logic_error("cannot read an invalid Inkling slot");
  const auto row = layout_.row_bytes();
  if (first > end || end > s.session->Position() || bytes.size() != (end - first) * row) throw std::invalid_argument("invalid Inkling row capture");
  s.session->TransferRows((int)c.slot, c.part == ContinuationLayout::Part::kValue, (long)first, (long)end, reinterpret_cast<std::uint8_t*>(bytes.data()), true);
  return State::Settled(true);
}
cache::Completion InklingContinuationAdapter::CopyRowsIn(cache::Slot& slot, cache::ComponentId id, cache::Rows first, cache::Rows end, std::span<const std::byte> bytes, cache::Stream&) {
  auto& s = As(slot); if (!s.restoring || s.failed) throw std::logic_error("Inkling load requires an unfailed restore");
  try {
    const auto& c = layout_.at(id);
    if (!cache::HasRows(c.descriptor.kind)) throw std::invalid_argument("Inkling component has no rows");
    const auto row = layout_.row_bytes(); const auto lowest = layout_.FirstRetainedRow(c, s.expected_position);
    if (first > end || first < lowest || end > s.expected_position || bytes.size() != (end - first) * row) throw std::invalid_argument("invalid Inkling row restore");
    s.session->TransferRows((int)c.slot, c.part == ContinuationLayout::Part::kValue, (long)first, (long)end, const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(bytes.data())), false);
    s.Cover(id.value - 1, first, end); return State::Settled(true);
  } catch (...) { s.failed = true; throw; }
}

bool InklingContinuationAdapter::Validate(cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& s = As(slot); if (!s.restoring) throw std::logic_error("Inkling validation requires an active restore");
  const auto refuse = [&] { s.failed = true; return false; };
  const auto& components = layout_.components();
  if (s.failed || positions.size() != components.size()) return refuse();
  for (std::size_t i = 0; i < positions.size(); ++i) if (positions[i].id.value != i + 1 || positions[i].valid_rows != s.expected_position) return refuse();
  const auto position = s.expected_position;
  for (std::size_t i = 0; i < components.size(); ++i) {
    if (!cache::HasRows(components[i].descriptor.kind)) continue;
    if (!s.Covers(i, layout_.FirstRetainedRow(components[i], position), position)) return refuse();
  }
  std::vector<std::uint8_t> blob;
  {
    const std::lock_guard lock(s.mutex); auto ranges = s.blob_loaded; std::sort(ranges.begin(), ranges.end()); std::size_t next = 0;
    for (const auto& [from, to] : ranges) { if (from != next) return refuse(); next = to; }
    const auto total = layout_.at(layout_.private_id()).descriptor.state_bytes;
    if (next != total || s.blob.size() != total) return refuse();
    blob = s.blob;
  }
  std::string error;
  if (!s.session->SetState((std::size_t)position, blob, &error)) return refuse();
  if (s.session->Position() != position) return refuse();
  s.restoring = false; s.ResetRestore(); return true;
}
bool InklingContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try { auto& s = As(slot); s.session->Reset(); s.failed = false; s.restoring = false; s.ResetRestore(); s.guard.AfterReset(); return true; } catch (...) { return false; }
}

}  // namespace gufo::models::inkling
