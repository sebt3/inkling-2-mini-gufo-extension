// Prefix-continuation adapter for Inkling sequences (see PLAN.md): K and V rows of every cache slot as row
// components (append rows for global slots, window rows for local ones) plus the convolution histories and the last
// logits as private state.
#pragma once
#include <cstdint>
#include <gufo/ext/adapter.hpp>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include "layout.hpp"
#include "session.hpp"

namespace gufo::models::inkling {

class InklingContinuationAdapter final : public cache::Adapter {
 public:
  InklingContinuationAdapter(std::shared_ptr<::inkling::GpuEngine> engine, std::uint32_t context, cache::Identity identity);
  ~InklingContinuationAdapter() override;

  Session& GetSession(cache::Slot&);

  std::vector<cache::BackingClass> BackingPlan(std::size_t ram_bytes) const override;
  cache::Capabilities GetCapabilities() const override;
  std::span<const cache::ComponentDescriptor> Components() const override;
  cache::Identity CompatibilityIdentity() const override;
  std::unique_ptr<cache::Slot> CreateSlot(cache::MutationGuard&) override;
  std::vector<cache::ComponentPosition> Positions(const cache::Slot&) const override;
  std::vector<cache::Rows> PlanPrefill(const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&, std::span<const cache::ComponentPosition>) override;
  cache::Completion CapturePrivate(const cache::Slot&, cache::ComponentId, std::span<std::byte>, cache::Stream&) override;
  cache::Completion CapturePrivatePiece(const cache::Slot&, cache::ComponentId, std::size_t, std::span<std::byte>, cache::Stream&) override;
  cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId, std::span<const std::byte>, cache::Stream&) override;
  cache::Completion LoadPrivatePiece(cache::Slot&, cache::ComponentId, std::size_t, std::span<const std::byte>, cache::Stream&) override;
  cache::Completion CopyRowsOut(const cache::Slot&, cache::ComponentId, cache::Rows first, cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId, cache::Rows first, cache::Rows end, std::span<const std::byte>, cache::Stream&) override;
  bool Validate(cache::Slot&, std::span<const cache::ComponentPosition>) override;
  bool Invalidate(cache::Slot&) noexcept override;

 private:
  class State;
  State& As(cache::Slot&) const;
  std::shared_ptr<::inkling::GpuEngine> engine_;
  std::uint32_t context_;
  cache::Identity identity_;
  ContinuationLayout layout_;
};

}  // namespace gufo::models::inkling
