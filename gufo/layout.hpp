// Geometry of the Inkling continuation cache: the K and V rows of the 48 cache slots as row components (append rows
// for a global slot, window rows for a local one) plus one private component holding the convolution histories of all
// slots and the logits of the last token. Pure host arithmetic.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "src/cache/types.hpp"

namespace gufo::models::inkling {

class ContinuationLayout {
 public:
  enum class Part : std::uint8_t { kKey, kValue, kPrivate };
  struct Component { cache::ComponentDescriptor descriptor; Part part{Part::kKey}; std::uint32_t slot{0}; };

  static constexpr cache::Rows kRowsPerChunk = 256;
  static constexpr std::uint32_t kLayoutVersion = 1;

  // is_local[s]: slot s is a sliding-window slot; window: its window in rows.
  ContinuationLayout(const std::vector<bool>& is_local, std::size_t row_bytes, cache::Rows window, std::size_t private_state_bytes, std::uint32_t context)
      : context_(context), row_bytes_(row_bytes) {
    if (!context || is_local.empty() || !row_bytes || !private_state_bytes) throw std::invalid_argument("invalid Inkling continuation geometry");
    const auto add = [&](Part part, std::uint32_t slot, cache::ComponentKind kind, std::size_t row, std::size_t state, cache::Rows win) {
      const auto id = static_cast<std::uint32_t>(components_.size() + 1);
      components_.push_back({{{id}, kLayoutVersion, kind, row, kind == cache::ComponentKind::kPrivateState ? 0 : kRowsPerChunk, state, win}, part, slot});
    };
    for (std::uint32_t s = 0; s < is_local.size(); ++s) {
      const auto kind = is_local[s] ? cache::ComponentKind::kWindowRows : cache::ComponentKind::kAppendRows;
      const cache::Rows win = is_local[s] ? window : 0;
      add(Part::kKey, s, kind, row_bytes_, 0, win);
      add(Part::kValue, s, kind, row_bytes_, 0, win);
    }
    private_id_ = static_cast<std::uint32_t>(components_.size() + 1);
    add(Part::kPrivate, 0, cache::ComponentKind::kPrivateState, 0, private_state_bytes, 0);
    for (const auto& c : components_) descriptors_.push_back(c.descriptor);
  }
  std::uint32_t context() const noexcept { return context_; }
  std::size_t row_bytes() const noexcept { return row_bytes_; }
  const std::vector<Component>& components() const noexcept { return components_; }
  const std::vector<cache::ComponentDescriptor>& descriptors() const noexcept { return descriptors_; }
  const Component& at(cache::ComponentId id) const {
    if (id.value == 0 || id.value > components_.size()) throw std::invalid_argument("invalid Inkling component id");
    return components_[id.value - 1];
  }
  cache::ComponentId private_id() const noexcept { return {private_id_}; }
  std::vector<cache::ComponentPosition> Positions(cache::Rows position) const {
    std::vector<cache::ComponentPosition> p; p.reserve(components_.size());
    for (const auto& c : components_) p.push_back({c.descriptor.id, position});
    return p;
  }
  cache::Rows FirstRetainedRow(const Component& c, cache::Rows position) const { return cache::FirstChunkRow(c.descriptor, position); }

 private:
  std::uint32_t context_; std::size_t row_bytes_; std::uint32_t private_id_{0};
  std::vector<Component> components_; std::vector<cache::ComponentDescriptor> descriptors_;
};

}  // namespace gufo::models::inkling
