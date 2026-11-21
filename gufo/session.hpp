// One sequence of the engine as the runner sees it: position, the logits of the last token, and the state needed to
// capture or restore it. Owns an engine sequence for its lifetime.
#pragma once
#include <memory>
#include <mutex>
#include <cstring>
#include <span>
#include <string>
#include <vector>
#include "../engine/engine_api.hpp"

namespace gufo::models::inkling {

class Session {
 public:
  static std::unique_ptr<Session> Create(std::shared_ptr<::inkling::GpuEngine> engine, std::string* error) {
    const int id = engine->CreateSeq(error); if (id < 0) return nullptr;
    return std::unique_ptr<Session>(new Session(std::move(engine), id));
  }
  ~Session() { engine_->DestroySeq(seq_); }
  Session(const Session&) = delete; Session& operator=(const Session&) = delete;

  int seq() const { return seq_; }
  std::size_t Position() const { return position_; }
  std::span<const float> Logits() const { return logits_; }
  ::inkling::GpuEngine& engine() { return *engine_; }

  bool Append(std::span<const std::int32_t> tokens, std::string* error) {
    if (tokens.empty()) return true;
    std::vector<float> lg;
    if (!engine_->Append(seq_, tokens.data(), (int)tokens.size(), (int)position_, &lg, error)) return false;
    logits_ = std::move(lg); position_ += tokens.size(); return true;
  }
  void Reset() { engine_->ResetSeq(seq_); position_ = 0; logits_.clear(); }

  // Private state = convolution histories followed by the logits (f32), the layout of the continuation component.
  std::size_t PrivateBytes() const { return engine_->info().private_bytes + (std::size_t)engine_->info().n_vocab * 4; }
  void CapturePrivate(std::vector<std::uint8_t>* out) const {
    out->assign(PrivateBytes(), 0); engine_->ExportPrivate(seq_, out->data());
    if (!logits_.empty()) std::memcpy(out->data() + engine_->info().private_bytes, logits_.data(), logits_.size() * 4);
  }
  bool SetState(std::size_t position, std::span<const std::uint8_t> blob, std::string* error) {
    if (blob.size() != PrivateBytes()) { if (error) *error = "private state size mismatch"; return false; }
    engine_->ImportPrivate(seq_, blob.data());
    logits_.assign((const float*)(blob.data() + engine_->info().private_bytes), (const float*)(blob.data() + blob.size()));
    position_ = position; return true;
  }
  void TransferRows(int slot, bool is_v, long first, long end, std::uint8_t* host, bool to_host) { engine_->TransferRows(seq_, slot, is_v, first, end, host, to_host); }

 private:
  Session(std::shared_ptr<::inkling::GpuEngine> e, int id) : engine_(std::move(e)), seq_(id) {}
  std::shared_ptr<::inkling::GpuEngine> engine_; int seq_; std::size_t position_{0}; std::vector<float> logits_;
};

}  // namespace gufo::models::inkling
