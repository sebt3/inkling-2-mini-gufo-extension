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
    auto s = std::unique_ptr<Session>(new Session(std::move(engine), id));
    if (s->engine_->has_dspark()) { s->dseq_ = s->engine_->DSparkCreateSeq(); if (s->dseq_ < 0) { if (error) *error = "no free drafter sequence"; return nullptr; } }
    return s;
  }
  ~Session() { if (dseq_ >= 0) engine_->DSparkDestroySeq(dseq_); engine_->DestroySeq(seq_); }
  Session(const Session&) = delete; Session& operator=(const Session&) = delete;

  int seq() const { return seq_; }
  std::size_t Position() const { return position_; }
  std::span<const float> Logits() const { return logits_; }
  ::inkling::GpuEngine& engine() { return *engine_; }

  int dseq() const { return dseq_; }
  bool speculative() const { return dseq_ >= 0; }
  bool Append(std::span<const std::int32_t> tokens, std::string* error, const int* embd_mask = nullptr, const float* embd = nullptr) {
    if (tokens.empty()) return true;
    const bool feats = dseq_ >= 0 && tokens.size() <= (std::size_t)engine_->info().max_batch;
    std::vector<float> lg;
    if (!engine_->Append(seq_, tokens.data(), (int)tokens.size(), (int)position_, &lg, error, embd_mask, embd, feats)) return false;
    if (feats) engine_->DSparkInject(dseq_, (long)position_, (int)tokens.size());
    else if (dseq_ >= 0) engine_->DSparkReset(dseq_, (long)(position_ + tokens.size()));
    logits_ = std::move(lg); position_ += tokens.size(); return true;
  }
  // Verification batch (<= 32 rows): the logits of every row are kept; Rewind(keep) drops the rows after the first `keep`.
  bool AppendAll(std::span<const std::int32_t> tokens, std::string* error) {
    if (!engine_->AppendRows(seq_, tokens.data(), (int)tokens.size(), (int)position_, &rows_, dseq_ >= 0, true, error)) return false;
    batch_pos_ = position_; batch_n_ = tokens.size(); position_ += tokens.size(); return true;
  }
  std::span<const float> Logits(std::size_t row) const { return rows_[row]; }
  void KeepLogitRow(std::size_t row) { logits_ = rows_[row]; }
  // keep the first `keep` tokens of the last AppendAll: roll the convolution state back, hand their features to the drafter
  void Rewind(std::size_t keep) {
    if (keep < batch_n_) engine_->RollbackConv(seq_, (int)keep);
    if (dseq_ >= 0) engine_->DSparkInject(dseq_, (long)batch_pos_, (int)keep);
    position_ = batch_pos_ + keep;
  }
  // result of a batched one-token decode step (see GpuEngine::DecodeBatch): the token is already in the caches
  void CommitDecoded(std::vector<float>&& logits) { logits_ = std::move(logits); ++position_; }
  void Reset() { engine_->ResetSeq(seq_); position_ = 0; logits_.clear(); if (dseq_ >= 0) engine_->DSparkReset(dseq_, 0); }

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
    position_ = position; if (dseq_ >= 0) engine_->DSparkReset(dseq_, (long)position); return true;
  }
  void TransferRows(int slot, bool is_v, long first, long end, std::uint8_t* host, bool to_host) { engine_->TransferRows(seq_, slot, is_v, first, end, host, to_host); }

 private:
  Session(std::shared_ptr<::inkling::GpuEngine> e, int id) : engine_(std::move(e)), seq_(id) {}
  std::shared_ptr<::inkling::GpuEngine> engine_; int seq_; int dseq_{-1}; std::size_t position_{0}, batch_pos_{0}, batch_n_{0}; std::vector<float> logits_; std::vector<std::vector<float>> rows_;
};

}  // namespace gufo::models::inkling
