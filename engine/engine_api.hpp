// Plain C++ facade over the HIP engine (no HIP types), for the Gufo adapter layer and tests.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inkling {

struct EngineInfo {
  int n_embd, n_slot, n_vocab, n_layer, n_head_kv, d_head, n_swa, max_batch;
  size_t row_bytes;       // bytes of one K or V row of a cache slot
  size_t private_bytes;   // convolution histories of one sequence
  size_t weight_bytes;
};

class GpuEngine {
 public:
  // Loads and uploads the model. `max_ctx` is the context length of every sequence.
  static std::unique_ptr<GpuEngine> Load(const std::string& path, int max_ctx, std::string* error);
  ~GpuEngine();
  GpuEngine(const GpuEngine&) = delete;
  GpuEngine& operator=(const GpuEngine&) = delete;

  const EngineInfo& info() const { return info_; }
  bool slot_is_local(int slot) const;
  size_t seq_bytes() const;  // device bytes of one sequence

  int CreateSeq(std::string* error);   // -1 on failure
  void DestroySeq(int seq);
  void ResetSeq(int seq);

  // Feed `n` consecutive tokens of one sequence (positions pos0..pos0+n-1), in chunks; returns the logits of the last.
  // embd_mask/embd (optional): for tokens with embd_mask[i] != 0 the row embd[i*n_embd ..) replaces the token embedding (images).
  bool Append(int seq, const int* tokens, int n, int pos0, std::vector<float>* logits, std::string* error, const int* embd_mask = nullptr, const float* embd = nullptr);
  // vision: load the projector (mmproj) once, then encode planar normalised [3][H][W] images (W and H multiples of 40)
  bool LoadVision(const std::string& mmproj_path, std::string* error);
  bool has_vision() const;
  int vision_dim() const;
  bool EncodeImage(const std::vector<float>& planar, int W, int H, std::vector<float>* rows, int* n_tokens, std::string* error);
  // One token for each of several sequences in a single pass; logits[i] belongs to seqs[i].
  bool DecodeBatch(const std::vector<int>& seqs, const std::vector<int>& tokens, const std::vector<int>& pos, std::vector<std::vector<float>>* logits, std::string* error);

  // K or V rows [first, end) (absolute row numbers) of a cache slot <-> host buffer of (end-first)*row_bytes
  void TransferRows(int seq, int slot, bool is_v, long first, long end, uint8_t* host, bool to_host);
  void ExportPrivate(int seq, uint8_t* out);        // info().private_bytes
  void ImportPrivate(int seq, const uint8_t* in);

 private:
  GpuEngine() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  EngineInfo info_{};
};

}  // namespace inkling
