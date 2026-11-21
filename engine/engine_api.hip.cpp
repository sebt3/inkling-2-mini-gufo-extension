#include "engine_api.hpp"
#include "engine.hpp"

namespace inkling {

struct GpuEngine::Impl { std::unique_ptr<gpu::Engine> e; };

std::unique_ptr<GpuEngine> GpuEngine::Load(const std::string& path, int max_ctx, std::string* error) {
  try {
    std::unique_ptr<GpuEngine> g(new GpuEngine());
    g->impl_ = std::make_unique<Impl>();
    g->impl_->e = std::make_unique<gpu::Engine>(path, max_ctx);
    auto& e = *g->impl_->e;
    g->info_ = {e.c.n_embd, e.c.n_slot(), e.c.n_vocab_unpadded, e.c.n_layer, e.c.n_head_kv, e.c.d_head, e.c.n_swa, gpu::Engine::MAX_T, e.row_bytes(), e.private_bytes(), e.weight_bytes()};
    return g;
  } catch (const std::exception& ex) { if (error) *error = ex.what(); return nullptr; }
}
GpuEngine::~GpuEngine() = default;
bool GpuEngine::slot_is_local(int slot) const { return impl_->e->slot_is_local(slot); }
size_t GpuEngine::seq_bytes() const { return impl_->e->seq_bytes(impl_->e->max_ctx()); }
int GpuEngine::CreateSeq(std::string* error) { try { return impl_->e->create_seq(); } catch (const std::exception& ex) { if (error) *error = ex.what(); return -1; } }
void GpuEngine::DestroySeq(int seq) { impl_->e->destroy_seq(seq); }
void GpuEngine::ResetSeq(int seq) { impl_->e->reset_sequence(seq); }

bool GpuEngine::Append(int seq, const int* tokens, int n, int pos0, std::vector<float>* logits, std::string* error) {
  try {
    for (int o = 0; o < n; o += gpu::Engine::MAX_T) {
      const int m = std::min(gpu::Engine::MAX_T, n - o); gpu::Batch b;
      for (int i = 0; i < m; ++i) { b.tokens.push_back(tokens[o + i]); b.seq.push_back(seq); b.pos.push_back(pos0 + o + i); }
      if (o + m == n) b.want_logits.push_back(m - 1);
      auto r = impl_->e->step(b); if (o + m == n && logits) *logits = std::move(r);
    }
    return true;
  } catch (const std::exception& ex) { if (error) *error = ex.what(); return false; }
}
bool GpuEngine::DecodeBatch(const std::vector<int>& seqs, const std::vector<int>& tokens, const std::vector<int>& pos, std::vector<std::vector<float>>* logits, std::string* error) {
  try {
    gpu::Batch b; b.tokens = tokens; b.seq = seqs; b.pos = pos; for (size_t i = 0; i < seqs.size(); ++i) b.want_logits.push_back((int)i);
    auto r = impl_->e->step(b); const int V = info_.n_vocab; logits->assign(seqs.size(), {});
    for (size_t i = 0; i < seqs.size(); ++i) (*logits)[i].assign(r.begin() + i * V, r.begin() + (i + 1) * V);
    return true;
  } catch (const std::exception& ex) { if (error) *error = ex.what(); return false; }
}
void GpuEngine::TransferRows(int seq, int slot, bool is_v, long first, long end, uint8_t* host, bool to_host) { impl_->e->transfer_rows(seq, slot, is_v, first, end, host, to_host); }
void GpuEngine::ExportPrivate(int seq, uint8_t* out) { impl_->e->export_private(seq, out); }
void GpuEngine::ImportPrivate(int seq, const uint8_t* in) { impl_->e->import_private(seq, in); }

}  // namespace inkling
