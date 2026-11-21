// Oracle tensor dump: run llama.cpp (the oracle's graph, CPU, F32 KV, no flash attention) on a token list and save
// selected tensors, token-major f32, to DIR/<name>.bin. Same layout as `inkling_ref --dump`.
// usage: oracle_dump MODEL.gguf TOKENS.txt DIR
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <regex>
#include <sys/stat.h>
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
static std::string g_dir; static std::regex g_re("^(l_out|ffn_inp|Qcur|Kcur|Vcur|kqv_out|ffn_moe_out|ffn_moe_topk|ffn_moe_logits|ffn_moe_gate_up_scaled|ffn_moe_situ|ffn_moe_down_scaled|ffn_moe_weights_scaled|result_norm|result_output)(-[0-9]+)?$");
static bool cb(struct ggml_tensor* t, bool ask, void*) {
  if (!std::regex_match(t->name, g_re)) return false;
  if (ask) return true;
  // strided copy (views such as the expert top-k are not contiguous)
  const size_t es = ggml_type_size(t->type); std::vector<uint8_t> buf; buf.reserve(ggml_nelements(t) * es);
  for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
    size_t o = buf.size(); buf.resize(o + t->ne[0] * es);
    ggml_backend_tensor_get(t, buf.data() + o, i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1], t->ne[0] * es); }
  if (t->type == GGML_TYPE_I32) { std::vector<float> f(buf.size() / 4); for (size_t i = 0; i < f.size(); ++i) f[i] = (float)((int32_t*)buf.data())[i]; buf.assign((uint8_t*)f.data(), (uint8_t*)f.data() + f.size() * 4); }
  else if (t->type != GGML_TYPE_F32) return true;
  std::string p = g_dir + "/" + t->name + ".bin"; FILE* f = fopen(p.c_str(), "ab"); fwrite(buf.data(), 1, buf.size(), f); fclose(f);
  return true;
}
int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s MODEL TOKENS DIR\n", argv[0]); return 2; }
  g_dir = argv[3]; mkdir(g_dir.c_str(), 0755);
  std::vector<llama_token> toks; { FILE* f = fopen(argv[2], "r"); int t; while (fscanf(f, "%d", &t) == 1) toks.push_back(t); fclose(f); }
  llama_backend_init();
  llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
  llama_model* m = llama_model_load_from_file(argv[1], mp); if (!m) return 1;
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 2048; cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.cb_eval = cb; cp.n_threads = cp.n_threads_batch = 14;
  llama_context* c = llama_init_from_model(m, cp); if (!c) return 1;
  llama_batch b = llama_batch_get_one(toks.data(), (int)toks.size());
  if (llama_decode(c, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
  const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
  FILE* f = fopen((g_dir + "/logits_last.bin").c_str(), "wb"); fwrite(llama_get_logits(c), 4, nv, f); fclose(f);
  fprintf(stderr, "ok, %zu tokens, vocab %d\n", toks.size(), nv); return 0;
}
