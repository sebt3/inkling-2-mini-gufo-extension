// GPU engine test harness: feed token ids (in chunks), write logits and optional per-tensor dumps, same layout as inkling_ref.
// usage: inkling_gpu MODEL TOKENS OUT.bin [--chunk N] [--ctx N] [--dump DIR] [--topk N] [--force-experts DIR]
#include <chrono>
#include <map>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "../engine/engine.hpp"
using namespace inkling::gpu;
int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s MODEL TOKENS OUT [--chunk N] [--ctx N] [--dump DIR] [--topk N]\n", argv[0]); return 2; }
  int chunk = 1, ctx = 8192, topk = 3; std::string dump, force;
  for (int i = 4; i < argc; ++i) { std::string a = argv[i]; if (a == "--chunk") chunk = atoi(argv[++i]); else if (a == "--ctx") ctx = atoi(argv[++i]); else if (a == "--dump") dump = argv[++i]; else if (a == "--topk") topk = atoi(argv[++i]); else if (a == "--force-experts") force = argv[++i]; }
  std::vector<int> toks; { FILE* f = fopen(argv[2], "r"); int t; while (fscanf(f, "%d", &t) == 1) toks.push_back(t); fclose(f); }
  auto t0 = std::chrono::steady_clock::now();
  Engine eng(argv[1], ctx);
  fprintf(stderr, "load+upload %.1fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  if (!dump.empty()) { (void)!system(("mkdir -p " + dump).c_str());
    eng.dump = [dump](const char* n, int slot, const float* dev, int T, int width) { std::vector<float> h((size_t)T * width); HIP_CHECK(hipMemcpy(h.data(), dev, h.size() * 4, hipMemcpyDeviceToHost));
      std::string name = n; if (slot >= 0) name += "-" + std::to_string(slot); FILE* f = fopen((dump + "/" + name + ".bin").c_str(), "ab"); fwrite(h.data(), 4, h.size(), f); fclose(f); }; }
  std::map<int, std::vector<float>> forced;
  if (!force.empty()) {
    for (int sl = 0; sl < 48; ++sl) { FILE* f = fopen((force + "/ffn_moe_topk-" + std::to_string(sl) + ".bin").c_str(), "rb"); if (!f) continue;
      fseek(f, 0, SEEK_END); long n = ftell(f) / 4; fseek(f, 0, SEEK_SET); std::vector<float> v(n); (void)!fread(v.data(), 4, n, f); fclose(f); forced[sl] = std::move(v); }
    eng.force_experts = [&forced](int slot, const std::vector<int>& pos, std::vector<int>& out) { auto it = forced.find(slot); if (it == forced.end()) return false;
      for (int p : pos) for (int i = 0; i < 8; ++i) out.push_back((int)it->second[(size_t)p * 8 + i]); return true; };
  }
  FILE* out = fopen(argv[3], "wb"); const int sid = eng.create_seq(); if (sid != 0) return 3;
  t0 = std::chrono::steady_clock::now(); size_t done = 0;
  while (done < toks.size()) {
    Batch b; const size_t n = std::min<size_t>(chunk, toks.size() - done);
    for (size_t i = 0; i < n; ++i) { b.tokens.push_back(toks[done + i]); b.seq.push_back(0); b.pos.push_back((int)(done + i)); }
    const size_t nl = std::min<size_t>(n, 32); for (size_t i = n - nl; i < n; ++i) b.want_logits.push_back((int)i);
    std::vector<float> lg = eng.step(b); const int V = eng.c.n_vocab_unpadded; fwrite(lg.data(), 4, lg.size(), out);
    for (size_t k = 0; k < nl; ++k) {
      const float* l = lg.data() + k * V; std::vector<int> idx(V); std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int c) { return l[a] > l[c]; });
      float mx = l[idx[0]]; double z = 0; for (int i = 0; i < V; ++i) z += std::exp(l[i] - mx);
      printf("pos %zu tok %d ->", done + n - nl + k, toks[done + n - nl + k]); for (int j = 0; j < topk; ++j) printf(" %d(%.3f)", idx[j], (double)(l[idx[j]] - mx) - std::log(z)); printf("\n");
    }
    done += n;
  }
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "%zu tokens in %.2fs (%.1f tok/s)\n", toks.size(), dt, toks.size() / dt); fclose(out);
}
