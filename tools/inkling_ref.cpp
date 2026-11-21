// CPU reference: feed token ids, write the logits of every position (f32, n_vocab_unpadded each) and print top-k.
// usage: inkling_ref MODEL.gguf TOKENS.txt OUT.bin [--topk N] [--trace]
#include <chrono>
#include <map>
#include "../ref/ref_model.hpp"
using namespace inkling::ref;
int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s MODEL TOKENS OUT [--topk N] [--trace]\n", argv[0]); return 2; }
  int topk = 5; bool tr = false; std::string dump, force;
  for (int i = 4; i < argc; ++i) { std::string a = argv[i]; if (a == "--topk" && i + 1 < argc) topk = atoi(argv[++i]); else if (a == "--trace") tr = true; else if (a == "--dump" && i + 1 < argc) dump = argv[++i]; else if (a == "--force-experts" && i + 1 < argc) force = argv[++i]; }
  std::vector<int> toks; { FILE* f = fopen(argv[2], "r"); if (!f) { perror("tokens"); return 2; } int t; while (fscanf(f, "%d", &t) == 1) toks.push_back(t); fclose(f); }
  Model m(argv[1]);
  std::map<int, std::vector<float>> forced;  // slot -> token-major expert ids from an oracle dump directory
  if (!force.empty()) {
    for (int sl = 0; sl < 48; ++sl) { FILE* f = fopen((force + "/ffn_moe_topk-" + std::to_string(sl) + ".bin").c_str(), "rb"); if (!f) continue;
      fseek(f, 0, SEEK_END); long n = ftell(f) / 4; fseek(f, 0, SEEK_SET); std::vector<float> v(n); (void)!fread(v.data(), 4, n, f); fclose(f); forced[sl] = std::move(v); }
    m.force_experts = [&forced](int slot, int pos, std::vector<int>& out) { auto it = forced.find(slot); if (it == forced.end()) return false;
      for (int i = 0; i < 8; ++i) out.push_back((int)it->second[(size_t)pos * 8 + i]); return true; };
  }
  if (tr) m.trace = [](const char* n, int slot, const Vec& v) {
    double s = 0, a = 0, su = 0; for (float x : v) { s += (double)x * x; a += std::fabs(x); su += x; }
    fprintf(stderr, "TRACE %s-%d n=%zu sum=%.6f rms=%.6g first=%.6g %.6g %.6g\n", n, slot, v.size(), su, std::sqrt(s / v.size()), v[0], v.size()>1?v[1]:0.f, v.size()>2?v[2]:0.f);
  };
  if (!dump.empty()) { std::string d = dump; (void)!system(("mkdir -p " + d).c_str()); m.trace = [d](const char* n, int slot, const Vec& v) {
    std::string name = n; if (slot >= 0) name += "-" + std::to_string(slot);
    FILE* f = fopen((d + "/" + name + ".bin").c_str(), "ab"); fwrite(v.data(), 4, v.size(), f); fclose(f); }; }
  FILE* out = fopen(argv[3], "wb");
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < toks.size(); ++i) {
    Vec lg = m.forward(toks[i]);
    fwrite(lg.data(), 4, lg.size(), out);
    std::vector<int> idx(lg.size()); std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    float mx = lg[idx[0]]; double z = 0; for (float x : lg) z += std::exp(x - mx);
    printf("pos %zu tok %d ->", i, toks[i]);
    for (int k = 0; k < topk; ++k) printf(" %d(%.3f)", idx[k], (double)(lg[idx[k]] - mx) - std::log(z));
    printf("\n"); fflush(stdout);
  }
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "%zu tokens in %.1fs (%.2f s/token)\n", toks.size(), dt, dt / toks.size());
  fclose(out);
}
