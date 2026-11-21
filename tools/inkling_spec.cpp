// DSpark test: greedy decoding with and without speculation must give the same tokens; reports acceptance and speed.
// usage: inkling_spec MODEL.gguf DSPARK.gguf TOKENS.txt [--n 64] [--ctx 8192]
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include "../engine/engine_api.hpp"
using namespace inkling;
static int argmax(const std::vector<float>& l) { int b = 0; for (size_t i = 1; i < l.size(); ++i) if (l[i] > l[b]) b = (int)i; return b; }
int main(int argc, char** argv) {
  if (argc < 4) return 2; int N = 64, ctx = 8192;
  for (int i = 4; i < argc; ++i) { std::string a = argv[i]; if (a == "--n") N = atoi(argv[++i]); else if (a == "--ctx") ctx = atoi(argv[++i]); }
  std::vector<int> prompt; { FILE* f = fopen(argv[3], "r"); int t; while (fscanf(f, "%d", &t) == 1) prompt.push_back(t); fclose(f); }
  std::string err; auto eng = GpuEngine::Load(argv[1], ctx, &err); if (!eng) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
  if (!eng->LoadDSpark(argv[2], &err)) { fprintf(stderr, "dspark: %s\n", err.c_str()); return 1; }
  auto now = [] { return std::chrono::steady_clock::now(); }; auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
  // baseline
  std::vector<int> base; { int s = eng->CreateSeq(&err); std::vector<float> lg; eng->Append(s, prompt.data(), (int)prompt.size(), 0, &lg, &err); int pos = (int)prompt.size(); auto t0 = now();
    for (int i = 0; i < N; ++i) { int t = argmax(lg); base.push_back(t); eng->Append(s, &t, 1, pos++, &lg, &err); }
    printf("baseline: %.2f tok/s\n", N / sec(t0, now())); eng->DestroySeq(s); }
  // speculative
  std::vector<int> spec; { int s = eng->CreateSeq(&err); int d = eng->DSparkCreateSeq(); std::vector<float> lg; int pos = 0;
    for (int o = 0; o < (int)prompt.size(); o += 1024) { int m = std::min<int>(1024, (int)prompt.size() - o); eng->Append(s, prompt.data() + o, m, pos, &lg, &err, nullptr, nullptr, true); eng->DSparkInject(d, pos, m); pos += m; }
    int anchor = argmax(lg); spec.push_back(anchor); long rounds = 0, drafted = 0, accepted = 0; auto t0 = now();
    while ((int)spec.size() < N) {
      auto dr = eng->DSparkDraft(d, anchor, pos, 7); std::vector<int> run{anchor}; run.insert(run.end(), dr.begin(), dr.end());
      std::vector<std::vector<float>> rows; if (!eng->AppendRows(s, run.data(), (int)run.size(), pos, &rows, true, true, &err)) { fprintf(stderr, "verify: %s\n", err.c_str()); return 1; }
      int k = (int)dr.size(), j = 0; std::vector<int> emit;
      for (j = 0; j <= k; ++j) { int t = argmax(rows[j]); emit.push_back(t); if (j < k && t == dr[j]) { ++accepted; continue; } break; }
      int kept = std::min(j + 1, k + 1); if (kept < (int)run.size()) eng->RollbackConv(s, kept);
      eng->DSparkInject(d, pos, kept); pos += kept; ++rounds; drafted += k;
      for (int t : emit) if ((int)spec.size() < N) spec.push_back(t);
      anchor = emit.back();
    }
    printf("speculative: %.2f tok/s, rounds=%ld, drafted=%ld accepted=%ld (%.2f accepted/round)\n", N / sec(t0, now()), rounds, drafted, accepted, rounds ? (double)accepted / rounds : 0.0); eng->DSparkDestroySeq(d); eng->DestroySeq(s); }
  int same = 0; for (int i = 0; i < N; ++i) if (base[i] == spec[i]) ++same; else break;
  printf("identical prefix: %d / %d\n", same, N);
  printf("base: "); for (int i = 0; i < std::min(N, 16); ++i) printf("%d ", base[i]); printf("\nspec: "); for (int i = 0; i < std::min(N, 16); ++i) printf("%d ", spec[i]); printf("\n");
}
