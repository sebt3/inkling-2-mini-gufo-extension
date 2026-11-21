// WMMA GEMM unit test and micro-benchmark against the scalar kernel (k_gemm): random Q8_0 / NVFP4 / BF16 weights.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "../engine/wmma.hpp"
using namespace inkling::gpu;
template <int WT> void run(const char* name, int K, int N, int T) {
  std::mt19937 rng(1); const size_t rb = row_bytes_of(WT, K); std::vector<uint8_t> w((size_t)N * rb);
  for (auto& b : w) b = rng() & 255;
  if (WT == W_Q8_0) for (int r = 0; r < N; ++r) for (int b = 0; b < K / 32; ++b) { uint8_t* p = &w[(size_t)r * rb + (size_t)b * 34]; uint16_t h = 0x2000 + (rng() & 0x3ff); std::memcpy(p, &h, 2); }   // small positive f16 scales
  if (WT == W_NVFP4) for (int r = 0; r < N; ++r) for (int b = 0; b < K / 64; ++b) for (int s = 0; s < 4; ++s) w[(size_t)r * rb + (size_t)b * 36 + s] = 0x30 + (rng() & 15);                 // valid ue4m3 scales
  if (WT == W_BF16) { uint16_t* p = (uint16_t*)w.data(); for (size_t i = 0; i < (size_t)N * K; ++i) p[i] = 0x3c00 + (rng() & 0x3ff) - 0x200; }
  std::vector<float> x((size_t)T * K); std::uniform_real_distribution<float> u(-1, 1); for (auto& v : x) v = u(rng);
  uint8_t* dw; float *dx, *dy1, *dy2; HIP_CHECK(hipMalloc(&dw, w.size())); HIP_CHECK(hipMalloc(&dx, x.size() * 4)); HIP_CHECK(hipMalloc(&dy1, (size_t)T * N * 4)); HIP_CHECK(hipMalloc(&dy2, (size_t)T * N * 4));
  HIP_CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  auto t = [&](auto f) { f(); HIP_CHECK(hipDeviceSynchronize()); auto a = std::chrono::steady_clock::now(); for (int i = 0; i < 5; ++i) f(); HIP_CHECK(hipDeviceSynchronize()); return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / 5; };
  double t1 = t([&] { k_gemm<WT, 4><<<dim3((N + 7) / 8, (T + 3) / 4), 256>>>(dw, K, N, dx, dy1, T, nullptr, nullptr, nullptr); });
  double t2 = t([&] { k_wmma_gemm<WT><<<dim3((N + 63) / 64, (T + 63) / 64), 128>>>(dw, K, N, dx, dy2, N, T, nullptr, nullptr, 1.f); });
  std::vector<float> y1((size_t)T * N), y2(y1.size()); HIP_CHECK(hipMemcpy(y1.data(), dy1, y1.size() * 4, hipMemcpyDeviceToHost)); HIP_CHECK(hipMemcpy(y2.data(), dy2, y2.size() * 4, hipMemcpyDeviceToHost));
  double num = 0, den = 0; for (size_t i = 0; i < y1.size(); ++i) { num += (y1[i] - y2[i]) * (double)(y1[i] - y2[i]); den += (double)y1[i] * y1[i]; }
  const double flop = 2.0 * T * N * K;
  printf("%-6s K=%d N=%d T=%d: rel_err=%.2e  scalar %.3f ms (%.2f TFLOPS)  wmma %.3f ms (%.2f TFLOPS)\n", name, K, N, T, std::sqrt(num / den), t1 * 1e3, flop / t1 / 1e12, t2 * 1e3, flop / t2 / 1e12);
}
int main() {
  run<W_Q8_0>("Q8_0", 2304, 3072, 1024); run<W_NVFP4>("NVFP4", 2304, 2304, 64); run<W_NVFP4>("NVFP4", 2304, 2304, 1024); run<W_Q5_K>("Q5_K", 2304, 2304, 256); run<W_BF16>("BF16", 2304, 128, 1024); run<W_Q8_0>("Q8_0", 2304, 3072, 24);
}
