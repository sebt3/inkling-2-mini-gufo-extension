#include <chrono>
#include <cstdio>
#include <hip/hip_runtime.h>
__global__ void nop(float* x) { if (threadIdx.x == 1000) x[0] = 1.f; }
int main() {
  float* d; (void)hipMalloc(&d, 4); const int N = 20000;
  for (int i = 0; i < 100; ++i) nop<<<1, 64>>>(d); (void)hipDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < N; ++i) nop<<<1, 64>>>(d); auto t1 = std::chrono::steady_clock::now(); (void)hipDeviceSynchronize(); auto t2 = std::chrono::steady_clock::now();
  printf("empty kernel: %.2f us CPU per launch, %.2f us total per kernel (GPU drain included)\n", std::chrono::duration<double, std::micro>(t1 - t0).count() / N, std::chrono::duration<double, std::micro>(t2 - t0).count() / N);
}
