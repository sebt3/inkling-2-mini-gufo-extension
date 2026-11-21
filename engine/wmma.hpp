// Matrix-core (WMMA, gfx11, wave32) GEMM with in-kernel dequantization of the weights: Y[t][r] = sum_k W[r][k] * X[t][k].
// Block tile: 64 weight rows x 64 tokens, K step 32, 4 waves (2 x 2), each wave 32 x 32 = 2 x 2 fragments of 16 x 16 x 16.
// Weights -> f16 in LDS (dq32), activations f32 -> f16 in LDS, accumulation in f32.
#pragma once
#include "kernels.hpp"

namespace inkling::gpu {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int WT_BM = 64, WT_BN = 64, WT_BK = 32, WT_PAD = 40;

// X rows are selected by xrow (or identity); `cnt` tokens in [0, cnt). Output element (token i, row r) goes to Y[(yrow? yrow[i] : i) * ldy + r] * scale.
template <int WT>
__global__ __launch_bounds__(128) void k_wmma_gemm(const uint8_t* __restrict__ W, int K, int N, const float* __restrict__ X, float* __restrict__ Y, int ldy, int cnt,
                                                   const int* __restrict__ xrow, const int* __restrict__ yrow, float scale) {
  __shared__ _Float16 As[WT_BN * WT_PAD], Bs[WT_BM * WT_PAD];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, wy = wave >> 1, wx = wave & 1;
  const int n0 = blockIdx.x * WT_BN, m0 = blockIdx.y * WT_BM; if (m0 >= cnt) return;
  const size_t rb = row_bytes_of(WT, K);
  float8 acc[2][2]; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
  for (int k0 = 0; k0 < K; k0 += WT_BK) {
    if (tid < 64) {  // weights: one row, 32 elements
      const int n = n0 + tid; _Float16* dst = &As[tid * WT_PAD];
      if (n < N) { float w[32]; dq32<WT>(W + (size_t)n * rb, k0 / 32, w); for (int i = 0; i < 32; ++i) dst[i] = (_Float16)w[i]; }
      else for (int i = 0; i < 32; ++i) dst[i] = (_Float16)0.f;
    } else {          // activations: one token, 32 elements
      const int ti = tid - 64, m = m0 + ti; _Float16* dst = &Bs[ti * WT_PAD];
      if (m < cnt) { const float* x = X + (size_t)(xrow ? xrow[m] : m) * K + k0; for (int i = 0; i < 32; i += 4) { const float4 v = *(const float4*)(x + i); dst[i] = (_Float16)v.x; dst[i + 1] = (_Float16)v.y; dst[i + 2] = (_Float16)v.z; dst[i + 3] = (_Float16)v.w; } }
      else for (int i = 0; i < 32; ++i) dst[i] = (_Float16)0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < WT_BK; kk += 16) {
      half16 a[2], b[2];
      for (int i = 0; i < 2; ++i) a[i] = *(const half16*)&As[(wy * 32 + i * 16 + (lane & 15)) * WT_PAD + kk];
      for (int j = 0; j < 2; ++j) b[j] = *(const half16*)&Bs[(wx * 32 + j * 16 + (lane & 15)) * WT_PAD + kk];
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  // C fragment: lane holds token column (lane & 15), weight rows (lane >> 4) + 2 * e, e = 0..7
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
    const int m = m0 + wx * 32 + j * 16 + (lane & 15); if (m >= cnt) continue; const size_t orow = (size_t)(yrow ? yrow[m] : m) * ldy;
    for (int e = 0; e < 8; ++e) { const int n = n0 + wy * 32 + i * 16 + (lane >> 4) + 2 * e; if (n < N) Y[orow + n] = acc[i][j][e] * scale; }
  }
}

// ---- grouped MoE variants: grid (row tiles, token tiles, experts); the pair lists (pair = token*8 + k) come from k_moe_lists.
// gate_up: A rows 0..31 = gate rows f0.., 32..63 = up rows f0.. of the expert; epilogue SITU through LDS -> hid[pair][f].
template <int WT>
__global__ __launch_bounds__(128) void k_wmma_moe_gate_up(const uint8_t* __restrict__ W, int K, int NF, const float* __restrict__ X, const int* __restrict__ lists, const int* __restrict__ counts,
                                                          int npairs, const float* __restrict__ scales, float* __restrict__ hid, float beta, float lin) {
  __shared__ _Float16 As[WT_BN * WT_PAD], Bs[WT_BM * WT_PAD]; __shared__ float Cs[64 * 65];
  const int e = blockIdx.z, tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, wy = wave >> 1, wx = wave & 1;
  const int cnt = counts[e], f0 = blockIdx.x * 32, m0 = blockIdx.y * WT_BM; if (m0 >= cnt) return;
  const int* pl = lists + (size_t)e * npairs; const size_t rb = row_bytes_of(WT, K);
  float8 acc[2][2]; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
  for (int k0 = 0; k0 < K; k0 += WT_BK) {
    if (tid < 64) {
      const int n = f0 + (tid & 31); _Float16* dst = &As[tid * WT_PAD];
      if (n < NF) { const size_t row = (size_t)e * 2 * NF + (tid < 32 ? n : NF + n); float w[32]; dq32<WT>(W + row * rb, k0 / 32, w); for (int i = 0; i < 32; ++i) dst[i] = (_Float16)w[i]; }
      else for (int i = 0; i < 32; ++i) dst[i] = (_Float16)0.f;
    } else {
      const int ti = tid - 64, m = m0 + ti; _Float16* dst = &Bs[ti * WT_PAD];
      if (m < cnt) { const float* x = X + (size_t)(pl[m] / 8) * K + k0; for (int i = 0; i < 32; i += 4) { const float4 v = *(const float4*)(x + i); dst[i] = (_Float16)v.x; dst[i + 1] = (_Float16)v.y; dst[i + 2] = (_Float16)v.z; dst[i + 3] = (_Float16)v.w; } }
      else for (int i = 0; i < 32; ++i) dst[i] = (_Float16)0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < WT_BK; kk += 16) {
      half16 a[2], b[2];
      for (int i = 0; i < 2; ++i) a[i] = *(const half16*)&As[(wy * 32 + i * 16 + (lane & 15)) * WT_PAD + kk];
      for (int j = 0; j < 2; ++j) b[j] = *(const half16*)&Bs[(wx * 32 + j * 16 + (lane & 15)) * WT_PAD + kk];
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) for (int q = 0; q < 8; ++q) Cs[(wy * 32 + i * 16 + (lane >> 4) + 2 * q) * 65 + wx * 32 + j * 16 + (lane & 15)] = acc[i][j][q];
  __syncthreads();
  const float s = scales ? scales[e] : 1.f;
  for (int idx = tid; idx < 32 * 64; idx += 128) {
    const int f = idx / 64, m = idx % 64; if (m0 + m >= cnt || f0 + f >= NF) continue;
    const float g = Cs[f * 65 + m] * s, u = Cs[(32 + f) * 65 + m] * s;
    const float act = beta * tanhf(g / beta) / (1.f + expf(-g)); const float hv = act * (lin * tanhf(u / lin));
    hid[(size_t)pl[m0 + m] * NF + f0 + f] = hv;
  }
}
// down: dn[pair][r] = scale[e] * W_e[r] . hid[pair]
template <int WT>
__global__ __launch_bounds__(128) void k_wmma_moe_down(const uint8_t* __restrict__ W, int K, int N, const float* __restrict__ hid, const int* __restrict__ lists, const int* __restrict__ counts,
                                                       int npairs, const float* __restrict__ scales, float* __restrict__ dn) {
  const int e = blockIdx.z; const int cnt = counts[e]; if ((int)blockIdx.y * WT_BM >= cnt) return;
  const int* pl = lists + (size_t)e * npairs; const uint8_t* We = W + (size_t)e * N * row_bytes_of(WT, K);
  // same tile kernel as the dense one, X rows and Y rows both selected by the expert's pair list
  __shared__ _Float16 As[WT_BN * WT_PAD], Bs[WT_BM * WT_PAD];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, wy = wave >> 1, wx = wave & 1; const int n0 = blockIdx.x * WT_BN, m0 = blockIdx.y * WT_BM; const size_t rb = row_bytes_of(WT, K);
  float8 acc[2][2]; for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
  for (int k0 = 0; k0 < K; k0 += WT_BK) {
    if (tid < 64) { const int n = n0 + tid; _Float16* dst = &As[tid * WT_PAD]; if (n < N) { float w[32]; dq32<WT>(We + (size_t)n * rb, k0 / 32, w); for (int i = 0; i < 32; ++i) dst[i] = (_Float16)w[i]; } else for (int i = 0; i < 32; ++i) dst[i] = (_Float16)0.f; }
    else { const int ti = tid - 64, m = m0 + ti; _Float16* dst = &Bs[ti * WT_PAD];
      if (m < cnt) { const float* x = hid + (size_t)pl[m] * K + k0; for (int i = 0; i < 32; i += 4) { const float4 v = *(const float4*)(x + i); dst[i] = (_Float16)v.x; dst[i + 1] = (_Float16)v.y; dst[i + 2] = (_Float16)v.z; dst[i + 3] = (_Float16)v.w; } }
      else for (int i = 0; i < 32; ++i) dst[i] = (_Float16)0.f; }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < WT_BK; kk += 16) {
      half16 a[2], b[2];
      for (int i = 0; i < 2; ++i) a[i] = *(const half16*)&As[(wy * 32 + i * 16 + (lane & 15)) * WT_PAD + kk];
      for (int j = 0; j < 2; ++j) b[j] = *(const half16*)&Bs[(wx * 32 + j * 16 + (lane & 15)) * WT_PAD + kk];
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  const float sc = scales ? scales[e] : 1.f;
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
    const int m = m0 + wx * 32 + j * 16 + (lane & 15); if (m >= cnt) continue; const size_t orow = (size_t)pl[m] * N;
    for (int q = 0; q < 8; ++q) { const int n = n0 + wy * 32 + i * 16 + (lane >> 4) + 2 * q; if (n < N) dn[orow + n] = acc[i][j][q] * sc; }
  }
}

}  // namespace inkling::gpu
