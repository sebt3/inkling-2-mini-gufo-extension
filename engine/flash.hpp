// Prefill attention on matrix cores (flash style) for one sequence of consecutive positions. Per block: one head and 64 query tokens
// (4 waves x 16 rows); keys in tiles of 64 from the f16 KV cache. logit = q.k * scale + bias[t][h][dist] with dist = qpos - kpos, the
// relative-position bias coming from a table built by a WMMA GEMM; the sink is a zero-valued key of logit 0 that starts the running softmax.
// The sigmoid output gate is applied at the end.
#pragma once
#include "wmma.hpp"

namespace inkling::gpu {

// Rpad[(t*NH+h)][32] = r[t][h][0..15], zero padded
__global__ void k_bias_pad(const float* r, float* out, size_t rows) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= rows * 32) return;
  const size_t row = i / 32; const int c = i % 32; out[i] = c < 16 ? r[row * 16 + c] : 0.f;
}
__global__ void k_pad_rb(const float* rb /*[1024][16]*/, float* out /*[1024][32]*/, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * 32) return; const int r = i / 32, c = i % 32; out[i] = c < 16 ? rb[r * 16 + c] : 0.f; }

constexpr int FA_LD = 136;   // K tile row stride (halves)
constexpr int FA_VLD = 72;   // V^T and P row stride (halves)

__global__ __launch_bounds__(128) void k_flash_attn(const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc, const float* __restrict__ bias_tab, const float* __restrict__ sinks,
                                                    const float* __restrict__ gate, const int* __restrict__ pos, int T, int NH, int NKV, int cap, int window, float* __restrict__ out) {
  __shared__ _Float16 kv_s[128 * FA_VLD > 64 * FA_LD ? 128 * FA_VLD : 64 * FA_LD]; __shared__ _Float16 P_s[4][16 * FA_VLD];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, h = blockIdx.y, kvh = h / (NH / NKV), q0 = blockIdx.x * 64; const int KVD = NKV * 128;
  const int qrow = q0 + wave * 16 + (lane & 15);              // row held for the A fragments (and the Q load)
  half16 qf[8];
  for (int kk = 0; kk < 8; ++kk) { half16 v; for (int i = 0; i < 16; ++i) v[i] = qrow < T ? (_Float16)q[(size_t)qrow * NH * 128 + (size_t)h * 128 + kk * 16 + i] : (_Float16)0.f; qf[kk] = v; }
  const int pmin = pos[q0], pmax = pos[min(q0 + 63, T - 1)];
  const int kstart = window ? max(0, pmin - window + 1) : 0; const int kfirst = (kstart / 64) * 64;
  float8 O[8]; for (int j = 0; j < 8; ++j) O[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
  float m[8], l[8]; for (int e = 0; e < 8; ++e) { m[e] = 0.f; l[e] = 1.f; }   // the sink: logit 0, no value
  int qp[8]; for (int e = 0; e < 8; ++e) { const int r = q0 + wave * 16 + 2 * e + (lane >> 4); qp[e] = r < T ? pos[r] : -1; }
  const float scale = rsqrtf(128.f);
  for (int k0 = kfirst; k0 <= pmax; k0 += 64) {
    __syncthreads();
    for (int idx = tid; idx < 64 * 16; idx += 128) {   // K tile
      const int key = idx / 16, ch = idx % 16, kp = k0 + key; uint4 v = make_uint4(0, 0, 0, 0);
      if (kp <= pmax && kp >= 0) v = *(const uint4*)(kc + (size_t)(kp % cap) * KVD + kvh * 128 + ch * 8);
      *(uint4*)&kv_s[key * FA_LD + ch * 8] = v;
    }
    __syncthreads();
    float8 S[4];
    for (int jn = 0; jn < 4; ++jn) { float8 a = (float8){0, 0, 0, 0, 0, 0, 0, 0};
      for (int kk = 0; kk < 8; ++kk) { half16 b = *(const half16*)&kv_s[(jn * 16 + (lane & 15)) * FA_LD + kk * 16]; a = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[kk], b, a); }
      S[jn] = a; }
    // scale, bias, mask; running softmax
    float pe[4][8];
    for (int e = 0; e < 8; ++e) {
      const int r = q0 + wave * 16 + 2 * e + (lane >> 4); float tm = -1e30f; float sv[4]; const float sk = r < T ? sinks[(size_t)r * NH + h] : 0.f;
      for (int jn = 0; jn < 4; ++jn) {
        const int kp = k0 + jn * 16 + (lane & 15); const int dist = qp[e] - kp; float s = -1e30f;
        if (r < T && dist >= 0 && (window == 0 || dist < window)) { s = S[jn][e] * scale - sk; if (dist < 1024) s += bias_tab[((size_t)r * NH + h) * 1024 + dist]; }
        sv[jn] = s; tm = fmaxf(tm, s);
      }
      for (int off = 8; off > 0; off >>= 1) tm = fmaxf(tm, __shfl_xor(tm, off, 16));
      const float nm = fmaxf(m[e], tm), alpha = expf(m[e] - nm); float ps = 0.f;
      for (int jn = 0; jn < 4; ++jn) { const float p = sv[jn] <= -1e29f ? 0.f : expf(sv[jn] - nm); pe[jn][e] = p; ps += p; }
      for (int off = 8; off > 0; off >>= 1) ps += __shfl_xor(ps, off, 16);
      l[e] = l[e] * alpha + ps; m[e] = nm; for (int jd = 0; jd < 8; ++jd) O[jd][e] *= alpha;
    }
    // P (C layout) -> LDS [16][72] per wave
    for (int e = 0; e < 8; ++e) for (int jn = 0; jn < 4; ++jn) P_s[wave][(2 * e + (lane >> 4)) * FA_VLD + jn * 16 + (lane & 15)] = (_Float16)pe[jn][e];
    __syncthreads();   // K tile no longer needed: load V^T
    for (int idx = tid; idx < 64 * 16; idx += 128) {
      const int key = idx / 16, ch = idx % 16, kp = k0 + key; uint4 v = make_uint4(0, 0, 0, 0);
      if (kp <= pmax && kp >= 0) v = *(const uint4*)(vc + (size_t)(kp % cap) * KVD + kvh * 128 + ch * 8);
      const _Float16* vh = (const _Float16*)&v; for (int i = 0; i < 8; ++i) kv_s[(ch * 8 + i) * FA_VLD + key] = vh[i];
    }
    __syncthreads();
    for (int kk = 0; kk < 4; ++kk) {
      const half16 a = *(const half16*)&P_s[wave][(lane & 15) * FA_VLD + kk * 16];
      for (int jd = 0; jd < 8; ++jd) { half16 b = *(const half16*)&kv_s[(jd * 16 + (lane & 15)) * FA_VLD + kk * 16]; O[jd] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, O[jd]); }
    }
  }
  for (int e = 0; e < 8; ++e) {
    const int r = q0 + wave * 16 + 2 * e + (lane >> 4); if (r >= T) continue; const float inv = 1.f / l[e];
    for (int jd = 0; jd < 8; ++jd) { const int d = jd * 16 + (lane & 15); const size_t o = (size_t)r * NH * 128 + (size_t)h * 128 + d;
      out[o] = O[jd][e] * inv / (1.f + expf(-gate[o])); }
  }
}

}  // namespace inkling::gpu
