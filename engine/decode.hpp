// Fused kernels of the single-token decode path (T == 1). Decoding is bound by the number of dependent kernel launches (about 2000 per
// token, ~18 us of latency each), not by memory traffic, so the small stages of a layer are merged: all projections from one input in a
// single launch, convolution + state update (+ residual), clip/norm/tau/KV store, loop controllers, routing + expert lists.
#pragma once
#include "kernels.hpp"

namespace inkling::gpu {

struct GJob { const uint8_t* W; const float* X; float* Y; int wt, K, N; };
struct GJobs { GJob j[8]; int n; };

__device__ inline float dotrow_any(int wt, const uint8_t* row, int K, const float* x, int lane) {
  float acc = 0.f; const int nit = K / 32;
  for (int it = lane; it < nit; it += kWave) {
    float w[32];
    switch (wt) { case W_Q8_0: dq32<W_Q8_0>(row, it, w, K); break; case W_BF16: dq32<W_BF16>(row, it, w, K); break; case W_F32: dq32<W_F32>(row, it, w, K); break;
      case W_NVFP4: dq32<W_NVFP4>(row, it, w, K); break; default: dq32<W_Q5_K>(row, it, w, K); break; }
    const float* xv = x + (size_t)it * 32; float s = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) s += w[k] * xv[k];
    acc += s;
  }
  return wave_sum(acc);
}
// y_j[r] = W_j[r] . x_j for several matrices in one launch; 8 rows (one wave each) per block
__global__ void k_multi_gemv(GJobs J) {
  int b = blockIdx.x, ji = 0;
  for (; ji < J.n; ++ji) { const int nb = (J.j[ji].N + 7) / 8; if (b < nb) break; b -= nb; }
  if (ji >= J.n) return;
  const GJob& j = J.j[ji]; const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave, r = b * 8 + wave; if (r >= j.N) return;
  const float v = dotrow_any(j.wt, j.W + (size_t)r * row_bytes_of(j.wt, j.K), j.K, j.X, lane);
  if (lane == 0) j.Y[r] = v;
}

// causal short convolution + history update + optional residual, one token. hist = table[seq[0]] ([3][ch], oldest first).
struct CJob { const float* x; const float* w; const float* dyn; float* const* tab; float* out; const float* resid; int ch, nh; };
struct CJobs { CJob j[2]; int n; };
__global__ void k_conv1(CJobs J, const int* seq) {
  int b = blockIdx.x, ji = 0;
  for (; ji < J.n; ++ji) { const int nb = (J.j[ji].ch + 127) / 128; if (b < nb) break; b -= nb; }
  if (ji >= J.n) return;
  const CJob& j = J.j[ji]; const int c = b * 128 + threadIdx.x; if (c >= j.ch) return;
  float* h = j.tab[seq[0]]; const float v0 = h[c], v1 = h[(size_t)j.ch + c], v2 = h[(size_t)2 * j.ch + c], v3 = j.x[c];
  float o = v0 * j.w[(size_t)c * 4] + v1 * j.w[(size_t)c * 4 + 1] + v2 * j.w[(size_t)c * 4 + 2] + v3 * j.w[(size_t)c * 4 + 3];
  if (j.dyn) { const int per = j.ch / j.nh, hd = c / per; o += v3 * j.dyn[hd * 4] + v2 * j.dyn[hd * 4 + 1] + v1 * j.dyn[hd * 4 + 2] + v0 * j.dyn[hd * 4 + 3]; }
  float r = (v3 + o) * rsqrtf((float)((j.dyn ? 2 : 1) * 4 + 1)); if (j.resid) r += j.resid[c];
  j.out[c] = r; h[c] = v1; h[(size_t)j.ch + c] = v2; h[(size_t)2 * j.ch + c] = v3;
}

// q/K/V/r post-processing and KV store for one token. blocks: 0..NH-1 q heads, then NKV K heads, NKV V heads, NH r heads.
__global__ void k_post1(float* q, float* K, float* V, float* r, const float* q_clip, const float* rel_clip, __half* const* kc, __half* const* vc, const int* seq, const int* pos,
                        int NH, int NKV, int nrel, float qk_clip, float rel_thr, float eps, int global, float floor_, float alpha, int cap) {
  __shared__ float red[128]; const int tid = threadIdx.x; int b = blockIdx.x; float* x; int d, kind; const float* scale = nullptr; float thr = 0.f;
  if (b < NH) { kind = 0; x = q + (size_t)b * 128; d = 128; scale = q_clip + (size_t)b * 128; thr = qk_clip; }
  else if ((b -= NH) < NKV) { kind = 1; x = K + (size_t)b * 128; d = 128; thr = qk_clip; }
  else if ((b -= NKV) < NKV) { kind = 2; x = V + (size_t)b * 128; d = 128; }
  else { b -= NKV; kind = 3; x = r + (size_t)b * nrel; d = nrel; scale = rel_clip + (size_t)b * nrel; thr = rel_thr; }
  float v = tid < d ? x[tid] : 0.f; if (scale && tid < d) v *= scale[tid];
  red[tid] = v * v; __syncthreads();
  for (int o = 64; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
  if (kind == 2) v *= rsqrtf(red[0] / d + eps);
  else v *= fminf(1.f, fmaxf(0.f, thr / (sqrtf(red[0]) + 1e-6f)));
  if (global && (kind == 0 || kind == 3)) { float tau = logf(fmaxf(1.f, (float)(pos[0] + 1) / floor_)); v *= alpha * tau + 1.f; }
  if (tid < d) x[tid] = v;
  if ((kind == 1 || kind == 2) && tid < 128) { const size_t o = (size_t)(pos[0] % cap) * (NKV * 128) + (size_t)b * 128 + tid; ((kind == 1) ? kc[seq[0]] : vc[seq[0]])[o] = __float2half(v); }
}

// loop controller, second half: gain/forget (rank-64 F32 matrices) and the update, one token
__global__ void k_control2(const float* cur, const float* msg, const float* low, const float* Wg, const float* Wf, float* out, int E, int R) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= E) return; float g = 0.f, f = 0.f;
  for (int i = 0; i < R; ++i) { g += Wg[(size_t)c * R + i] * low[i]; f += Wf[(size_t)c * R + i] * low[i]; }
  out[c] = cur[c] - tanhf(f) * cur[c] + tanhf(g) * msg[c];
}

// routing + expert lists for one token (pair k -> token 0): counts[e] = 1 for the chosen experts
__global__ void k_route_lists1(const float* logits, const float* bias, int NE, int NU, float wscale, int* ids, float* wts, int* lists, int* counts, int npairs) {
  __shared__ float p[128], sel[128], rv[128]; __shared__ int ri[128], chosen[16]; const int e = threadIdx.x; float pr = 0.f, sl = -1e30f;
  if (e < NE) { pr = 1.f / (1.f + expf(-logits[e])); sl = pr + (bias ? bias[e] : 0.f); }
  p[e] = pr; sel[e] = sl; counts[e] = 0; __syncthreads();
  for (int k = 0; k < NU; ++k) {
    rv[e] = sel[e]; ri[e] = e; __syncthreads();
    for (int o = 64; o > 0; o >>= 1) { if (e < o) { const float a = rv[e], b = rv[e + o]; const int ia = ri[e], ib = ri[e + o]; if (b > a || (b == a && ib < ia)) { rv[e] = b; ri[e] = ib; } } __syncthreads(); }
    if (e == 0) { chosen[k] = ri[0]; sel[ri[0]] = -1e30f - k; }
    __syncthreads();
  }
  if (e == 0) {
    float ws = 0.f; for (int k = 0; k < NU; ++k) ws += p[chosen[k]]; ws += 1e-20f;
    for (int k = 0; k < NU; ++k) { ids[k] = chosen[k]; wts[k] = p[chosen[k]] / ws * wscale; counts[chosen[k]] = 1; lists[(size_t)chosen[k] * npairs] = k; }
  }
}

}  // namespace inkling::gpu
