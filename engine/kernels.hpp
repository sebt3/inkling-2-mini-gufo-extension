// HIP kernels of the Inkling-2-Mini engine. Correctness first: F32 activations, F16 KV, weights read in their
// native quantized layout. One translation unit (hipcc); no Gufo dependency.
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace inkling::gpu {

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); abort(); } } while (0)

enum WType : int { W_F32 = 0, W_BF16 = 1, W_Q8_0 = 2, W_Q5_K = 3, W_NVFP4 = 4 };
constexpr int kWave = 32;

__host__ __device__ inline size_t row_bytes_of(int wt, int K) {
  switch (wt) { case W_F32: return (size_t)K * 4; case W_BF16: return (size_t)K * 2; case W_Q8_0: return (size_t)(K / 32) * 34;
    case W_Q5_K: return (size_t)(K / 256) * 176; case W_NVFP4: return (size_t)(K / 64) * 36; }
  return 0;
}

__device__ inline float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ inline float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ inline float ue4m3_to_f32(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.f;
  int e = (x >> 3) & 0xF, m = x & 7;
  float raw = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.f + (float)m / 8.f, e - 7);
  return raw * 0.5f;
}
__constant__ int8_t kFp4D[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Dequantize 32 consecutive elements (work item `it` of a row) into out[32].
template <int WT>
__device__ inline void dq32(const uint8_t* row, int it, float* out) {
  if constexpr (WT == W_F32) {
    const float* p = (const float*)row + it * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = p[i];
  } else if constexpr (WT == W_BF16) {
    const uint16_t* p = (const uint16_t*)row + it * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = bf2f(p[i]);
  } else if constexpr (WT == W_Q8_0) {
    const uint8_t* b = row + (size_t)it * 34;
    float d = h2f(*(const uint16_t*)b);
    const int8_t* q = (const int8_t*)(b + 2);
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = d * q[i];
  } else if constexpr (WT == W_NVFP4) {
    const uint8_t* b = row + (size_t)(it >> 1) * 36; const int half = it & 1;  // 64-element block = 2 items
    const uint8_t* qs = b + 4;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int sb = half * 2 + s; float d = ue4m3_to_f32(b[sb]);
#pragma unroll
      for (int j = 0; j < 8; ++j) { uint8_t q = qs[sb * 8 + j]; out[s * 16 + j] = kFp4D[q & 0xF] * d; out[s * 16 + j + 8] = kFp4D[q >> 4] * d; }
    }
  } else if constexpr (WT == W_Q5_K) {
    const uint8_t* b = row + (size_t)(it >> 3) * 176; const int g = it & 7;  // 256-element block = 8 items
    float d = h2f(*(const uint16_t*)b), mn = h2f(*(const uint16_t*)(b + 2));
    const uint8_t* sc12 = b + 4; const uint8_t* qh = b + 16; const uint8_t* ql = b + 48 + (g >> 1) * 32;
    uint8_t sc, m;
    if (g < 4) { sc = sc12[g] & 63; m = sc12[g + 4] & 63; }
    else { sc = (sc12[g + 4] & 0xF) | ((sc12[g - 4] >> 6) << 4); m = (sc12[g + 4] >> 4) | ((sc12[g] >> 6) << 4); }
    const float d1 = d * sc, m1 = mn * m; const uint8_t mask = (uint8_t)(1u << g); const bool hi = g & 1;
#pragma unroll
    for (int l = 0; l < 32; ++l) { int q = (hi ? (ql[l] >> 4) : (ql[l] & 0xF)) + ((qh[l] & mask) ? 16 : 0); out[l] = d1 * q - m1; }
  }
}

__device__ inline float wave_sum(float v) {
#pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
  return v;
}

// Y[t, r] = W[r, :] . X[t, :], NT tokens per block, one wave per output row. Wave32, 8 waves per block.
// xrow/yrow: optional row indirection (token index for X rows / output rows). rows beyond `T` are skipped.
template <int WT, int NT>
__global__ void k_gemm(const uint8_t* __restrict__ W, int K, int N, const float* __restrict__ X, float* __restrict__ Y, int T,
                       const int* __restrict__ xrow, const int* __restrict__ yrow, const float* __restrict__ row_scale) {
  const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  const int r = blockIdx.x * 8 + wave; if (r >= N) return;
  const int t0 = blockIdx.y * NT;
  const uint8_t* row = W + (size_t)r * row_bytes_of(WT, K);
  float acc[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i] = 0.f;
  const int nit = K / 32;
  for (int it = lane; it < nit; it += kWave) {
    float w[32]; dq32<WT>(row, it, w);
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      if (t0 + i < T) {
        const int xr = xrow ? xrow[t0 + i] : t0 + i; const float* x = X + (size_t)xr * K + it * 32; float s = 0.f;
#pragma unroll
        for (int k = 0; k < 32; ++k) s += w[k] * x[k];
        acc[i] += s;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    float v = wave_sum(acc[i]);
    if (lane == 0 && t0 + i < T) { const int yr = yrow ? yrow[t0 + i] : t0 + i; Y[(size_t)yr * N + r] = row_scale ? v * row_scale[0] : v; }
  }
}

// ---------------------------------------------------------------------------------------------------------------
__global__ void k_embed_norm(const int* tokens, const uint8_t* emb, size_t row_bytes, const float* norm, float* X, int E, float eps) {
  extern __shared__ float sh[];  // E floats + 32
  const int t = blockIdx.x; const uint8_t* row = emb + (size_t)tokens[t] * row_bytes;
  float ss = 0.f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    const uint8_t* b = row + (size_t)(i / 32) * 34; float v = h2f(*(const uint16_t*)b) * (float)((const int8_t*)(b + 2))[i % 32];
    sh[i] = v; ss += v * v;
  }
  float* red = sh + E; red[threadIdx.x] = ss; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  const float r = 1.f / sqrtf(red[0] / E + eps);
  for (int i = threadIdx.x; i < E; i += blockDim.x) X[(size_t)t * E + i] = sh[i] * r * norm[i];
}

// rows of length E: out = rms(in) * w (w optional)
__global__ void k_rmsnorm(const float* in, const float* w, float* out, int E, float eps) {
  __shared__ float red[256]; const int t = blockIdx.x; const float* x = in + (size_t)t * E; float ss = 0.f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) ss += x[i] * x[i];
  red[threadIdx.x] = ss; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  const float r = 1.f / sqrtf(red[0] / E + eps);
  for (int i = threadIdx.x; i < E; i += blockDim.x) out[(size_t)t * E + i] = x[i] * r * (w ? w[i] : 1.f);
}

// attention over depth. states: nb block buffers + cur (all [T,E]); query [nq,E]; outs[q] -> [T,E].
struct MixArgs { const float* st[6]; int ns; float* out[4]; int nq; };
__global__ void k_mix(MixArgs a, const float* query, int E, float eps) {
  __shared__ float red[24][256]; __shared__ float inv[6]; __shared__ float wt[4][6];
  const int t = blockIdx.x, tid = threadIdx.x;
  float ss[6], sc[24];
  for (int s = 0; s < a.ns; ++s) ss[s] = 0.f;
  for (int q = 0; q < 24; ++q) sc[q] = 0.f;
  for (int i = tid; i < E; i += blockDim.x) for (int s = 0; s < a.ns; ++s) { float v = a.st[s][(size_t)t * E + i]; ss[s] += v * v; }
  for (int s = 0; s < a.ns; ++s) red[s][tid] = ss[s];
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (tid < o) for (int s = 0; s < a.ns; ++s) red[s][tid] += red[s][tid + o]; __syncthreads(); }
  if (tid < a.ns) inv[tid] = 1.f / sqrtf(red[tid][0] / E + eps);
  __syncthreads();
  for (int i = tid; i < E; i += blockDim.x) for (int s = 0; s < a.ns; ++s) { float v = a.st[s][(size_t)t * E + i] * inv[s]; for (int q = 0; q < a.nq; ++q) sc[s * 4 + q] += v * query[(size_t)q * E + i]; }
  for (int k = 0; k < a.ns * 4; ++k) red[k][tid] = sc[k];
  __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (tid < o) for (int k = 0; k < a.ns * 4; ++k) red[k][tid] += red[k][tid + o]; __syncthreads(); }
  if (tid < a.nq) {
    float mx = -1e30f; for (int s = 0; s < a.ns; ++s) { float v = red[s * 4 + tid][0] * rsqrtf((float)E); wt[tid][s] = v; mx = fmaxf(mx, v); }
    float sum = 0.f; for (int s = 0; s < a.ns; ++s) { wt[tid][s] = expf(wt[tid][s] - mx); sum += wt[tid][s]; }
    for (int s = 0; s < a.ns; ++s) wt[tid][s] /= sum;
  }
  __syncthreads();
  for (int i = tid; i < E; i += blockDim.x)
    for (int q = 0; q < a.nq; ++q) { float o = 0.f; for (int s = 0; s < a.ns; ++s) o += wt[q][s] * a.st[s][(size_t)t * E + i] * inv[s]; a.out[q][(size_t)t * E + i] = o; }
}

// gate[t] = [rms(cur[t]), rms(msg[t])]
__global__ void k_control_gate(const float* cur, const float* msg, float* gate, int E, float eps) {
  __shared__ float red[2][256]; const int t = blockIdx.x, tid = threadIdx.x; float a = 0.f, b = 0.f;
  for (int i = tid; i < E; i += blockDim.x) { float x = cur[(size_t)t * E + i], y = msg[(size_t)t * E + i]; a += x * x; b += y * y; }
  red[0][tid] = a; red[1][tid] = b; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (tid < o) { red[0][tid] += red[0][tid + o]; red[1][tid] += red[1][tid + o]; } __syncthreads(); }
  const float ra = 1.f / sqrtf(red[0][0] / E + eps), rb = 1.f / sqrtf(red[1][0] / E + eps);
  for (int i = tid; i < E; i += blockDim.x) { gate[(size_t)t * 2 * E + i] = cur[(size_t)t * E + i] * ra; gate[(size_t)t * 2 * E + E + i] = msg[(size_t)t * E + i] * rb; }
}
__global__ void k_control_apply(const float* cur, const float* msg, const float* gain, const float* forget, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  out[i] = cur[i] - tanhf(forget[i]) * cur[i] + tanhf(gain[i]) * msg[i];
}

// causal short convolution with residual. x [T,ch]; w [ch][4] (w[k + c*4]); dyn [T, nh*4] or null.
// hist: per-sequence pointer to [3][ch] (oldest first). A previous token in the same batch is used when it belongs to
// the same sequence at the right position, else the stored history.
__global__ void k_conv(const float* x, const float* w, const float* dyn, int nh, const int* seq, const int* pos, const int* seq_first, float* const* hist, float* out, int T, int ch) {
  const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ch) return;
  float v[4];  // index k = 0..3 oldest..current
  v[3] = x[(size_t)t * ch + c];
  const int i = t - seq_first[seq[t]];  // rank of the token among the batch tokens of its sequence (consecutive positions)
  for (int lag = 1; lag <= 3; ++lag) {
    float val;
    if (lag <= i) val = x[(size_t)(t - lag) * ch + c];
    else { const int hl = lag - i;  // lags before the batch come from the stored history (newest entry = position of the first batch token - 1)
      val = (pos[t] - lag >= 0 && hl <= 3) ? hist[seq[t]][(size_t)(3 - hl) * ch + c] : 0.f; }
    v[3 - lag] = val;
  }
  float o = 0.f;
  for (int k = 0; k < 4; ++k) o += v[k] * w[k + (size_t)c * 4];
  if (dyn) { const int per = ch / nh, h = c / per; for (int lag = 0; lag < 4; ++lag) o += v[3 - lag] * dyn[(size_t)t * nh * 4 + h * 4 + lag]; }
  out[(size_t)t * ch + c] = (v[3] + o) * rsqrtf((float)((dyn ? 2 : 1) * 4 + 1));
}
// new history of each sequence = last 3 inputs of [old history, this batch's inputs of the sequence]
__global__ void k_conv_state(const float* x, const int* seq_first, const int* seq_count, float* const* hist, int ch) {
  const int s = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ch) return;
  const int n = seq_count[s]; if (n == 0) return; const int last = seq_first[s] + n - 1;
  float* h = hist[s]; float nw[3];
  for (int l = 0; l < 3; ++l) nw[2 - l] = l < n ? x[(size_t)(last - l) * ch + c] : h[(size_t)(2 - (l - n)) * ch + c];
  for (int i = 0; i < 3; ++i) h[(size_t)i * ch + c] = nw[i];
}

// rows of d (<=128) values: optional per-channel scale, then scale down to an L2 norm <= thr
__global__ void k_clip(float* x, int d, const float* scale, int rows_per_token, float thr) {
  __shared__ float red[128]; const size_t row = blockIdx.x; const int tid = threadIdx.x; const int r = row % rows_per_token;
  float v = 0.f; if (tid < d) { v = x[row * d + tid]; if (scale) v *= scale[(size_t)r * d + tid]; }
  red[tid] = v * v; __syncthreads();
  for (int o = 64; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
  const float nrm = sqrtf(red[0]) + 1e-6f; const float f = fminf(1.f, fmaxf(0.f, thr / nrm));
  if (tid < d) x[row * d + tid] = v * f;
}
// per-head rms norm without gamma (V)
__global__ void k_rms_head(float* x, int d, float eps) {
  __shared__ float red[128]; const size_t row = blockIdx.x; const int tid = threadIdx.x; float v = tid < d ? x[row * d + tid] : 0.f;
  red[tid] = v * v; __syncthreads();
  for (int o = 64; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
  if (tid < d) x[row * d + tid] = v * rsqrtf(red[0] / d + eps);
}
// global layers: q and r multiplied by tau(pos)
__global__ void k_tau(float* q, int nq, float* r, int nr, const int* pos, float floor_, float alpha) {
  const int t = blockIdx.x; float tau = logf(fmaxf(1.f, (float)(pos[t] + 1) / floor_)); tau = alpha * tau + 1.f;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) q[(size_t)t * nq + i] *= tau;
  for (int i = threadIdx.x; i < nr; i += blockDim.x) r[(size_t)t * nr + i] *= tau;
}

// KV store (f16). cache layout [cap][kvd]; index = pos % cap
__global__ void k_kv_store(const float* K, const float* V, __half* const* kc, __half* const* vc, const int* seq, const int* pos, int cap, int kvd) {
  const int t = blockIdx.x; const size_t idx = (size_t)(pos[t] % cap) * kvd;
  for (int i = threadIdx.x; i < kvd; i += blockDim.x) { kc[seq[t]][idx + i] = __float2half(K[(size_t)t * kvd + i]); vc[seq[t]][idx + i] = __float2half(V[(size_t)t * kvd + i]); }
}

// split-K attention. grid (T*NH, nsplit), 128 threads. Partial results: m, l, acc[128] per (token, head, split).
__global__ void k_attn_part(const float* q, const float* r, const float* sinks, const __half* const* kc, const __half* const* vc, const float* relbias,
                            const int* seq, const int* pos, int NH, int NKV, int D, int nrel, int cap, int window /*0 = global*/, int chunk,
                            float* pm, float* pl, float* pacc, int nsplit) {
  __shared__ float qs[128], rs[16], lg[128]; __shared__ float red[128];
  const int t = blockIdx.x / NH, h = blockIdx.x % NH, sp = blockIdx.y, tid = threadIdx.x;
  const int p = pos[t], kv = h / (NH / NKV); const int sq = seq[t];
  const int jlo = window ? max(0, p - window + 1) : 0;
  const int j0 = jlo + sp * chunk, j1 = min(p + 1, j0 + chunk);
  const size_t o = ((size_t)blockIdx.x * nsplit + sp);
  if (j0 >= j1) { if (tid == 0) { pm[o] = -1e30f; pl[o] = 0.f; } pacc[o * 128 + tid] = 0.f; return; }
  qs[tid] = tid < D ? q[((size_t)t * NH + h) * D + tid] : 0.f; if (tid < nrel) rs[tid] = r[((size_t)t * NH + h) * nrel + tid];
  const float sink = sinks[(size_t)t * NH + h], scale = rsqrtf((float)D); __syncthreads();
  const __half* kbase = kc[sq]; const __half* vbase = vc[sq];
  float m = -1e30f, l = 0.f, acc = 0.f;
  for (int j = j0; j < j1; j += 128) {
    const int n = min(128, j1 - j); float s = -1e30f;
    if (tid < n) {
      const __half* kr = kbase + (size_t)((j + tid) % cap) * (NKV * D) + kv * D; float dot = 0.f;
      for (int d = 0; d < D; ++d) dot += qs[d] * __half2float(kr[d]);
      const int dist = p - (j + tid); float bias = 0.f;
      if (dist >= 0 && dist < 1024) { const float* rb = relbias + (size_t)dist * nrel; for (int i = 0; i < nrel; ++i) bias += rb[i] * rs[i]; }
      s = dot * scale + bias - sink;
    }
    lg[tid] = s; red[tid] = s; __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]); __syncthreads(); }
    const float mt = red[0]; const float mn = fmaxf(m, mt); const float corr = expf(m - mn);
    acc *= corr; l *= corr; m = mn; __syncthreads();
    float a2 = 0.f, l2 = 0.f;
    for (int i = 0; i < n; ++i) { const float w = expf(lg[i] - m); l2 += w; a2 += w * __half2float(vbase[(size_t)((j + i) % cap) * (NKV * D) + kv * D + tid]); }
    acc += a2; l += l2; __syncthreads();
  }
  if (tid == 0) { pm[o] = m; pl[o] = l; }
  pacc[o * 128 + tid] = acc;
}
// combine splits with the (zero) sink logit, apply the sigmoid output gate
__global__ void k_attn_combine(const float* pm, const float* pl, const float* pacc, const float* gate, float* out, int NH, int D, int nsplit) {
  const size_t b = blockIdx.x; const int tid = threadIdx.x; float M = 0.f;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pm[b * nsplit + s]);
  float den = expf(-M), a = 0.f;
  for (int s = 0; s < nsplit; ++s) { const float e = expf(pm[b * nsplit + s] - M); den += pl[b * nsplit + s] * e; a += pacc[(b * nsplit + s) * 128 + tid] * e; }
  if (tid < D) { const float g = gate[b * D + tid]; out[b * D + tid] = a / den / (1.f + expf(-g)); }
}

__global__ void k_add(const float* a, const float* b, float* out, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] + b[i]; }

// MoE routing: sigmoid, biased top-k selection (ties -> lowest index), normalised and scaled weights.
__global__ void k_route(const float* logits, const float* bias, int NE, int NU, float wscale, int* ids, float* wts, const int* forced) {
  __shared__ float p[128], sel[128], rv[128]; __shared__ int ri[128]; __shared__ int chosen[16];
  const int t = blockIdx.x, e = threadIdx.x; float pr = 0.f, sl = -1e30f;
  if (e < NE) { pr = 1.f / (1.f + expf(-logits[(size_t)t * NE + e])); sl = pr + (bias ? bias[e] : 0.f); }
  p[e] = pr; sel[e] = sl; __syncthreads();
  if (forced) { if (e == 0) for (int k = 0; k < NU; ++k) chosen[k] = forced[(size_t)t * NU + k]; __syncthreads(); }
  for (int k = 0; k < NU && !forced; ++k) {
    rv[e] = sel[e]; ri[e] = e; __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
      if (e < o) { const float a = rv[e], b = rv[e + o]; const int ia = ri[e], ib = ri[e + o]; if (b > a || (b == a && ib < ia)) { rv[e] = b; ri[e] = ib; } }
      __syncthreads();
    }
    if (e == 0) { chosen[k] = ri[0]; sel[ri[0]] = -1e30f - k; }
    __syncthreads();
  }
  if (e == 0) {
    float ws = 0.f; for (int k = 0; k < NU; ++k) ws += p[chosen[k]]; ws += 1e-20f;
    for (int k = 0; k < NU; ++k) { ids[(size_t)t * NU + k] = chosen[k]; wts[(size_t)t * NU + k] = p[chosen[k]] / ws * wscale; }
  }
}
// deterministic per-expert lists of pair indices (t*NU + k): one block per expert, in pair order
__global__ void k_moe_lists(const int* ids, int npairs, int* lists /*[NE][npairs]*/, int* counts) {
  __shared__ int base; __shared__ int scan[256]; const int e = blockIdx.x, tid = threadIdx.x;
  if (tid == 0) base = 0; __syncthreads();
  for (int p0 = 0; p0 < npairs; p0 += 256) {
    const int p = p0 + tid; const int hit = p < npairs && ids[p] == e ? 1 : 0;
    scan[tid] = hit; __syncthreads();
    for (int o = 1; o < 256; o <<= 1) { int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
    if (hit) lists[(size_t)e * npairs + base + scan[tid] - 1] = p;
    __syncthreads(); if (tid == 255) base += scan[255]; __syncthreads();
  }
  if (tid == 0) counts[e] = base;
}
// expert gate_up with SITU epilogue: wave per hidden row, NT pairs per block. hid[pair][f]
template <int WT, int NT>
__global__ void k_moe_gate_up(const uint8_t* W, int K, int NF, const float* X /*[T,K]*/, const int* lists, const int* counts, int npairs,
                              const float* scales /*[NE] or null*/, float* hid, float beta, float lin) {
  const int e = blockIdx.z, wave = threadIdx.x / kWave, lane = threadIdx.x % kWave; const int f = blockIdx.x * 8 + wave;
  const int cnt = counts[e]; const int p0 = blockIdx.y * NT; if (f >= NF || p0 >= cnt) return;
  const size_t rb = row_bytes_of(WT, K); const uint8_t* Wg = W + ((size_t)e * 2 * NF + f) * rb; const uint8_t* Wu = W + ((size_t)e * 2 * NF + NF + f) * rb;
  float ag[NT], au[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) { ag[i] = 0.f; au[i] = 0.f; }
  for (int it = lane; it < K / 32; it += kWave) {
    float wg[32], wu[32]; dq32<WT>(Wg, it, wg); dq32<WT>(Wu, it, wu);
#pragma unroll
    for (int i = 0; i < NT; ++i) if (p0 + i < cnt) {
      const int tok = lists[(size_t)e * npairs + p0 + i] / 8; const float* x = X + (size_t)tok * K + it * 32; float sg = 0.f, su = 0.f;
#pragma unroll
      for (int k = 0; k < 32; ++k) { sg += wg[k] * x[k]; su += wu[k] * x[k]; }
      ag[i] += sg; au[i] += su;
    }
  }
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    float g = wave_sum(ag[i]), u = wave_sum(au[i]);
    if (lane == 0 && p0 + i < cnt) {
      const float s = scales ? scales[e] : 1.f; g *= s; u *= s;
      const float act = beta * tanhf(g / beta) / (1.f + expf(-g)) * 1.f;  // beta*tanh(g/beta)*sigmoid(g)
      const float hv = act * (lin * tanhf(u / lin));
      hid[(size_t)lists[(size_t)e * npairs + p0 + i] * NF + f] = hv;
    }
  }
}
// expert down projection: dn[pair] = scale[e] * W_e . hid[pair]
template <int WT, int NT>
__global__ void k_moe_down(const uint8_t* W, int K, int N, const float* hid, const int* lists, const int* counts, int npairs, const float* scales, float* dn) {
  const int e = blockIdx.z, wave = threadIdx.x / kWave, lane = threadIdx.x % kWave; const int r = blockIdx.x * 8 + wave;
  const int cnt = counts[e]; const int p0 = blockIdx.y * NT; if (r >= N || p0 >= cnt) return;
  const uint8_t* row = W + ((size_t)e * N + r) * row_bytes_of(WT, K);
  float acc[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i] = 0.f;
  for (int it = lane; it < K / 32; it += kWave) {
    float w[32]; dq32<WT>(row, it, w);
#pragma unroll
    for (int i = 0; i < NT; ++i) if (p0 + i < cnt) {
      const float* x = hid + (size_t)lists[(size_t)e * npairs + p0 + i] * K + it * 32; float s = 0.f;
#pragma unroll
      for (int k = 0; k < 32; ++k) s += w[k] * x[k];
      acc[i] += s;
    }
  }
#pragma unroll
  for (int i = 0; i < NT; ++i) { float v = wave_sum(acc[i]); if (lane == 0 && p0 + i < cnt) dn[(size_t)lists[(size_t)e * npairs + p0 + i] * N + r] = v * (scales ? scales[e] : 1.f); }
}
__global__ void k_moe_combine(const float* dn, const float* wts, float* out, int NU, int E) {
  const int t = blockIdx.x;
  for (int i = threadIdx.x; i < E; i += blockDim.x) { float a = 0.f; for (int k = 0; k < NU; ++k) a += wts[(size_t)t * NU + k] * dn[((size_t)t * NU + k) * E + i]; out[(size_t)t * E + i] = a; }
}
__global__ void k_sigmoid_mul(float* att, const float* gate, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) att[i] *= 1.f / (1.f + expf(-gate[i])); }
__global__ void k_softcap(float* x, int n, float cap) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = cap * tanhf(x[i] / cap); }
// rows given as input embeddings (images) replace the token embedding, without the embedding norm
__global__ void k_embd_override(float* X, const float* embd, const int* mask, int E) { const int t = blockIdx.x; if (!mask[t]) return; for (int i = threadIdx.x; i < E; i += blockDim.x) X[(size_t)t * E + i] = embd[(size_t)t * E + i]; }
// copy a [T,E] block into column group `col` (of `groups`) of a [T, groups*E] feature buffer
__global__ void k_copy_cols(const float* src, float* dst, int E, int groups, int col) { const int t = blockIdx.x; for (int i = threadIdx.x; i < E; i += blockDim.x) dst[(size_t)t * groups * E + (size_t)col * E + i] = src[(size_t)t * E + i]; }
// history after token index keep-1 of a verification batch: last 3 of [old history, saved inputs 0..keep-1]
__global__ void k_conv_rollback(const float* const* old, const float* const* saved, const int* chans, int keep, float* const* hist) {
  const int sk = blockIdx.y, ch = chans[sk], c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= ch) return;
  float nw[3];
  for (int l = 0; l < 3; ++l) { const int idx = 3 + keep - 1 - l; nw[2 - l] = idx >= 3 ? saved[sk][(size_t)(idx - 3) * ch + c] : old[sk][(size_t)idx * ch + c]; }
  for (int i = 0; i < 3; ++i) hist[sk][(size_t)i * ch + c] = nw[i];
}
__global__ void k_gather_rows(const float* in, const int* rows, float* out, int E) { const int i = blockIdx.x; for (int c = threadIdx.x; c < E; c += blockDim.x) out[(size_t)i * E + c] = in[(size_t)rows[i] * E + c]; }

}  // namespace inkling::gpu
