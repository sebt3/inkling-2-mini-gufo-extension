// DSpark drafter of Inkling-2-Mini (GGUF architecture `dflash`, unsloth/Inkling-2-Mini-GGUF dspark-*.gguf): a 3-layer block drafter
// that reads fused features of the target (taps 5/23/41 and the final normalised state) as extra keys and values, proposes a block of
// 8 positions (anchor + 7 mask rows) in parallel, and picks the tokens with a small recurrent Markov head on top of the target's
// lm_head. Reference: llama.cpp PR #32218 (src/models/dspark.cpp, dflash.cpp) and the DFlash drafters of the Flint extension.
// Proposals only need to be good, not exact: the target verifies every token, so a drafter detail that differs from the reference
// costs acceptance, never correctness.
#pragma once
#include "engine.hpp"

namespace inkling::gpu {

// out[t] = rms(w0*F0 + w1*F1 + w2*F2 + w3*F3) * norm, F = the 4 tap features of token t ([T][4][E])
__global__ void k_ds_fuse(const float* feat, float w0, float w1, float w2, float w3, const float* norm, float* out, int E, float eps) {
  __shared__ float red[256]; const size_t t = blockIdx.x; const float* f = feat + t * 4 * E; float ss = 0.f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) { const float v = w0 * f[i] + w1 * f[E + i] + w2 * f[2 * E + i] + w3 * f[3 * E + i]; ss += v * v; }
  red[threadIdx.x] = ss; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  const float r = rsqrtf(red[0] / E + eps);
  for (int i = threadIdx.x; i < E; i += blockDim.x) { const float v = w0 * f[i] + w1 * f[E + i] + w2 * f[2 * E + i] + w3 * f[3 * E + i]; out[t * E + i] = v * r * norm[i]; }
}
// per-head rms norm (weight w[128] or none), then NEOX rope on [x_i, x_{i+64}]; x [T][heads][128]
__global__ void k_ds_headnorm_rope(float* x, int heads, const float* w, const int* pos, float theta, float eps, int rope, int adjacent) {
  __shared__ float red[64]; const size_t row = (size_t)blockIdx.x * heads + blockIdx.y; float* v = x + row * 128; const int i = threadIdx.x;
  // elements held by this thread: NEOX pairs (i, i+64); adjacent pairs (2i, 2i+1)
  const int ia = adjacent ? 2 * i : i, ib = adjacent ? 2 * i + 1 : i + 64;
  float a = v[ia], b = v[ib]; red[i] = a * a + b * b; __syncthreads();
  for (int o = 32; o > 0; o >>= 1) { if (i < o) red[i] += red[i + o]; __syncthreads(); }
  const float r = rsqrtf(red[0] / 128.f + eps); a *= r; b *= r; if (w) { a *= w[ia]; b *= w[ib]; }
  if (rope) { const float ang = (float)pos[blockIdx.x] * powf(theta, -2.f * i / 128.f); const float c = cosf(ang), s = sinf(ang); const float na = a * c - b * s, nb = a * s + b * c; a = na; b = nb; }
  v[ia] = a; v[ib] = b;
}
__global__ void k_ds_store(const float* K, const float* V, __half* kc, __half* vc, const int* pos, int cap, int kvd) {
  const int t = blockIdx.x; const size_t o = (size_t)(pos[t] % cap) * kvd;
  for (int i = threadIdx.x; i < kvd; i += blockDim.x) { kc[o + i] = __float2half(K[(size_t)t * kvd + i]); vc[o + i] = __float2half(V[(size_t)t * kvd + i]); }
}
// block attention (non-causal inside the block). grid (rows, heads), 128 threads. keys: cache positions [lo, hi) then the block's own keys.
__global__ void k_ds_attn(const float* q, const float* kb, const float* vb, const __half* kc, const __half* vc, int cap, long lo, long hi, int rows, int NH, int NKV, float* out) {
  __shared__ float qs[128], lg[128], red[128]; const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, kv = h / (NH / NKV); const int KVD = NKV * 128;
  qs[tid] = q[((size_t)r * NH + h) * 128 + tid]; __syncthreads();
  float m = -1e30f, l = 0.f, acc = 0.f; const float scale = rsqrtf(128.f);
  const long total = (hi - lo) + rows;
  for (long j0 = 0; j0 < total; j0 += 128) {
    const int cnt = (int)min(128L, total - j0); float s = -1e30f;
    if (tid < cnt) { const long j = j0 + tid; float d = 0.f;
      if (j < hi - lo) { const __half* kr = kc + (size_t)((lo + j) % cap) * KVD + kv * 128; for (int i = 0; i < 128; ++i) d += qs[i] * __half2float(kr[i]); }
      else { const float* kr = kb + (size_t)(j - (hi - lo)) * KVD + kv * 128; for (int i = 0; i < 128; ++i) d += qs[i] * kr[i]; }
      s = d * scale; }
    lg[tid] = s; red[tid] = s; __syncthreads();
    for (int o = 64; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    const float mn = fmaxf(m, red[0]), corr = expf(m - mn); acc *= corr; l *= corr; m = mn; __syncthreads();
    float a2 = 0.f, l2 = 0.f;
    for (int i = 0; i < cnt; ++i) { const long j = j0 + i; const float w = expf(lg[i] - m); l2 += w;
      a2 += w * (j < hi - lo ? __half2float(vc[(size_t)((lo + j) % cap) * KVD + kv * 128 + tid]) : vb[(size_t)(j - (hi - lo)) * KVD + kv * 128 + tid]); }
    acc += a2; l += l2; __syncthreads();
  }
  out[((size_t)r * NH + h) * 128 + tid] = l > 0.f ? acc / l : 0.f;
}
__global__ void k_ds_swiglu(const float* g, const float* u, float* out, size_t n) { const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { const float x = g[i]; out[i] = x / (1.f + expf(-x)) * u[i]; } }
__global__ void k_ds_fill_rows(float* X, const float* row0, const float* mask, int rows, int E) { const int t = blockIdx.x; for (int i = threadIdx.x; i < E; i += blockDim.x) X[(size_t)t * E + i] = t == 0 ? row0[i] : mask[i]; }
// joint = [state(256) | markov_w1[prev] (256, bf16 row) | hidden row (E) | context (E)]
__global__ void k_ds_joint(const float* state, const uint16_t* markov, const int* prev, const float* row, const float* ctx, float* joint, int rank, int E) {
  const int tot = rank + rank + 2 * E;
  for (int i = threadIdx.x; i < tot; i += blockDim.x) {
    float v; if (i < rank) v = state[i]; else if (i < 2 * rank) v = bf2f(markov[(size_t)prev[0] * rank + (i - rank)]); else if (i < 2 * rank + E) v = row[i - 2 * rank]; else v = ctx[i - 2 * rank - E];
    joint[i] = v;
  }
}
// gate = sigmoid(c0), cand = tanh(c1), state = cand + gate*(state - cand), latent = tanh(c2)
__global__ void k_ds_gru(const float* c, float* state, float* latent, int rank) { const int i = threadIdx.x; if (i >= rank) return;
  const float gate = 1.f / (1.f + expf(-c[i])), cand = tanhf(c[rank + i]); state[i] = cand + gate * (state[i] - cand); latent[i] = tanhf(c[2 * rank + i]); }
__global__ void k_ds_add(const float* a, const float* b, float* out, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] + b[i]; }
// argmax over logits (lowest index on ties): partial per block then final
__global__ void k_ds_argmax_part(const float* x, int n, float* pv, int* pi) {
  __shared__ float rv[256]; __shared__ int ri[256]; float best = -INFINITY; int bi = n;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) { if (x[i] > best || (x[i] == best && i < bi)) { best = x[i]; bi = i; } }
  rv[threadIdx.x] = best; ri[threadIdx.x] = bi; __syncthreads();
  for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { const float a = rv[threadIdx.x], b = rv[threadIdx.x + o]; const int ia = ri[threadIdx.x], ib = ri[threadIdx.x + o]; if (b > a || (b == a && ib < ia)) { rv[threadIdx.x] = b; ri[threadIdx.x] = ib; } } __syncthreads(); }
  if (threadIdx.x == 0) { pv[blockIdx.x] = rv[0]; pi[blockIdx.x] = ri[0]; }
}
__global__ void k_ds_argmax_final(const float* pv, const int* pi, int nb, int* out, int slot) {
  __shared__ float rv[256]; __shared__ int ri[256]; float best = -INFINITY; int bi = 1 << 30;
  for (int i = threadIdx.x; i < nb; i += 256) { if (pv[i] > best || (pv[i] == best && pi[i] < bi)) { best = pv[i]; bi = pi[i]; } }
  rv[threadIdx.x] = best; ri[threadIdx.x] = bi; __syncthreads();
  for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { const float a = rv[threadIdx.x], b = rv[threadIdx.x + o]; const int ia = ri[threadIdx.x], ib = ri[threadIdx.x + o]; if (b > a || (b == a && ib < ia)) { rv[threadIdx.x] = b; ri[threadIdx.x] = ib; } } __syncthreads(); }
  if (threadIdx.x == 0) out[slot] = ri[0];
}

class DSpark {
 public:
  static constexpr int E = 2304, NL = 3, NH = 24, NKV = 8, FF = 9216, RANK = 256, BLOCK = 8, SWA_CAP = 2048, FULL_CAP = 16384, FULL_WINDOW = 8192, HN = 9;
  explicit DSpark(Engine& eng, const std::string& path) : e_(eng), g_(path) {
    adj_ = std::getenv("DS_ROPE_ADJACENT") != nullptr;
    if ((int)g_.num("dflash.block_count") != NL || (int)g_.num("dflash.embedding_length") != E || (int)g_.num("dflash.block_size") != BLOCK || (int)g_.num("dflash.dspark.markov_rank") != RANK)
      throw std::runtime_error("dspark: unsupported drafter geometry");
    eps_ = (float)g_.num("dflash.attention.layer_norm_rms_epsilon"); theta_ = (float)g_.num("dflash.rope.freq_base"); window_ = (int)g_.num("dflash.attention.sliding_window") + 1;
    const auto& pat = g_.m("dflash.attention.sliding_window_pattern").arr; for (int l = 0; l < NL; ++l) swa_[l] = pat[l] != 0;
    // softmax of the 4 tap weights of each fuse row (rows 0..2: layers, row 3: the final context vector)
    const float* fz = (const float*)g_.get("dspark_fuse.weight").data;
    for (int r = 0; r < 4; ++r) { float mx = -1e30f, s = 0.f, w[4]; for (int j = 0; j < 4; ++j) mx = std::max(mx, fz[r * 4 + j]); for (int j = 0; j < 4; ++j) { w[j] = std::exp(fz[r * 4 + j] - mx); s += w[j]; } for (int j = 0; j < 4; ++j) fuse_[r][j] = w[j] / s; }
    out_norm_ = e_.f32v(&g_.get("output_norm.weight")); dspark_norm_ = e_.f32v(&g_.get("dspark_norm.weight")); mask_ = e_.f32v(&g_.get("mask_embd.weight"));
    markov_ = (const uint16_t*)e_.up(g_.get("markov_w1.weight").data, g_.get("markov_w1.weight").nbytes());
    joint_ = e_.dt(&g_.get("joint_proj.weight")); joint_b_ = e_.f32v(&g_.get("joint_proj.bias")); hout_ = e_.dt(&g_.get("dspark_hidden_out.weight")); hout_b_ = e_.f32v(&g_.get("dspark_hidden_out.bias"));
    for (int l = 0; l < NL; ++l) { auto n = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; }; Layer& L = L_[l];
      L.attn_norm = e_.f32v(&g_.get(n("attn_norm.weight"))); L.q_norm = e_.f32v(&g_.get(n("attn_q_norm.weight"))); L.k_norm = e_.f32v(&g_.get(n("attn_k_norm.weight")));
      L.kctx_norm = e_.f32v(&g_.get(n("attn_k_ctx_norm.weight"))); L.dflare = e_.f32v(&g_.get(n("dflare_norm.weight"))); L.ffn_norm = e_.f32v(&g_.get(n("ffn_norm.weight")));
      L.q = e_.dt(&g_.get(n("attn_q.weight"))); L.k = e_.dt(&g_.get(n("attn_k.weight"))); L.v = e_.dt(&g_.get(n("attn_v.weight"))); L.o = e_.dt(&g_.get(n("attn_output.weight")));
      L.kctx = e_.dt(&g_.get(n("attn_k_ctx.weight"))); L.vctx = e_.dt(&g_.get(n("attn_v_ctx.weight")));
      L.gate = e_.dt(&g_.get(n("ffn_gate.weight"))); L.up = e_.dt(&g_.get(n("ffn_up.weight"))); L.down = e_.dt(&g_.get(n("ffn_down.weight"))); }
    const int T = Engine::MAX_T;
    fused_ = e_.dmalloc((size_t)T * E); kk_ = e_.dmalloc((size_t)T * 1024); vv_ = e_.dmalloc((size_t)T * 1024); dpos_ = ialloc(T);
    X_ = e_.dmalloc((size_t)BLOCK * E); xn_ = e_.dmalloc((size_t)BLOCK * E); qb_ = e_.dmalloc((size_t)BLOCK * 3072); kb_ = e_.dmalloc((size_t)BLOCK * 1024); vb_ = e_.dmalloc((size_t)BLOCK * 1024);
    ab_ = e_.dmalloc((size_t)BLOCK * 3072); ob_ = e_.dmalloc((size_t)BLOCK * E); gb_ = e_.dmalloc((size_t)BLOCK * FF); ub_ = e_.dmalloc((size_t)BLOCK * FF); db_ = e_.dmalloc((size_t)BLOCK * E);
    hid_ = e_.dmalloc((size_t)BLOCK * E); anchor_row_ = e_.dmalloc(E); state_ = e_.dmalloc(RANK); joint_buf_ = e_.dmalloc(2 * RANK + 2 * E); gates_ = e_.dmalloc(3 * RANK);
    latent_ = e_.dmalloc(RANK); corr_ = e_.dmalloc(E); h2_ = e_.dmalloc(E); logit_ = e_.dmalloc(e_.c.n_vocab_unpadded); pv_ = e_.dmalloc(256); pi_ = ialloc(256); prev_ = ialloc(1); toks_ = ialloc(BLOCK); ctx_vec_ = e_.dmalloc(E);
  }
  struct Seq { bool used = false; __half *k[NL] = {}, *v[NL] = {}; int cap[NL] = {}; float* hn = nullptr; std::vector<long> hn_pos; long ctx_from = 0; };
  int create_seq() {
    for (int i = 0; i < Engine::MAXSEQ; ++i) if (!s_[i].used) { Seq& s = s_[i]; s.used = true; for (int l = 0; l < NL; ++l) { s.cap[l] = swa_[l] ? SWA_CAP : FULL_CAP;
        HIP_CHECK(hipMalloc(&s.k[l], (size_t)s.cap[l] * 1024 * 2)); HIP_CHECK(hipMalloc(&s.v[l], (size_t)s.cap[l] * 1024 * 2)); }
      s.hn = e_.dmalloc((size_t)HN * E); s.hn_pos.assign(HN, -1); s.ctx_from = 0; return i; }
    return -1;
  }
  void destroy_seq(int i) { Seq& s = s_[i]; if (!s.used) return; for (int l = 0; l < NL; ++l) { hipFree(s.k[l]); hipFree(s.v[l]); } hipFree(s.hn); s = Seq{}; }
  // the target restarts at `pos` (after a reset or a cache restore): no context before it
  void reset(int i, long pos) { Seq& s = s_[i]; s.ctx_from = pos; s.hn_pos.assign(HN, -1); }
  long ctx_from(int i) const { return s_[i].ctx_from; }

  // Context of target tokens [pos0, pos0+n): fused features -> keys/values of the 3 layers; the final fused vector of the last 9 positions is kept.
  void inject(int si, long pos0, int n) {
    std::lock_guard<std::mutex> lk(e_.mu); Seq& s = s_[si]; if (!n) return;
    std::vector<int> pos(n); for (int i = 0; i < n; ++i) pos[i] = (int)(pos0 + i); HIP_CHECK(hipMemcpy(dpos_, pos.data(), n * 4, hipMemcpyHostToDevice));
    const float* feat = e_.d_feat_;
    for (int l = 0; l < NL; ++l) {
      Layer& L = L_[l]; k_ds_fuse<<<n, 256>>>(feat, fuse_[l][0], fuse_[l][1], fuse_[l][2], fuse_[l][3], L.dflare, fused_, E, eps_);
      e_.gemm(L.kctx, fused_, kk_, n); e_.gemm(L.vctx, fused_, vv_, n);
      k_ds_headnorm_rope<<<dim3(n, NKV), 64>>>(kk_, NKV, L.kctx_norm, dpos_, theta_, eps_, 1, adj_);
      k_ds_store<<<n, 256>>>(kk_, vv_, s.k[l], s.v[l], dpos_, s.cap[l], 1024);
    }
    k_ds_fuse<<<n, 256>>>(feat, fuse_[3][0], fuse_[3][1], fuse_[3][2], fuse_[3][3], dspark_norm_, fused_, E, eps_);
    for (int i = std::max(0, n - HN); i < n; ++i) { const long p = pos0 + i; HIP_CHECK(hipMemcpy(s.hn + (size_t)(p % HN) * E, fused_ + (size_t)i * E, (size_t)E * 4, hipMemcpyDeviceToDevice)); s.hn_pos[p % HN] = p; }
  }
  // Propose k (<= 7) tokens after `anchor` (at position pos0; context available up to pos0-1). Empty when the context is missing.
  std::vector<int> draft(int si, int anchor, long pos0, int k) {
    std::lock_guard<std::mutex> lk(e_.mu); Seq& s = s_[si]; std::vector<int> out;
    if (k <= 0 || pos0 < 1 || s.ctx_from > pos0 - 1 || s.hn_pos[(pos0 - 1) % HN] != pos0 - 1) return out;
    k = std::min(k, BLOCK - 1);
    // block input: anchor embedding (target tok_embd + norm) then 7 mask rows
    HIP_CHECK(hipMemcpy(dtok_, &anchor, 4, hipMemcpyHostToDevice));
    k_embed_norm<<<1, 256, (E + 256) * 4>>>(dtok_, (const uint8_t*)e_.tok_embd_.d, row_bytes_of(W_Q8_0, E), e_.tok_norm_, anchor_row_, E, e_.c.eps);
    k_ds_fill_rows<<<BLOCK, 256>>>(X_, anchor_row_, mask_, BLOCK, E);
    std::vector<int> pos(BLOCK); for (int i = 0; i < BLOCK; ++i) pos[i] = (int)(pos0 + i); HIP_CHECK(hipMemcpy(dpos_, pos.data(), BLOCK * 4, hipMemcpyHostToDevice));
    for (int l = 0; l < NL; ++l) {
      Layer& L = L_[l];
      k_rmsnorm<<<BLOCK, 256>>>(X_, L.attn_norm, xn_, E, eps_);
      e_.gemm(L.q, xn_, qb_, BLOCK); e_.gemm(L.k, xn_, kb_, BLOCK); e_.gemm(L.v, xn_, vb_, BLOCK);
      k_ds_headnorm_rope<<<dim3(BLOCK, NH), 64>>>(qb_, NH, L.q_norm, dpos_, theta_, eps_, 1, adj_);
      k_ds_headnorm_rope<<<dim3(BLOCK, NKV), 64>>>(kb_, NKV, L.k_norm, dpos_, theta_, eps_, 1, adj_);
      const long win = swa_[l] ? std::min<long>(window_ - 1, 1 << 30) : (long)FULL_WINDOW;
      const long lo = std::max<long>(s.ctx_from, pos0 - win); const long hi = pos0;
      k_ds_attn<<<dim3(BLOCK, NH), 128>>>(qb_, kb_, vb_, s.k[l], s.v[l], s.cap[l], lo, std::max(hi, lo), BLOCK, NH, NKV, ab_);
      e_.gemm(L.o, ab_, ob_, BLOCK); { const int n = BLOCK * E; k_ds_add<<<(n + 255) / 256, 256>>>(X_, ob_, ob_, n); }   // ob_ = X + attn
      k_rmsnorm<<<BLOCK, 256>>>(ob_, L.ffn_norm, xn_, E, eps_);
      e_.gemm(L.gate, xn_, gb_, BLOCK); e_.gemm(L.up, xn_, ub_, BLOCK); { const size_t n = (size_t)BLOCK * FF; k_ds_swiglu<<<(n + 255) / 256, 256>>>(gb_, ub_, gb_, n); }
      e_.gemm(L.down, gb_, db_, BLOCK); { const int n = BLOCK * E; k_ds_add<<<(n + 255) / 256, 256>>>(ob_, db_, X_, n); }
    }
    k_rmsnorm<<<BLOCK, 256>>>(X_, out_norm_, hid_, E, eps_);
    // head: 7 sequential steps; the context vector is the fused feature of the last committed token
    HIP_CHECK(hipMemset(state_, 0, RANK * 4)); HIP_CHECK(hipMemcpy(prev_, &anchor, 4, hipMemcpyHostToDevice));
    const float* ctx = s.hn + (size_t)((pos0 - 1) % HN) * E; const int V = e_.c.n_vocab_unpadded;
    for (int i = 1; i <= k; ++i) {
      const float* row = hid_ + (size_t)i * E;
      k_ds_joint<<<1, 256>>>(state_, markov_, prev_, row, ctx, joint_buf_, RANK, E);
      e_.gemm(joint_, joint_buf_, gates_, 1); k_bias_add<<<(3 * RANK + 255) / 256, 256>>>(gates_, joint_b_, 3 * RANK, 3 * RANK);
      k_ds_gru<<<1, 256>>>(gates_, state_, latent_, RANK);
      e_.gemm(hout_, latent_, corr_, 1); k_bias_add<<<(E + 255) / 256, 256>>>(corr_, hout_b_, E, E);
      k_ds_add<<<(E + 255) / 256, 256>>>(row, corr_, h2_, E);                       // lm_head(row) + lm_head(correction) = lm_head(row + correction)
      e_.gemm(e_.output_, h2_, logit_, 1, V); k_softcap<<<(V + 255) / 256, 256>>>(logit_, V, e_.c.softcap);
      k_ds_argmax_part<<<256, 256>>>(logit_, V, pv_, pi_); k_ds_argmax_final<<<1, 256>>>(pv_, pi_, 256, toks_, i - 1);
      HIP_CHECK(hipMemcpy(prev_, toks_ + (i - 1), 4, hipMemcpyDeviceToDevice));
    }
    out.resize(k); HIP_CHECK(hipMemcpy(out.data(), toks_, k * 4, hipMemcpyDeviceToHost)); return out;
  }

 private:
  struct Layer { float *attn_norm, *q_norm, *k_norm, *kctx_norm, *dflare, *ffn_norm; DTensor q, k, v, o, kctx, vctx, gate, up, down; };
  Engine& e_; Gguf g_; float eps_, theta_; int window_; int adj_ = 0; bool swa_[NL]; float fuse_[4][4]; Layer L_[NL]; Seq s_[Engine::MAXSEQ];
  float *out_norm_, *dspark_norm_, *mask_, *joint_b_, *hout_b_; const uint16_t* markov_; DTensor joint_, hout_;
  float *fused_, *kk_, *vv_, *X_, *xn_, *qb_, *kb_, *vb_, *ab_, *ob_, *gb_, *ub_, *db_, *hid_, *anchor_row_, *state_, *joint_buf_, *gates_, *latent_, *corr_, *h2_, *logit_, *pv_, *ctx_vec_;
  int *dpos_, *pi_, *prev_, *toks_; int* dtok_ = ialloc(1);
  static int* ialloc(size_t n) { int* p; HIP_CHECK(hipMalloc(&p, n * 4)); return p; }
};

}  // namespace inkling::gpu
