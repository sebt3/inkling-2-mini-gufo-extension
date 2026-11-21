// Inkling-2 vision encoder on the GPU: a ViT with a CLS token, learned (bilinearly resized) position embeddings and 2D RoPE,
// 2x2 patch merging and a two-layer projector. Reference: llama.cpp tools/mtmd (inkling2v). F32 activations, BF16/F32 weights.
#pragma once
#include <cmath>
#include <functional>
#include "kernels.hpp"
#include "../core/gguf.hpp"

namespace inkling::gpu {

// patches [n_patches][KP] (KP = 1216: 3*20*20 padded with zeros) from a planar [3][H][W] image
__global__ void k_patchify(const float* pix, float* patches, int W, int H, int KP) {
  const int p = blockIdx.x, px = p % (W / 20), py = p / (W / 20);
  for (int k = threadIdx.x; k < KP; k += blockDim.x) {
    float v = 0.f;
    if (k < 1200) { const int ch = k / 400, r = k % 400, ky = r / 20, kx = r % 20; v = pix[(size_t)ch * H * W + (size_t)(py * 20 + ky) * W + px * 20 + kx]; }
    patches[(size_t)p * KP + k] = v;
  }
}
// X[0] = cls; X[t+1] = conv[t] + pos[t]
__global__ void k_vis_assemble(const float* conv, const float* pos, const float* cls, float* X, int E) {
  const int t = blockIdx.x;
  for (int e = threadIdx.x; e < E; e += blockDim.x) X[(size_t)t * E + e] = t == 0 ? cls[e] : conv[(size_t)(t - 1) * E + e] + pos[(size_t)(t - 1) * E + e];
}
__global__ void k_layernorm(const float* in, const float* w, const float* b, float* out, int E, float eps) {
  __shared__ float red[256]; const size_t t = blockIdx.x; const float* x = in + t * E; float s = 0.f;
  for (int i = threadIdx.x; i < E; i += blockDim.x) s += x[i];
  red[threadIdx.x] = s; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  const float mean = red[0] / E; __syncthreads();
  float v = 0.f; for (int i = threadIdx.x; i < E; i += blockDim.x) { const float d = x[i] - mean; v += d * d; }
  red[threadIdx.x] = v; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  const float r = rsqrtf(red[0] / E + eps);
  for (int i = threadIdx.x; i < E; i += blockDim.x) out[t * E + i] = (x[i] - mean) * r * w[i] + b[i];
}
__global__ void k_bias_add(float* Y, const float* b, int N, size_t total) { const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < total) Y[i] += b[i % N]; }
// 2D RoPE (adjacent pairs): dims [0,32) rotated by pos_w, dims [32,64) by pos_h; token 0 = CLS at (0,0), token t>=1: (x+1, y+1)
__global__ void k_rope2d(float* x, int heads, int Wp, float theta) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;  // i = pair index 0..31 over both halves
  const int half = i / 16, j = i % 16; int pw = 0, ph = 0; if (t > 0) { const int p = t - 1; pw = p % Wp + 1; ph = p / Wp + 1; }
  const float pos = half == 0 ? (float)pw : (float)ph; const float ang = pos * powf(theta, -2.f * j / 32.f); const float c = cosf(ang), s = sinf(ang);
  float* v = x + ((size_t)t * heads + h) * 64 + half * 32 + 2 * j; const float a = v[0], b = v[1]; v[0] = a * c - b * s; v[1] = a * s + b * c;
}
// bidirectional attention, d = 64, split over 128-key tiles; one block per (token, head)
__global__ void k_vit_attn(const float* q, const float* k, const float* v, float* out, int n, int heads) {
  __shared__ float qs[64], lg[128], red[128]; const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x; const int E = heads * 64;
  if (tid < 64) qs[tid] = q[(size_t)t * E + h * 64 + tid]; __syncthreads();
  float m = -1e30f, l = 0.f, acc = 0.f;
  for (int j0 = 0; j0 < n; j0 += 128) {
    const int cnt = min(128, n - j0); float s = -1e30f;
    if (tid < cnt) { const float* kr = k + (size_t)(j0 + tid) * E + h * 64; float d = 0.f; for (int i = 0; i < 64; ++i) d += qs[i] * kr[i]; s = d * 0.125f; }
    lg[tid] = s; red[tid] = s; __syncthreads();
    for (int o = 64; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    const float mn = fmaxf(m, red[0]), corr = expf(m - mn); acc *= corr; l *= corr; m = mn; __syncthreads();
    float a2 = 0.f, l2 = 0.f;
    if (tid < 64) for (int i = 0; i < cnt; ++i) { const float w = expf(lg[i] - m); l2 += w; a2 += w * v[(size_t)(j0 + i) * E + h * 64 + tid]; }
    acc += a2; l += l2; __syncthreads();
  }
  if (tid < 64) out[(size_t)t * E + h * 64 + tid] = acc / l;
}
__global__ void k_scale_residual(const float* res, const float* x, const float* ls, float* out, int E, size_t total) { const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < total) out[i] = res[i] + x[i] * ls[i % E]; }
__global__ void k_gelu_erf(float* x, size_t n) { const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { const float v = x[i]; x[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); } }
// merged token (my, mx): features [e + 1024*dx + 2048*dy] from patch (2my+dy, 2mx+dx); source rows are offset by the CLS token
__global__ void k_merge_gather(const float* vit, float* out, int Wp, int E) {
  const int tm = blockIdx.x, Wm = Wp / 2, mx = tm % Wm, my = tm / Wm;
  for (int f = threadIdx.x; f < 4 * E; f += blockDim.x) { const int e = f % E, dx = (f / E) % 2, dy = f / (2 * E); out[(size_t)tm * 4 * E + f] = vit[(size_t)(1 + (2 * my + dy) * Wp + 2 * mx + dx) * E + e]; }
}

class Vision {
 public:
  static constexpr int E = 1024, FF = 4096, NL = 23, KP = 1216;
  std::function<void(const char*, const float*, size_t)> dump;  // (name, device pointer, element count)

  explicit Vision(const std::string& path) : g_(path) {
    eps_ = (float)g_.num("clip.vision.attention.layer_norm_epsilon");
    if ((int)g_.num("clip.vision.embedding_length") != E || (int)g_.num("clip.vision.block_count") != NL || (int)g_.num("clip.vision.patch_size") != 20 ||
        (int)g_.num("clip.vision.spatial_merge_size") != 2) throw std::runtime_error("vision: unsupported projector geometry");
    n_per_side_ = (int)std::sqrt((double)g_.get("v.position_embd.weight").ne[1]);
    // patch embedding: rows padded from 1200 to KP floats
    { const Tensor& t = g_.get("v.patch_embd.weight"); std::vector<float> w((size_t)E * KP, 0.f);
      for (int o = 0; o < E; ++o) memcpy(&w[(size_t)o * KP], (const float*)t.data + (size_t)o * 1200, 1200 * 4);
      patch_w_ = up(w.data(), w.size() * 4); patch_dt_ = {patch_w_, W_F32, KP, E}; }
    pos_embd_.assign((const float*)g_.get("v.position_embd.weight").data, (const float*)g_.get("v.position_embd.weight").data + (size_t)E * n_per_side_ * n_per_side_);
    { std::vector<float> c(E); const float* a = (const float*)g_.get("v.class_embd").data; const float* b = (const float*)g_.get("v.class_position_embd.weight").data; for (int i = 0; i < E; ++i) c[i] = a[i] + b[i]; cls_ = (float*)up(c.data(), E * 4); }
    pre_w_ = f32("v.pre_ln.weight"); pre_b_ = f32("v.pre_ln.bias");
    for (int i = 0; i < NL; ++i) {
      auto n = [&](const char* s) { return "v.blk." + std::to_string(i) + "." + s; }; Blk& b = blk_[i];
      b.ln1w = f32(n("ln1.weight")); b.ln1b = f32(n("ln1.bias")); b.ln2w = f32(n("ln2.weight")); b.ln2b = f32(n("ln2.bias")); b.ls1 = f32(n("ls1.weight")); b.ls2 = f32(n("ls2.weight"));
      b.q = mat(n("attn_q.weight")); b.k = mat(n("attn_k.weight")); b.v = mat(n("attn_v.weight")); b.o = mat(n("attn_out.weight"));
      b.qb = f32(n("attn_q.bias")); b.kb = f32(n("attn_k.bias")); b.vb = f32(n("attn_v.bias")); b.ob = f32(n("attn_out.bias"));
      b.up = mat(n("ffn_up.weight")); b.dn = mat(n("ffn_down.weight")); b.upb = f32(n("ffn_up.bias")); b.dnb = f32(n("ffn_down.bias"));
    }
    mm1_ = mat("mm.1.weight"); mm2_ = mat("mm.2.weight");
    if (g_.find("mm.1.bias")) mm1b_ = f32("mm.1.bias"); if (g_.find("mm.2.bias")) mm2b_ = f32("mm.2.bias");
    out_dim_ = mm2_.N;
  }
  int out_dim() const { return out_dim_; }
  // pix: planar [3][H][W] floats already normalised; W, H multiples of 40. Returns the image embeddings [n_tok][out_dim] on the host.
  std::vector<float> encode(const std::vector<float>& pix, int W, int H, int* n_tok) {
    const int Wp = W / 20, Hp = H / 20, np = Wp * Hp, n = np + 1; *n_tok = np / 4;
    float *d_pix = dm(pix.size()), *d_patch = dm((size_t)np * KP), *d_conv = dm((size_t)np * E), *d_pos = dm((size_t)np * E), *X = dm((size_t)n * E), *Y = dm((size_t)n * E);
    float *xn = dm((size_t)n * E), *q = dm((size_t)n * E), *k = dm((size_t)n * E), *v = dm((size_t)n * E), *att = dm((size_t)n * E), *tmp = dm((size_t)n * E), *hid = dm((size_t)n * FF);
    HIP_CHECK(hipMemcpy(d_pix, pix.data(), pix.size() * 4, hipMemcpyHostToDevice));
    std::vector<float> pos = resized_pos(Wp, Hp); HIP_CHECK(hipMemcpy(d_pos, pos.data(), pos.size() * 4, hipMemcpyHostToDevice));
    k_patchify<<<np, 256>>>(d_pix, d_patch, W, H, KP); gemm(patch_dt_, d_patch, d_conv, np); dmp("patch_embd", d_conv, (size_t)np * E);
    k_vis_assemble<<<n, 256>>>(d_conv, d_pos, cls_, X, E); dmp("embeddings", X, (size_t)n * E);
    k_layernorm<<<n, 256>>>(X, pre_w_, pre_b_, Y, E, eps_); std::swap(X, Y); dmp("pre_ln", X, (size_t)n * E);
    const size_t ne = (size_t)n * E;
    for (int i = 0; i < NL; ++i) {
      Blk& b = blk_[i];
      k_layernorm<<<n, 256>>>(X, b.ln1w, b.ln1b, xn, E, eps_); if (i == 0) dmp("layer_inp_normed-0", xn, ne);
      gemm(b.q, xn, q, n); gemm(b.k, xn, k, n); gemm(b.v, xn, v, n);
      k_bias_add<<<(ne + 255) / 256, 256>>>(q, b.qb, E, ne); k_bias_add<<<(ne + 255) / 256, 256>>>(k, b.kb, E, ne); k_bias_add<<<(ne + 255) / 256, 256>>>(v, b.vb, E, ne);
      if (i == 0) { dmp("Qcur-0", q, ne); dmp("Kcur-0", k, ne); dmp("Vcur-0", v, ne); }
      k_rope2d<<<dim3(n, 16), 32>>>(q, 16, Wp, 10000.f); k_rope2d<<<dim3(n, 16), 32>>>(k, 16, Wp, 10000.f);
      if (i == 0) { dmp("Qcur_pos-0", q, ne); dmp("Kcur_pos-0", k, ne); }
      k_vit_attn<<<dim3(n, 16), 128>>>(q, k, v, att, n, 16);
      gemm(b.o, att, tmp, n); k_bias_add<<<(ne + 255) / 256, 256>>>(tmp, b.ob, E, ne);
      k_scale_residual<<<(ne + 255) / 256, 256>>>(X, tmp, b.ls1, Y, E, ne); std::swap(X, Y);   // X = X + attn * ls1
      if (i == 0) dmp("ffn_inp-0", X, ne);
      k_layernorm<<<n, 256>>>(X, b.ln2w, b.ln2b, xn, E, eps_);
      gemm(b.up, xn, hid, n); { const size_t nh = (size_t)n * FF; k_bias_add<<<(nh + 255) / 256, 256>>>(hid, b.upb, FF, nh); k_gelu_erf<<<(nh + 255) / 256, 256>>>(hid, nh); }
      gemm(b.dn, hid, tmp, n); k_bias_add<<<(ne + 255) / 256, 256>>>(tmp, b.dnb, E, ne);
      k_scale_residual<<<(ne + 255) / 256, 256>>>(X, tmp, b.ls2, Y, E, ne); std::swap(X, Y);
      if (i == 0 || i == 1 || i == 11 || i == NL - 1) dmp(("layer_out-" + std::to_string(i)).c_str(), X, ne);
    }
    float *merged = dm((size_t)(np / 4) * 4 * E), *h1 = dm((size_t)(np / 4) * 4 * E), *res = dm((size_t)(np / 4) * out_dim_);
    k_merge_gather<<<np / 4, 256>>>(X, merged, Wp, E); dmp("pixel_shuffle", merged, (size_t)(np / 4) * 4 * E);
    gemm(mm1_, merged, h1, np / 4); if (mm1b_) k_bias_add<<<((size_t)(np / 4) * mm1_.N + 255) / 256, 256>>>(h1, mm1b_, mm1_.N, (size_t)(np / 4) * mm1_.N);
    { const size_t nh = (size_t)(np / 4) * mm1_.N; k_gelu_erf<<<(nh + 255) / 256, 256>>>(h1, nh); }
    gemm(mm2_, h1, res, np / 4); if (mm2b_) k_bias_add<<<((size_t)(np / 4) * mm2_.N + 255) / 256, 256>>>(res, mm2b_, mm2_.N, (size_t)(np / 4) * mm2_.N);
    dmp("projected", res, (size_t)(np / 4) * out_dim_);
    std::vector<float> out((size_t)(np / 4) * out_dim_); HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipMemcpy(out.data(), res, out.size() * 4, hipMemcpyDeviceToHost));
    for (float* p : {d_pix, d_patch, d_conv, d_pos, X, Y, xn, q, k, v, att, tmp, hid, merged, h1, res}) hipFree(p);
    return out;
  }

 private:
  struct Blk { float *ln1w, *ln1b, *ln2w, *ln2b, *ls1, *ls2, *qb, *kb, *vb, *ob, *upb, *dnb; struct M { void* d = nullptr; int wt = W_F32, K = 0, N = 0; } q, k, v, o, up, dn; };
  using M = Blk::M;
  Gguf g_; float eps_; int n_per_side_, out_dim_; void* patch_w_; M patch_dt_; std::vector<float> pos_embd_; float *cls_, *pre_w_, *pre_b_, *mm1b_ = nullptr, *mm2b_ = nullptr; Blk blk_[NL]; M mm1_, mm2_;
  static float* dm(size_t n) { float* p; HIP_CHECK(hipMalloc(&p, n * 4)); return p; }
  static void* up(const void* s, size_t n) { void* d; HIP_CHECK(hipMalloc(&d, n)); HIP_CHECK(hipMemcpy(d, s, n, hipMemcpyHostToDevice)); return d; }
  float* f32(const std::string& n) { const Tensor& t = g_.get(n); if (t.type != T_F32) throw std::runtime_error("vision: expected F32 " + n); return (float*)up(t.data, t.nbytes()); }
  M mat(const std::string& n) { const Tensor& t = g_.get(n); M m; m.wt = t.type == T_BF16 ? W_BF16 : t.type == T_F32 ? W_F32 : -1; if (m.wt < 0) throw std::runtime_error("vision: unsupported type " + n);
    m.K = (int)t.ne[0]; m.N = (int)t.ne[1]; m.d = up(t.data, t.nbytes()); return m; }
  void gemm(const M& m, const float* X, float* Y, int T) {
    constexpr int NT = 4; dim3 g((m.N + 7) / 8, (T + NT - 1) / NT);
    if (m.wt == W_BF16) k_gemm<W_BF16, NT><<<g, 256>>>((const uint8_t*)m.d, m.K, m.N, X, Y, T, nullptr, nullptr, nullptr);
    else k_gemm<W_F32, NT><<<g, 256>>>((const uint8_t*)m.d, m.K, m.N, X, Y, T, nullptr, nullptr, nullptr);
  }
  void dmp(const char* name, const float* p, size_t n) { if (dump) { HIP_CHECK(hipDeviceSynchronize()); dump(name, p, n); } }
  // bilinear resize (align_corners = false, half-pixel centres) of the n_per_side^2 learned grid to (Wp x Hp) patches
  std::vector<float> resized_pos(int Wp, int Hp) const {
    const int S = n_per_side_; std::vector<float> out((size_t)Wp * Hp * E);
    const float sf0 = (float)Wp / S, sf1 = (float)Hp / S;
    for (int Y = 0; Y < Hp; ++Y) {
      const float y = ((float)Y + 0.5f) / sf1 - 0.5f; int y0 = (int)std::floor(y), y1 = y0 + 1; y0 = std::max(0, std::min(y0, S - 1)); y1 = std::max(0, std::min(y1, S - 1));
      const float dy = std::max(0.f, std::min(y - (float)y0, 1.f));
      for (int X = 0; X < Wp; ++X) {
        const float x = ((float)X + 0.5f) / sf0 - 0.5f; int x0 = (int)std::floor(x), x1 = x0 + 1; x0 = std::max(0, std::min(x0, S - 1)); x1 = std::max(0, std::min(x1, S - 1));
        const float dx = std::max(0.f, std::min(x - (float)x0, 1.f));
        for (int e = 0; e < E; ++e) {
          const float a = pos_embd_[(size_t)(x0 + S * y0) * E + e], b = pos_embd_[(size_t)(x1 + S * y0) * E + e], c = pos_embd_[(size_t)(x0 + S * y1) * E + e], d = pos_embd_[(size_t)(x1 + S * y1) * E + e];
          out[((size_t)Y * Wp + X) * E + e] = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy;
        }
      }
    }
    return out;
  }
};

}  // namespace inkling::gpu
