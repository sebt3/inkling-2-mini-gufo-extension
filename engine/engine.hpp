// Inkling-2-Mini GPU engine (HIP). Mirrors ref/ref_model.hpp, batched over tokens and sequences.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <chrono>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include "../core/gguf.hpp"
#include "kernels.hpp"
#include "wmma.hpp"
#include "flash.hpp"
#include "vision.hpp"

namespace inkling::gpu {

class DSpark;
struct DTensor { void* d = nullptr; int wt = W_F32; int K = 0, N = 0, NE = 1; };

struct Config {
  int n_embd = 2304, n_layer = 24, n_head = 24, n_head_kv = 8, d_head = 128;
  int n_expert = 128, n_expert_used = 8, n_ff_exp = 1152;
  int loop_start = 6, loop_end = 18, n_passes = 3, gate_rank = 64;
  int n_vocab = 201024, n_vocab_unpadded = 200058;
  int n_swa = 1024, n_conv = 4, block = 6, n_rel = 16, log_floor = 8192;
  float eps = 1e-6f, qk_clip = 22.627417f, rel_clip = 8.0f, log_alpha = 0.1f, situ_beta = 4.f, situ_lin = 25.f, w_scale = 8.f, softcap = 30.f;
  std::vector<bool> is_swa;
  int kv_dim() const { return n_head_kv * d_head; }
  int n_slot() const { return n_layer + (n_passes - 1) * (loop_end - loop_start); }
};

struct Batch {
  std::vector<int> tokens, seq, pos;
  std::vector<int> want_logits;  // indices (into the batch) of tokens whose logits are returned
  std::vector<int> embd_mask;    // optional, size T: 1 where embd row t replaces the token embedding (images)
  std::vector<float> embd;       // optional, T x n_embd
  bool capture_features = false; // keep the residual stream at the DSpark tap points (see feature_*)
  bool save_conv = false;        // keep what is needed to roll the convolution state back (verification batches, one sequence, T <= 16)
};

class Engine {
  friend class DSpark;
 public:
  static constexpr int MAXSEQ = 16, LOCAL_CAP = 2048, MAX_T = 1024;
  Config c;
  // debugging: called with (name, slot, device pointer, tokens, width) at the dump points of the reference
  std::function<void(const char*, int, const float*, int, int)> dump;
  // optional teacher forcing: expert ids (T*8, token order of the batch) for a slot; return false to route normally
  std::function<bool(int, const std::vector<int>&, std::vector<int>&)> force_experts;

  // max_ctx: default context of sequences created with create_seq(0). Weights are uploaded here.
  Engine(const std::string& path, int max_ctx) : g_(path), max_ctx_(max_ctx) { load_config(); upload(); alloc_scratch(); alloc_tables(); }
  ~Engine() { if (prof_) print_profile(); }

  // ---- sequences: independent KV caches and convolution histories, created on demand
  int create_seq(int max_ctx = 0) {
    max_ctx = max_ctx_;  // one context length per engine: the global KV capacity is a kernel argument
    int id = -1; for (int i = 0; i < MAXSEQ; ++i) if (!seqs_[i].used) { id = i; break; }
    if (id < 0) throw std::runtime_error("engine: no free sequence");
    SeqMem& m = seqs_[id]; m.max_ctx = max_ctx; const int NS = c.n_slot(), KV = c.kv_dim();
    m.k.assign(NS, nullptr); m.v.assign(NS, nullptr); m.h.assign(NS * 4, nullptr);
    for (int slot = 0; slot < NS; ++slot) {
      const int ip = slot < c.n_layer ? slot : c.loop_start + (slot - c.n_layer) % (c.loop_end - c.loop_start); const size_t cap = c.is_swa[ip] ? LOCAL_CAP : (size_t)max_ctx;
      HIP_CHECK(hipMalloc(&m.k[slot], cap * KV * 2)); HIP_CHECK(hipMalloc(&m.v[slot], cap * KV * 2));
      for (int kind = 0; kind < 4; ++kind) { HIP_CHECK(hipMalloc(&m.h[slot * 4 + kind], 3 * hist_ch(kind) * 4)); }
      kc_host_[slot][id] = m.k[slot]; vc_host_[slot][id] = m.v[slot]; for (int kind = 0; kind < 4; ++kind) hist_host_[slot * 4 + kind][id] = m.h[slot * 4 + kind];
    }
    m.used = true; sync_tables(); reset_sequence(id); return id;
  }
  void destroy_seq(int id) {
    SeqMem& m = seqs_[id]; if (!m.used) return; const int NS = c.n_slot();
    for (int slot = 0; slot < NS; ++slot) { hipFree(m.k[slot]); hipFree(m.v[slot]); for (int kind = 0; kind < 4; ++kind) hipFree(m.h[slot * 4 + kind]);
      kc_host_[slot][id] = nullptr; vc_host_[slot][id] = nullptr; for (int kind = 0; kind < 4; ++kind) hist_host_[slot * 4 + kind][id] = nullptr; }
    m = SeqMem{}; sync_tables();
  }
  // zero the convolution history (KV rows are simply overwritten as positions advance)
  void reset_sequence(int s) {
    for (int slot = 0; slot < c.n_slot(); ++slot) for (int k = 0; k < 4; ++k) HIP_CHECK(hipMemset(seqs_[s].h[slot * 4 + k], 0, 3 * hist_ch(k) * 4));
  }
  int seq_capacity(int slot, int s) const { const int ip = slot < c.n_layer ? slot : c.loop_start + (slot - c.n_layer) % (c.loop_end - c.loop_start); return c.is_swa[ip] ? LOCAL_CAP : seqs_[s].max_ctx; }
  bool slot_is_local(int slot) const { const int ip = slot < c.n_layer ? slot : c.loop_start + (slot - c.n_layer) % (c.loop_end - c.loop_start); return c.is_swa[ip]; }
  size_t row_bytes() const { return (size_t)c.kv_dim() * 2; }
  // K or V rows [first, end) of a cache slot, absolute row numbers (ring for local slots); host buffer (end-first)*row_bytes
  void transfer_rows(int s, int slot, bool is_v, long first, long end, uint8_t* host, bool to_host) {
    const size_t rb = row_bytes(); const long cap = seq_capacity(slot, s); uint8_t* base = (uint8_t*)(is_v ? seqs_[s].v[slot] : seqs_[s].k[slot]);
    for (long r = first; r < end;) {
      const long idx = r % cap; const long n = std::min(end - r, cap - idx);
      if (to_host) HIP_CHECK(hipMemcpy(host + (size_t)(r - first) * rb, base + (size_t)idx * rb, n * rb, hipMemcpyDeviceToHost));
      else HIP_CHECK(hipMemcpy(base + (size_t)idx * rb, host + (size_t)(r - first) * rb, n * rb, hipMemcpyHostToDevice));
      r += n;
    }
  }
  // convolution histories of all cache slots, slot-major then kind (3 x channels floats each)
  size_t private_bytes() const { size_t n = 0; for (int k = 0; k < 4; ++k) n += 3 * (size_t)hist_ch(k) * 4; return n * c.n_slot(); }
  void export_private(int s, uint8_t* out) {
    for (int slot = 0; slot < c.n_slot(); ++slot) for (int k = 0; k < 4; ++k) { const size_t b = 3 * (size_t)hist_ch(k) * 4; HIP_CHECK(hipMemcpy(out, seqs_[s].h[slot * 4 + k], b, hipMemcpyDeviceToHost)); out += b; }
  }
  void import_private(int s, const uint8_t* in) {
    for (int slot = 0; slot < c.n_slot(); ++slot) for (int k = 0; k < 4; ++k) { const size_t b = 3 * (size_t)hist_ch(k) * 4; HIP_CHECK(hipMemcpy(seqs_[s].h[slot * 4 + k], in, b, hipMemcpyHostToDevice)); in += b; }
  }
  size_t seq_bytes(int max_ctx) const {
    size_t n = 0; const size_t KV = c.kv_dim();
    for (int slot = 0; slot < c.n_slot(); ++slot) { const int ip = slot < c.n_layer ? slot : c.loop_start + (slot - c.n_layer) % (c.loop_end - c.loop_start); n += (c.is_swa[ip] ? (size_t)LOCAL_CAP : (size_t)max_ctx) * KV * 2 * 2; }
    return n + private_bytes();
  }
  size_t weight_bytes() const { return weight_bytes_; }
  int max_ctx() const { return max_ctx_; }
  const float* features_dev() const { return d_feat_; }
  // after a verification batch (Batch::save_conv): restore the convolution histories as they were after the first `keep` batch tokens
  void rollback_conv(int seq, int keep) {
    if (keep < 1 || keep > save_T_) throw std::runtime_error("engine: bad rollback length");
    const int NS = c.n_slot(); std::vector<const float*> op, vp; std::vector<float*> hp; std::vector<int> ch;
    for (int slot = 0; slot < NS; ++slot) for (int k = 0; k < 4; ++k) { op.push_back(osave_[slot * 4 + k]); vp.push_back(vsave_[slot * 4 + k]); hp.push_back(hist_host_[slot * 4 + k][seq]); ch.push_back(hist_ch(k)); }
    const float **dop, **dvp; float** dhp; int* dch; const int n = NS * 4;
    HIP_CHECK(hipMalloc(&dop, n * 8)); HIP_CHECK(hipMalloc(&dvp, n * 8)); HIP_CHECK(hipMalloc(&dhp, n * 8)); HIP_CHECK(hipMalloc(&dch, n * 4));
    HIP_CHECK(hipMemcpy(dop, op.data(), n * 8, hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(dvp, vp.data(), n * 8, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dhp, hp.data(), n * 8, hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(dch, ch.data(), n * 4, hipMemcpyHostToDevice));
    k_conv_rollback<<<dim3((2304 + 127) / 128, n), 128>>>(dop, dvp, dch, keep, dhp);
    HIP_CHECK(hipDeviceSynchronize()); hipFree(dop); hipFree(dvp); hipFree(dhp); hipFree(dch);
  }
  void load_vision(const std::string& path) { vision_ = std::make_unique<Vision>(path); }
  bool has_vision() const { return (bool)vision_; }
  int vision_dim() const { return vision_ ? vision_->out_dim() : 0; }
  std::vector<float> encode_image(const std::vector<float>& pix, int W, int H, int* n_tok) { std::lock_guard<std::mutex> lk(mu); return vision_->encode(pix, W, H, n_tok); }
  std::mutex mu;  // serialises step(): one device stream
  // INKLING_PROFILE=1: accumulated wall time per kernel group (synchronises after every group; for relative breakdowns only)
  bool prof_ = std::getenv("INKLING_PROFILE") != nullptr; std::map<std::string, std::pair<double, long>> prof_ms_;
  template <class F> void timed(const char* name, F f) {
    if (!prof_) { f(); return; }
    HIP_CHECK(hipDeviceSynchronize()); auto t0 = std::chrono::steady_clock::now(); f(); HIP_CHECK(hipDeviceSynchronize());
    auto& e = prof_ms_[name]; e.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); e.second++;
  }
  void print_profile() { double tot = 0; for (auto& [k, v] : prof_ms_) tot += v.first; for (auto& [k, v] : prof_ms_) fprintf(stderr, "[profile] %-14s %9.2f ms %6.1f%% (%ld calls, %.3f ms each)\n", k.c_str(), v.first, 100 * v.first / tot, v.second, v.first / v.second); }

  // run one batch; returns logits (n_vocab_unpadded each) for batch.want_logits
  std::vector<float> step(const Batch& b);

 private:
  Gguf g_; int max_ctx_;
  struct L {
    DTensor wq, wk, wv, wo, gate, rel, kcp, vcp, sink, gate_inp, gate_up, down; DTensor rbpad;
    float *q_clip = 0, *rel_bias = 0, *rel_clip = 0, *k_conv = 0, *v_conv = 0, *post_conv = 0, *ffn_post_conv = 0, *attn_norm = 0, *q_inp = 0, *k_inp = 0, *v_inp = 0,
          *attn_res = 0, *ffn_res = 0, *ffn_norm = 0, *probs_b = 0, *gate_up_s = 0, *down_s = 0, *msg_query = 0;
    DTensor c_down[4], c_gain[4], c_forget[4];
    bool has_loop = false;
  };
  std::vector<L> layers_;
  DTensor tok_embd_, output_; float *tok_norm_ = 0, *out_norm_ = 0, *out_res_ = 0;
  // per-sequence state
  struct SeqMem { bool used = false; int max_ctx = 0; std::vector<__half*> k, v; std::vector<float*> h; };
  SeqMem seqs_[MAXSEQ];
  size_t weight_bytes_ = 0; std::unique_ptr<Vision> vision_;
  // pointer tables indexed by sequence: host mirrors and device copies, per slot (and slot*4+kind for histories)
  std::vector<std::vector<__half*>> kc_host_, vc_host_; std::vector<std::vector<float*>> hist_host_;
  std::vector<__half**> kc_tab_, vc_tab_; std::vector<float**> hist_tab_;
  // scratch (max batch)
  float *xn_[3], *q_, *k0_, *v0_, *r_, *sinks_, *gate_, *kd_, *vd_, *K_, *V_, *att_, *o_, *tmp_e_, *tmp_e2_, *logits_gate_, *moe_, *wts_, *hid_, *dn_, *ctrl_gate_, *ctrl_low_, *ctrl_a_, *ctrl_b_;
  float* d_embd_ = nullptr; int* d_emask_ = nullptr; float *bias_tab_ = nullptr, *rpad_ = nullptr;   // flash attention: bias table [T][NH][1024], padded r
  float* d_feat_ = nullptr;                                   // [MAX_T][4*E] tap features of the last captured batch
  std::vector<float*> vsave_, osave_; bool save_conv_ = false; int save_seq_ = 0, save_T_ = 0;  // rollback data, per (slot, kind)
  float *pm_, *pl_, *pacc_; int *forced_ = nullptr; int *ids_, *lists_, *counts_;
  int *d_tok_, *d_seq_, *d_pos_, *d_first_, *d_count_, *d_rows_; float* d_logit_in_; float* d_logits_;
  std::vector<float*> pool_; std::vector<int> pool_rc_;

  int hist_ch(int kind) const { return kind < 2 ? c.kv_dim() : c.n_embd; }

  void load_config() {
    const std::string a = "inkling2_loop."; auto n = [&](const char* k) { return (int)g_.num(a + k); }; auto f = [&](const char* k) { return (float)g_.num(a + k); };
    c.n_embd = n("embedding_length"); c.n_layer = n("block_count"); c.n_head = (int)g_.m(a + "attention.head_count").arr[0]; c.n_head_kv = (int)g_.m(a + "attention.head_count_kv").arr[0];
    c.d_head = n("attention.key_length"); c.n_expert = n("expert_count"); c.n_expert_used = n("expert_used_count"); c.n_ff_exp = n("expert_feed_forward_length");
    c.loop_start = n("loop.start"); c.loop_end = n("loop.end"); c.n_passes = n("loop.count"); c.gate_rank = n("loop.gate_rank");
    c.n_vocab = n("vocab_size"); c.n_vocab_unpadded = n("vocab_size_unpadded"); c.n_swa = n("attention.sliding_window"); c.n_conv = n("shortconv.l_cache");
    c.block = n("attn_res.block_size"); c.n_rel = n("attention.relative_length"); c.log_floor = n("attention.log_scale_floor");
    c.eps = f("attention.layer_norm_rms_epsilon"); c.qk_clip = f("attention.qk_clip"); c.rel_clip = f("attention.relative_clip"); c.log_alpha = f("attention.log_scale_alpha");
    c.situ_beta = f("activation.situ_beta"); c.situ_lin = f("activation.situ_linear_beta"); c.w_scale = f("expert_weights_scale"); c.softcap = f("final_logit_softcapping");
    for (double v : g_.m(a + "attention.sliding_window_pattern").arr) c.is_swa.push_back(v != 0);
    if (c.n_conv != 4 || c.d_head != 128 || c.n_expert != 128 || c.n_expert_used != 8) throw std::runtime_error("engine: unsupported model dimensions");
  }

  static int wtype(uint32_t t) {
    switch (t) { case T_F32: return W_F32; case T_BF16: return W_BF16; case T_Q8_0: return W_Q8_0; case T_Q5_K: return W_Q5_K; case T_NVFP4: return W_NVFP4; }
    throw std::runtime_error(std::string("engine: unsupported tensor type ") + type_name(t));
  }
  void* up(const void* src, size_t n) {
    void* d; HIP_CHECK(hipMalloc(&d, n)); weight_bytes_ += n; const size_t CH = 256u << 20;
    for (size_t o = 0; o < n; o += CH) HIP_CHECK(hipMemcpy((char*)d + o, (const char*)src + o, std::min(CH, n - o), hipMemcpyHostToDevice));
    return d;
  }
  DTensor dt(const Tensor* t) {
    DTensor d; d.wt = wtype(t->type); d.K = (int)t->ne[0]; d.N = t->ne.size() > 1 ? (int)t->ne[1] : 1; d.NE = t->ne.size() > 2 ? (int)t->ne[2] : 1; d.d = up(t->data, t->nbytes()); return d;
  }
  float* f32v(const Tensor* t) { if (!t) return nullptr; if (t->type != T_F32) throw std::runtime_error("expected F32 tensor " + t->name); return (float*)up(t->data, t->nbytes()); }

  void upload() {
    tok_embd_ = dt(&g_.get("token_embd.weight")); output_ = dt(&g_.get("output.weight"));
    tok_norm_ = f32v(&g_.get("token_embd_norm.weight")); out_norm_ = f32v(&g_.get("output_norm.weight")); out_res_ = f32v(&g_.get("output_res_query.weight"));
    layers_.resize(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) {
      auto& Ly = layers_[i]; auto n = [&](const char* s) { return "blk." + std::to_string(i) + "." + s; }; auto opt = [&](const char* s) { return g_.find(n(s)); };
      Ly.wq = dt(&g_.get(n("attn_q.weight"))); Ly.wk = dt(&g_.get(n("attn_k.weight"))); Ly.wv = dt(&g_.get(n("attn_v.weight"))); Ly.wo = dt(&g_.get(n("attn_output.weight")));
      Ly.gate = dt(&g_.get(n("attn_gate.weight"))); Ly.rel = dt(&g_.get(n("attn_rel.weight"))); Ly.sink = dt(&g_.get(n("attn_sink_proj.weight")));
      if (opt("attn_k_conv_proj.weight")) { Ly.kcp = dt(opt("attn_k_conv_proj.weight")); Ly.vcp = dt(opt("attn_v_conv_proj.weight")); }
      Ly.gate_inp = dt(&g_.get(n("ffn_gate_inp.weight"))); Ly.gate_up = dt(&g_.get(n("ffn_gate_up_exps.weight"))); Ly.down = dt(&g_.get(n("ffn_down_exps.weight")));
      { const Tensor* rbT = &g_.get(n("attn_rel_bias.weight")); float* raw = f32v(rbT); float* pad; HIP_CHECK(hipMalloc(&pad, (size_t)1024 * 32 * 4)); k_pad_rb<<<(1024 * 32 + 255) / 256, 256>>>(raw, pad, 1024); HIP_CHECK(hipDeviceSynchronize()); hipFree(raw); Ly.rbpad.d = pad; Ly.rbpad.wt = W_F32; Ly.rbpad.K = 32; Ly.rbpad.N = 1024; }
      Ly.q_clip = f32v(&g_.get(n("attn_q_clip.weight"))); Ly.rel_bias = f32v(&g_.get(n("attn_rel_bias.weight"))); Ly.rel_clip = f32v(&g_.get(n("attn_rel_clip.weight")));
      Ly.k_conv = f32v(&g_.get(n("attn_k_conv.weight"))); Ly.v_conv = f32v(&g_.get(n("attn_v_conv.weight"))); Ly.post_conv = f32v(&g_.get(n("attn_post_conv.weight")));
      Ly.ffn_post_conv = f32v(&g_.get(n("ffn_post_conv.weight"))); Ly.attn_norm = f32v(opt("attn_norm.weight")); Ly.q_inp = f32v(opt("attn_q_inp_norm.weight"));
      Ly.k_inp = f32v(opt("attn_k_inp_norm.weight")); Ly.v_inp = f32v(opt("attn_v_inp_norm.weight")); Ly.attn_res = f32v(opt("attn_res_query.weight"));
      Ly.ffn_res = f32v(&g_.get(n("ffn_res_query.weight"))); Ly.ffn_norm = f32v(&g_.get(n("ffn_norm.weight"))); Ly.probs_b = f32v(opt("exp_probs_b.bias"));
      Ly.gate_up_s = f32v(opt("ffn_gate_up_exps.scale")); Ly.down_s = f32v(opt("ffn_down_exps.scale"));
      if (opt("loop_msg_query.weight")) {
        Ly.has_loop = true; Ly.msg_query = f32v(opt("loop_msg_query.weight")); static const char* cn[4] = {"q", "k", "v", "ffn"};
        for (int k = 0; k < 4; ++k) {
          Ly.c_down[k] = dt(opt((std::string("loop_") + cn[k] + "_ctrl_down.weight").c_str())); Ly.c_gain[k] = dt(opt((std::string("loop_") + cn[k] + "_ctrl_gain.weight").c_str()));
          Ly.c_forget[k] = dt(opt((std::string("loop_") + cn[k] + "_ctrl_forget.weight").c_str()));
        }
      }
    }
  }

  float* dmalloc(size_t nfloats) { float* p; HIP_CHECK(hipMalloc(&p, nfloats * 4)); return p; }
  void alloc_scratch() {
    const int T = MAX_T, E = c.n_embd, KV = c.kv_dim(), QD = c.n_head * c.d_head;
    for (int i = 0; i < 3; ++i) xn_[i] = dmalloc((size_t)T * E);
    q_ = dmalloc((size_t)T * QD); k0_ = dmalloc((size_t)T * KV); v0_ = dmalloc((size_t)T * KV); r_ = dmalloc((size_t)T * c.n_head * c.n_rel); sinks_ = dmalloc((size_t)T * c.n_head);
    gate_ = dmalloc((size_t)T * QD); kd_ = dmalloc((size_t)T * 32); vd_ = dmalloc((size_t)T * 32); K_ = dmalloc((size_t)T * KV); V_ = dmalloc((size_t)T * KV);
    att_ = dmalloc((size_t)T * QD); o_ = dmalloc((size_t)T * E); tmp_e_ = dmalloc((size_t)T * E); tmp_e2_ = dmalloc((size_t)T * E);
    logits_gate_ = dmalloc((size_t)T * c.n_expert); moe_ = dmalloc((size_t)T * E); wts_ = dmalloc((size_t)T * 8); hid_ = dmalloc((size_t)T * 8 * c.n_ff_exp); dn_ = dmalloc((size_t)T * 8 * E);
    ctrl_gate_ = dmalloc((size_t)T * 2 * E); ctrl_low_ = dmalloc((size_t)T * c.gate_rank); ctrl_a_ = dmalloc((size_t)T * E); ctrl_b_ = dmalloc((size_t)T * E);
    const int NSPLIT = 16; pm_ = dmalloc((size_t)T * c.n_head * NSPLIT); pl_ = dmalloc((size_t)T * c.n_head * NSPLIT); pacc_ = dmalloc((size_t)T * c.n_head * NSPLIT * 128);
    HIP_CHECK(hipMalloc(&ids_, (size_t)T * 8 * 4)); HIP_CHECK(hipMalloc(&forced_, (size_t)T * 8 * 4)); HIP_CHECK(hipMalloc(&lists_, (size_t)c.n_expert * T * 8 * 4)); HIP_CHECK(hipMalloc(&counts_, c.n_expert * 4));
    HIP_CHECK(hipMalloc(&d_tok_, T * 4)); HIP_CHECK(hipMalloc(&d_seq_, T * 4)); HIP_CHECK(hipMalloc(&d_pos_, T * 4)); HIP_CHECK(hipMalloc(&d_first_, MAXSEQ * 4)); HIP_CHECK(hipMalloc(&d_count_, MAXSEQ * 4));
    HIP_CHECK(hipMalloc(&d_rows_, T * 4)); bias_tab_ = dmalloc((size_t)T * c.n_head * 1024); rpad_ = dmalloc((size_t)T * c.n_head * 32); d_feat_ = dmalloc((size_t)T * 4 * E); d_embd_ = dmalloc((size_t)T * E); HIP_CHECK(hipMalloc(&d_emask_, T * 4)); d_logit_in_ = dmalloc((size_t)T * E); d_logits_ = dmalloc((size_t)MAXSEQ * 4 * c.n_vocab_unpadded);
  }
  void alloc_tables() {
    const int NS = c.n_slot(); kc_host_.assign(NS, std::vector<__half*>(MAXSEQ, nullptr)); vc_host_.assign(NS, std::vector<__half*>(MAXSEQ, nullptr)); hist_host_.assign(NS * 4, std::vector<float*>(MAXSEQ, nullptr));
    for (int slot = 0; slot < NS; ++slot) {
      __half** dk; __half** dv; HIP_CHECK(hipMalloc(&dk, MAXSEQ * 8)); HIP_CHECK(hipMalloc(&dv, MAXSEQ * 8)); kc_tab_.push_back(dk); vc_tab_.push_back(dv);
      for (int kind = 0; kind < 4; ++kind) { float** dh; HIP_CHECK(hipMalloc(&dh, MAXSEQ * 8)); hist_tab_.push_back(dh); }
    }
    for (int slot = 0; slot < NS; ++slot) for (int kind = 0; kind < 4; ++kind) { vsave_.push_back(dmalloc((size_t)16 * hist_ch(kind))); osave_.push_back(dmalloc((size_t)3 * hist_ch(kind))); }
    sync_tables();
  }
  void sync_tables() {
    for (int slot = 0; slot < c.n_slot(); ++slot) {
      HIP_CHECK(hipMemcpy(kc_tab_[slot], kc_host_[slot].data(), MAXSEQ * 8, hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(vc_tab_[slot], vc_host_[slot].data(), MAXSEQ * 8, hipMemcpyHostToDevice));
      for (int kind = 0; kind < 4; ++kind) HIP_CHECK(hipMemcpy(hist_tab_[slot * 4 + kind], hist_host_[slot * 4 + kind].data(), MAXSEQ * 8, hipMemcpyHostToDevice));
    }
  }

  // ---- buffer pool of [T,E] float buffers with reference counts
  float* pget() { for (size_t i = 0; i < pool_.size(); ++i) if (pool_rc_[i] == 0) { pool_rc_[i] = 1; return pool_[i]; } pool_.push_back(dmalloc((size_t)MAX_T * c.n_embd)); pool_rc_.push_back(1); return pool_.back(); }
  void pinc(float* p) { for (size_t i = 0; i < pool_.size(); ++i) if (pool_[i] == p) { ++pool_rc_[i]; return; } }
  void pdec(float* p) { for (size_t i = 0; i < pool_.size(); ++i) if (pool_[i] == p) { --pool_rc_[i]; return; } }

  // ---- launch helpers
  int T_ = 0, NSEQ_ = 0, maxpos_ = 0, step_cap_ = 0; std::vector<int> pos_h_;
  template <int WT> void gemm_t(const DTensor& W, const float* X, float* Y, int T, int N) {
    constexpr int NT = 4; dim3 grid((N + 7) / 8, (T + NT - 1) / NT);
    k_gemm<WT, NT><<<grid, 256>>>((const uint8_t*)W.d, W.K, N, X, Y, T, nullptr, nullptr, nullptr);
  }
  template <int WT> void gemm_wmma(const DTensor& W, const float* X, float* Y, int T, int N) {
    k_wmma_gemm<WT><<<dim3((N + 63) / 64, (T + 63) / 64), 128>>>((const uint8_t*)W.d, W.K, N, X, Y, N, T, nullptr, nullptr, 1.f);
  }
  bool use_wmma = std::getenv("INKLING_NO_WMMA") == nullptr; bool use_flash_ = std::getenv("INKLING_NO_FLASH") == nullptr; bool single_seq_ = false; int d_seq0_ = 0;
  // exact: keep the scalar f32 path (the router: a rounding there would flip expert choices)
  void gemm(const DTensor& W, const float* X, float* Y, int T, int N = -1, bool exact = false) {
    if (N < 0) N = W.N;
    if (use_wmma && T >= 16 && !exact && W.K % 32 == 0) {
      switch (W.wt) { case W_F32: gemm_wmma<W_F32>(W, X, Y, T, N); return; case W_BF16: gemm_wmma<W_BF16>(W, X, Y, T, N); return; case W_Q8_0: gemm_wmma<W_Q8_0>(W, X, Y, T, N); return; case W_Q5_K: gemm_wmma<W_Q5_K>(W, X, Y, T, N); return; case W_NVFP4: gemm_wmma<W_NVFP4>(W, X, Y, T, N); return; }
    }
    switch (W.wt) { case W_F32: gemm_t<W_F32>(W, X, Y, T, N); break; case W_BF16: gemm_t<W_BF16>(W, X, Y, T, N); break; case W_Q8_0: gemm_t<W_Q8_0>(W, X, Y, T, N); break;
      case W_Q5_K: gemm_t<W_Q5_K>(W, X, Y, T, N); break; case W_NVFP4: gemm_t<W_NVFP4>(W, X, Y, T, N); break; }
  }
  void rms(const float* in, const float* w, float* out, int T) { timed("rmsnorm", [&] { k_rmsnorm<<<T, 256>>>(in, w, out, c.n_embd, c.eps); }); }
  void mix(const std::vector<float*>& blocks, float* cur, const float* query, int nq, std::vector<float*>& outs) {
    MixArgs a{}; a.ns = (int)blocks.size() + 1; for (size_t i = 0; i < blocks.size(); ++i) a.st[i] = blocks[i]; a.st[blocks.size()] = cur; a.nq = nq;
    outs.clear(); for (int q = 0; q < nq; ++q) { outs.push_back(pget()); a.out[q] = outs.back(); }
    timed("mix", [&] { k_mix<<<T_, 256>>>(a, query, c.n_embd, c.eps); });
  }
  float* control(float* cur, float* msg, const L& Ly, int ctrl) {
    const int E = c.n_embd, R = c.gate_rank; float* out = pget(); const int n = T_ * E;
    timed("control", [&] { k_control_gate<<<T_, 256>>>(cur, msg, ctrl_gate_, E, c.eps);
      gemm(Ly.c_down[ctrl], ctrl_gate_, ctrl_low_, T_); gemm(Ly.c_gain[ctrl], ctrl_low_, ctrl_a_, T_); gemm(Ly.c_forget[ctrl], ctrl_low_, ctrl_b_, T_);
      k_control_apply<<<(n + 255) / 256, 256>>>(cur, msg, ctrl_a_, ctrl_b_, out, n); });
    (void)R; return out;
  }
  void conv(const float* x, const float* w, const float* dyn, int slot, int kind, float* out, int ch) {
    dim3 g((ch + 127) / 128, T_);
    k_conv<<<g, 128>>>(x, w, dyn, dyn ? c.n_head_kv : 0, d_seq_, d_pos_, d_first_, hist_tab_[slot * 4 + kind], out, T_, ch);
    if (save_conv_) {  // single-sequence verification batch: keep the old history and the inputs
      HIP_CHECK(hipMemcpy(osave_[slot * 4 + kind], hist_host_[slot * 4 + kind][save_seq_], (size_t)3 * ch * 4, hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(vsave_[slot * 4 + kind], x, (size_t)T_ * ch * 4, hipMemcpyDeviceToDevice));
    }
    dim3 g2((ch + 127) / 128, MAXSEQ); k_conv_state<<<g2, 128>>>(x, d_first_, d_count_, hist_tab_[slot * 4 + kind], ch);
  }
  void dmp(const char* name, int slot, const float* p, int width) { if (dump) { HIP_CHECK(hipDeviceSynchronize()); dump(name, slot, p, T_, width); } }

  float* layer(int slot, int ip, float* cur, std::vector<float*>& blocks, const std::vector<float*>* msgs, const L* loop);
};

inline float* Engine::layer(int slot, int ip, float* cur, std::vector<float*>& blocks, const std::vector<float*>* msgs, const L* loop) {
  const L& Ly = layers_[ip]; const int E = c.n_embd, NH = c.n_head, NKV = c.n_head_kv, D = c.d_head, KV = c.kv_dim(), T = T_; const bool swa = c.is_swa[ip];
  float* inpSA = cur;
  float* in[3] = {cur, cur, cur}; std::vector<float*> owned;
  if (Ly.attn_res) {
    std::vector<float*> mixed; mix(blocks, cur, Ly.attn_res, swa ? 1 : 3, mixed);
    for (int j = 0; j < 3; ++j) in[j] = mixed[swa ? 0 : j];
    for (float* m : mixed) owned.push_back(m);
  }
  if (msgs) for (int j = 0; j < 3; ++j) { float* o = control(in[j], (*msgs)[j], *loop, j); owned.push_back(o); in[j] = o; }
  float *x0 = xn_[0], *x1 = xn_[1], *x2 = xn_[2];
  if (swa) { rms(in[0], Ly.attn_norm, x0, T); if (msgs) { rms(in[1], Ly.attn_norm, x1, T); rms(in[2], Ly.attn_norm, x2, T); } else { x1 = x0; x2 = x0; } }
  else { rms(in[0], Ly.q_inp, x0, T); rms(in[1], Ly.k_inp, x1, T); rms(in[2], Ly.v_inp, x2, T); }
  for (float* m : owned) pdec(m);
  timed("proj_qkv_rel_gate", [&] { gemm(Ly.wq, x0, q_, T); gemm(Ly.wk, x1, k0_, T); gemm(Ly.wv, x2, v0_, T); gemm(Ly.rel, x0, r_, T); gemm(Ly.sink, x0, sinks_, T); gemm(Ly.gate, x0, gate_, T);
    if (!swa) { gemm(Ly.kcp, x1, kd_, T); gemm(Ly.vcp, x2, vd_, T); } });
  timed("conv_kv", [&] { conv(k0_, Ly.k_conv, swa ? nullptr : kd_, slot, 0, K_, KV); conv(v0_, Ly.v_conv, swa ? nullptr : vd_, slot, 1, V_, KV); });
  timed("clip_norm_tau", [&] { k_clip<<<T * NH, 128>>>(q_, D, Ly.q_clip, NH, c.qk_clip); k_clip<<<T * NKV, 128>>>(K_, D, nullptr, NKV, c.qk_clip); k_rms_head<<<T * NKV, 128>>>(V_, D, c.eps);
    k_clip<<<T * NH, 128>>>(r_, c.n_rel, Ly.rel_clip, NH, c.rel_clip);
    if (!swa) k_tau<<<T, 256>>>(q_, NH * D, r_, NH * c.n_rel, d_pos_, (float)c.log_floor, c.log_alpha); });
  dmp("Qcur", slot, q_, NH * D); dmp("Kcur", slot, K_, KV); dmp("Vcur", slot, V_, KV);
  const int cap = swa ? LOCAL_CAP : step_cap_;
  timed("kv_store", [&] { k_kv_store<<<T, 128>>>(K_, V_, kc_tab_[slot], vc_tab_[slot], d_seq_, d_pos_, cap, KV); });
  int nsplit = 1, chunk = 1 << 30;
  if (!swa) { const int len = maxpos_ + 1; chunk = std::max(2048, (len + 15) / 16); nsplit = (len + chunk - 1) / chunk; }
  else { chunk = c.n_swa; nsplit = 1; }
  const bool flash = use_flash_ && T >= 16 && single_seq_;
  if (flash) {
    timed(swa ? "attn_local" : "attn_global", [&] {
      k_bias_pad<<<((size_t)T * NH * 32 + 255) / 256, 256>>>(r_, rpad_, (size_t)T * NH);
      k_wmma_gemm<W_F32><<<dim3(1024 / 64, (T * NH + 63) / 64), 128>>>((const uint8_t*)Ly.rbpad.d, 32, 1024, rpad_, bias_tab_, 1024, T * NH, nullptr, nullptr, 1.f);
      k_flash_attn<<<dim3((T + 63) / 64, NH), 128>>>(q_, (const __half*)kc_host_[slot][d_seq0_], (const __half*)vc_host_[slot][d_seq0_], bias_tab_, sinks_, gate_, d_pos_, T, NH, NKV, cap, swa ? c.n_swa : 0, att_);
    });
  } else
  {
  timed(swa ? "attn_local" : "attn_global", [&] {
  k_attn_part<<<dim3(T * NH, nsplit), 128>>>(q_, r_, sinks_, kc_tab_[slot] ? (const __half* const*)kc_tab_[slot] : nullptr, (const __half* const*)vc_tab_[slot], Ly.rel_bias, d_seq_, d_pos_, NH, NKV, D, c.n_rel, cap,
                                            swa ? c.n_swa : 0, chunk, pm_, pl_, pacc_, nsplit);
  k_attn_combine<<<T * NH, 128>>>(pm_, pl_, pacc_, gate_, att_, NH, D, nsplit);
  });
  }
  timed("wo_conv", [&] { gemm(Ly.wo, att_, o_, T); });
  float* A = pget(); timed("wo_conv", [&] { conv(o_, Ly.post_conv, nullptr, slot, 2, A, E); });
  dmp("attn_out", slot, A, E);
  if (ip % c.block == 0) { blocks.push_back(inpSA); pinc(inpSA); cur = A; }
  else { float* C = pget(); const int n = T * E; k_add<<<(n + 255) / 256, 256>>>(inpSA, A, C, n); pdec(A); cur = C; }
  float* ffn_inp = cur; dmp("ffn_inp", slot, ffn_inp, E);
  std::vector<float*> fm; mix(blocks, cur, Ly.ffn_res, 1, fm); float* xm = fm[0];
  if (msgs) { float* o = control(xm, (*msgs)[3], *loop, 3); pdec(xm); xm = o; }
  rms(xm, Ly.ffn_norm, tmp_e_, T); pdec(xm);
  // MoE
  const int NE = c.n_expert, NF = c.n_ff_exp, npairs = T * 8;
  timed("moe_route", [&] {
  gemm(Ly.gate_inp, tmp_e_, logits_gate_, T, -1, true);
  const int* forced = nullptr;
  if (force_experts) { std::vector<int> f; if (force_experts(slot, pos_h_, f) && (int)f.size() == T * 8) { HIP_CHECK(hipMemcpy(forced_, f.data(), f.size() * 4, hipMemcpyHostToDevice)); forced = forced_; } }
  k_route<<<T, 128>>>(logits_gate_, Ly.probs_b, NE, 8, c.w_scale, ids_, wts_, forced);
  k_moe_lists<<<NE, 256>>>(ids_, npairs, lists_, counts_);
  });
  timed("moe_experts", [&] {
  if (use_wmma && T >= 32) {   // grouped matrix-core GEMMs: ~64 tokens per expert tile
    dim3 g((NF + 31) / 32, (T + 63) / 64, NE), g2((E + 63) / 64, (T + 63) / 64, NE);
    if (Ly.gate_up.wt == W_NVFP4) k_wmma_moe_gate_up<W_NVFP4><<<g, 128>>>((const uint8_t*)Ly.gate_up.d, E, NF, tmp_e_, lists_, counts_, npairs, Ly.gate_up_s, hid_, c.situ_beta, c.situ_lin);
    else if (Ly.gate_up.wt == W_Q5_K) k_wmma_moe_gate_up<W_Q5_K><<<g, 128>>>((const uint8_t*)Ly.gate_up.d, E, NF, tmp_e_, lists_, counts_, npairs, Ly.gate_up_s, hid_, c.situ_beta, c.situ_lin);
    else throw std::runtime_error("engine: unsupported gate_up type");
    if (Ly.down.wt == W_NVFP4) k_wmma_moe_down<W_NVFP4><<<g2, 128>>>((const uint8_t*)Ly.down.d, NF, E, hid_, lists_, counts_, npairs, Ly.down_s, dn_);
    else if (Ly.down.wt == W_Q8_0) k_wmma_moe_down<W_Q8_0><<<g2, 128>>>((const uint8_t*)Ly.down.d, NF, E, hid_, lists_, counts_, npairs, Ly.down_s, dn_);
    else throw std::runtime_error("engine: unsupported down type");
  } else
  { constexpr int NT = 4; dim3 g((NF + 7) / 8, (T + NT - 1) / NT, NE);
    if (Ly.gate_up.wt == W_NVFP4) k_moe_gate_up<W_NVFP4, NT><<<g, 256>>>((const uint8_t*)Ly.gate_up.d, E, NF, tmp_e_, lists_, counts_, npairs, Ly.gate_up_s, hid_, c.situ_beta, c.situ_lin);
    else if (Ly.gate_up.wt == W_Q5_K) k_moe_gate_up<W_Q5_K, NT><<<g, 256>>>((const uint8_t*)Ly.gate_up.d, E, NF, tmp_e_, lists_, counts_, npairs, Ly.gate_up_s, hid_, c.situ_beta, c.situ_lin);
    else throw std::runtime_error("engine: unsupported gate_up type");
    dim3 g2((E + 7) / 8, (T + NT - 1) / NT, NE);
    if (Ly.down.wt == W_NVFP4) k_moe_down<W_NVFP4, NT><<<g2, 256>>>((const uint8_t*)Ly.down.d, NF, E, hid_, lists_, counts_, npairs, Ly.down_s, dn_);
    else if (Ly.down.wt == W_Q8_0) k_moe_down<W_Q8_0, NT><<<g2, 256>>>((const uint8_t*)Ly.down.d, NF, E, hid_, lists_, counts_, npairs, Ly.down_s, dn_);
    else throw std::runtime_error("engine: unsupported down type"); }
  });
  k_moe_combine<<<T, 256>>>(dn_, wts_, moe_, 8, E);
  if (dump) { HIP_CHECK(hipDeviceSynchronize()); std::vector<int> hid(T * 8); HIP_CHECK(hipMemcpy(hid.data(), ids_, T * 8 * 4, hipMemcpyDeviceToHost)); std::vector<float> f(hid.begin(), hid.end()); HIP_CHECK(hipMemcpy(tmp_e2_, f.data(), f.size() * 4, hipMemcpyHostToDevice)); dump("ffn_moe_topk", slot, tmp_e2_, T, 8); }
  dmp("ffn_moe_out", slot, moe_, E);
  float* out = pget(); conv(moe_, Ly.ffn_post_conv, nullptr, slot, 3, tmp_e_, E);
  { const int n = T * E; k_add<<<(n + 255) / 256, 256>>>(ffn_inp, tmp_e_, out, n); }
  pdec(ffn_inp);  // the caller keeps its own reference to the layer input; ffn_inp is either A (owned) or C (owned)
  dmp("l_out", slot, out, E);
  return out;
}

inline std::vector<float> Engine::step(const Batch& b) {
  const int T = (int)b.tokens.size(), E = c.n_embd; if (T > MAX_T) throw std::runtime_error("batch too large");
  std::lock_guard<std::mutex> lk(mu);
  single_seq_ = true; d_seq0_ = b.seq.empty() ? 0 : b.seq[0]; for (int i = 0; i < T; ++i) if (b.seq[i] != d_seq0_ || (i && b.pos[i] != b.pos[i - 1] + 1)) { single_seq_ = false; break; }
  save_conv_ = b.save_conv; save_seq_ = b.seq.empty() ? 0 : b.seq[0]; save_T_ = T;
  if (save_conv_ && T > 16) throw std::runtime_error("engine: verification batches have at most 16 tokens");
  const bool capf = b.capture_features; int app = 0;
  static const int shift = std::getenv("DS_TAP_SHIFT") ? std::atoi(std::getenv("DS_TAP_SHIFT")) : 0;  // experiment: taps at 5+s, 23+s, 41+s
  auto tap = [&](float* x) { if (capf && (app == 5 + shift || app == 23 + shift || app == 41 + shift)) k_copy_cols<<<T_, 256>>>(x, d_feat_, c.n_embd, 4, app == 5 + shift ? 0 : app == 23 + shift ? 1 : 2); ++app; };
  T_ = T; pos_h_ = b.pos; maxpos_ = 0; step_cap_ = max_ctx_; for (int p : b.pos) maxpos_ = std::max(maxpos_, p);
  std::vector<int> first(MAXSEQ, 0), count(MAXSEQ, 0);
  for (int t = 0; t < T; ++t) { const int s = b.seq[t]; if (count[s] == 0) first[s] = t; count[s]++; }
  HIP_CHECK(hipMemcpy(d_tok_, b.tokens.data(), T * 4, hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(d_seq_, b.seq.data(), T * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_pos_, b.pos.data(), T * 4, hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(d_first_, first.data(), MAXSEQ * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_count_, count.data(), MAXSEQ * 4, hipMemcpyHostToDevice));
  float* cur = pget();
  k_embed_norm<<<T, 256, (E + 256) * 4>>>(d_tok_, (const uint8_t*)tok_embd_.d, row_bytes_of(W_Q8_0, E), tok_norm_, cur, E, c.eps);
  if (!b.embd_mask.empty()) {
    HIP_CHECK(hipMemcpy(d_embd_, b.embd.data(), (size_t)T * E * 4, hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(d_emask_, b.embd_mask.data(), T * 4, hipMemcpyHostToDevice));
    k_embd_override<<<T, 256>>>(cur, d_embd_, d_emask_, E);
  }
  dmp("inp_embd", -1, cur, E);
  std::vector<float*> blocks;
  auto adv = [&](float* nw) { pdec(cur); cur = nw; };
  for (int il = 0; il < c.loop_start; ++il) { tap(cur); adv(layer(il, il, cur, blocks, nullptr, nullptr)); }
  float* prelude = cur; pinc(prelude); std::vector<float*> prelude_blocks = blocks; for (float* p : prelude_blocks) pinc(p);
  const int core = c.loop_end - c.loop_start;
  for (int pass = 0; pass < c.n_passes; ++pass) {
    std::vector<float*> bank; float* bank_out = nullptr;
    if (pass > 0) {
      bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end()); for (float* p : bank) pinc(p);
      bank_out = cur; pinc(bank_out);
      for (float* p : blocks) pdec(p);          // drop the previous pass's block list
      blocks = prelude_blocks; for (float* p : blocks) pinc(p);
      pdec(cur); cur = prelude; pinc(cur);
    }
    for (int ip = c.loop_start; ip < c.loop_end; ++ip) {
      const int slot = pass == 0 ? ip : c.n_layer + (pass - 1) * core + (ip - c.loop_start);
      tap(cur);
      if (pass == 0) { adv(layer(slot, ip, cur, blocks, nullptr, nullptr)); continue; }
      std::vector<float*> msgs; mix(bank, bank_out, layers_[ip].msg_query, 4, msgs);
      float* nw = layer(slot, ip, cur, blocks, &msgs, &layers_[ip]); for (float* m : msgs) pdec(m); adv(nw);
    }
    if (pass > 0) { for (float* p : bank) pdec(p); pdec(bank_out); }
  }
  for (int il = c.loop_end; il < c.n_layer; ++il) { tap(cur); adv(layer(il, il, cur, blocks, nullptr, nullptr)); }
  std::vector<float*> om; mix(blocks, cur, out_res_, 1, om); rms(om[0], out_norm_, tmp_e_, T); pdec(om[0]);
  if (capf) k_copy_cols<<<T_, 256>>>(tmp_e_, d_feat_, E, 4, 3);
  dmp("result_norm", -1, tmp_e_, E);
  // logits for the requested rows
  const int nl = (int)b.want_logits.size(); std::vector<float> res((size_t)nl * c.n_vocab_unpadded);
  if (nl) {
    HIP_CHECK(hipMemcpy(d_rows_, b.want_logits.data(), nl * 4, hipMemcpyHostToDevice)); k_gather_rows<<<nl, 256>>>(tmp_e_, d_rows_, d_logit_in_, E);
    timed("lm_head", [&] { gemm(output_, d_logit_in_, d_logits_, nl, c.n_vocab_unpadded); });
    const int n = nl * c.n_vocab_unpadded; k_softcap<<<(n + 255) / 256, 256>>>(d_logits_, n, c.softcap);
    HIP_CHECK(hipMemcpy(res.data(), d_logits_, res.size() * 4, hipMemcpyDeviceToHost));
  }
  HIP_CHECK(hipDeviceSynchronize());
  // release everything still held by this batch
  pdec(cur); pdec(prelude); for (float* p : blocks) pdec(p); for (float* p : prelude_blocks) pdec(p);
  return res;
}

}  // namespace inkling::gpu

#include "dspark.hpp"
