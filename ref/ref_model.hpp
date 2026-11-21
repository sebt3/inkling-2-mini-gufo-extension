// CPU F32 reference forward pass of Inkling-2-Mini (architecture `inkling2_loop`), one token at a time.
// A direct port of the llama.cpp graph (src/models/inkling2.cpp, inkling2-loop.cpp, build_moe_ffn, build_attn_mha).
// Slow by design: it exists to validate the GPU engine layer by layer. Weights are dequantized on the fly.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "../core/gguf.hpp"
#include "../core/quant.hpp"

namespace inkling::ref {

using Vec = std::vector<float>;

struct Hparams {
  int n_embd = 2304, n_layer = 24, n_head = 24, n_head_kv = 8, d_head = 128;
  int n_expert = 128, n_expert_used = 8, n_ff_exp = 1152;
  int loop_start = 6, loop_end = 18, n_passes = 3, gate_rank = 64;
  int n_vocab = 201024, n_vocab_unpadded = 200058;
  int n_swa = 1024, n_conv = 4, block = 6, n_rel = 16, rel_extent = 1024, log_floor = 8192;
  float eps = 1e-6f, qk_clip = 22.627417f, rel_clip = 8.0f, log_alpha = 0.1f;
  float situ_beta = 4.0f, situ_lin = 25.0f, w_scale = 8.0f, softcap = 30.0f;
  std::vector<bool> is_swa;  // per physical layer
  int n_kv_dim() const { return n_head_kv * d_head; }
};

struct Layer {
  const Tensor *wq = 0, *wk = 0, *wv = 0, *wo = 0, *gate = 0, *q_clip = 0, *sink = 0, *rel = 0, *rel_bias = 0, *rel_clip = 0;
  const Tensor *k_conv = 0, *v_conv = 0, *post_conv = 0, *ffn_post_conv = 0;
  const Tensor *attn_norm = 0, *q_inp = 0, *k_inp = 0, *v_inp = 0, *kc_proj = 0, *vc_proj = 0;
  const Tensor *attn_res = 0, *ffn_res = 0, *ffn_norm = 0, *gate_inp = 0, *probs_b = 0, *gate_up = 0, *gate_up_s = 0, *down = 0, *down_s = 0;
  // loop controllers (core layers only)
  const Tensor *msg_query = 0, *c_down[4] = {}, *c_gain[4] = {}, *c_forget[4] = {};
};

// recurrent convolution history: the last n_conv-1 inputs of each of the four convolutions of a cache slot,
// oldest first ([lag][channel])
struct ConvState { Vec k, v, a, f; };
struct SlotCache {
  std::vector<Vec> K, V;  // per position: n_head_kv * d_head (after convolution and clipping)
  ConvState conv;
};

class Model {
 public:
  explicit Model(const std::string& path) : g_(path) {
    load_hparams(); load_tensors();
    n_slot_ = hp.n_layer + (hp.n_passes - 1) * (hp.loop_end - hp.loop_start);
    reset();
  }
  void reset() {
    slots_.assign(n_slot_, {});
    const int c = hp.n_conv - 1;
    for (auto& s : slots_) {
      s.conv.k.assign(c * hp.n_kv_dim(), 0.f); s.conv.v.assign(c * hp.n_kv_dim(), 0.f);
      s.conv.a.assign(c * hp.n_embd, 0.f); s.conv.f.assign(c * hp.n_embd, 0.f);
    }
    pos_ = 0;
  }
  // Feed one token, return the cropped, softcapped logits.
  Vec forward(int token);
  int position() const { return pos_; }
  Hparams hp;
  // optional tracing: (name, slot, data) for layer-by-layer comparison
  std::function<void(const char*, int, const Vec&)> trace;

 private:
  Gguf g_;
  std::vector<Layer> layers_;
  const Tensor *tok_embd_ = 0, *tok_norm_ = 0, *out_norm_ = 0, *out_res_ = 0, *output_ = 0;
  int n_slot_ = 0, pos_ = 0;
  std::vector<SlotCache> slots_;

  void load_hparams() {
    auto& h = hp; const std::string a = "inkling2_loop.";
    h.n_embd = (int)g_.num(a + "embedding_length"); h.n_layer = (int)g_.num(a + "block_count");
    h.n_head = (int)g_.m(a + "attention.head_count").arr[0]; h.n_head_kv = (int)g_.m(a + "attention.head_count_kv").arr[0];
    h.d_head = (int)g_.num(a + "attention.key_length");
    h.n_expert = (int)g_.num(a + "expert_count"); h.n_expert_used = (int)g_.num(a + "expert_used_count");
    h.n_ff_exp = (int)g_.num(a + "expert_feed_forward_length");
    h.loop_start = (int)g_.num(a + "loop.start"); h.loop_end = (int)g_.num(a + "loop.end");
    h.n_passes = (int)g_.num(a + "loop.count"); h.gate_rank = (int)g_.num(a + "loop.gate_rank");
    h.n_vocab = (int)g_.num(a + "vocab_size"); h.n_vocab_unpadded = (int)g_.num(a + "vocab_size_unpadded");
    h.n_swa = (int)g_.num(a + "attention.sliding_window"); h.n_conv = (int)g_.num(a + "shortconv.l_cache");
    h.block = (int)g_.num(a + "attn_res.block_size"); h.n_rel = (int)g_.num(a + "attention.relative_length");
    h.rel_extent = (int)g_.num(a + "attention.relative_extent"); h.log_floor = (int)g_.num(a + "attention.log_scale_floor");
    h.eps = (float)g_.num(a + "attention.layer_norm_rms_epsilon"); h.qk_clip = (float)g_.num(a + "attention.qk_clip");
    h.rel_clip = (float)g_.num(a + "attention.relative_clip"); h.log_alpha = (float)g_.num(a + "attention.log_scale_alpha");
    h.situ_beta = (float)g_.num(a + "activation.situ_beta"); h.situ_lin = (float)g_.num(a + "activation.situ_linear_beta");
    h.w_scale = (float)g_.num(a + "expert_weights_scale"); h.softcap = (float)g_.num(a + "final_logit_softcapping");
    for (double v : g_.m(a + "attention.sliding_window_pattern").arr) h.is_swa.push_back(v != 0);
  }

  void load_tensors() {
    tok_embd_ = &g_.get("token_embd.weight"); tok_norm_ = &g_.get("token_embd_norm.weight");
    out_norm_ = &g_.get("output_norm.weight"); out_res_ = &g_.get("output_res_query.weight"); output_ = &g_.get("output.weight");
    layers_.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; ++i) {
      auto& L = layers_[i]; auto n = [&](const char* s) { return "blk." + std::to_string(i) + "." + s; };
      auto opt = [&](const char* s) { return g_.find(n(s)); };
      L.wq = &g_.get(n("attn_q.weight")); L.wk = &g_.get(n("attn_k.weight")); L.wv = &g_.get(n("attn_v.weight")); L.wo = &g_.get(n("attn_output.weight"));
      L.gate = &g_.get(n("attn_gate.weight")); L.q_clip = &g_.get(n("attn_q_clip.weight")); L.sink = &g_.get(n("attn_sink_proj.weight"));
      L.rel = &g_.get(n("attn_rel.weight")); L.rel_bias = &g_.get(n("attn_rel_bias.weight")); L.rel_clip = &g_.get(n("attn_rel_clip.weight"));
      L.k_conv = &g_.get(n("attn_k_conv.weight")); L.v_conv = &g_.get(n("attn_v_conv.weight"));
      L.post_conv = &g_.get(n("attn_post_conv.weight")); L.ffn_post_conv = &g_.get(n("ffn_post_conv.weight"));
      L.attn_norm = opt("attn_norm.weight"); L.q_inp = opt("attn_q_inp_norm.weight"); L.k_inp = opt("attn_k_inp_norm.weight");
      L.v_inp = opt("attn_v_inp_norm.weight"); L.kc_proj = opt("attn_k_conv_proj.weight"); L.vc_proj = opt("attn_v_conv_proj.weight");
      L.attn_res = opt("attn_res_query.weight"); L.ffn_res = &g_.get(n("ffn_res_query.weight")); L.ffn_norm = &g_.get(n("ffn_norm.weight"));
      L.gate_inp = &g_.get(n("ffn_gate_inp.weight")); L.probs_b = opt("exp_probs_b.bias");
      L.gate_up = &g_.get(n("ffn_gate_up_exps.weight")); L.gate_up_s = opt("ffn_gate_up_exps.scale");
      L.down = &g_.get(n("ffn_down_exps.weight")); L.down_s = opt("ffn_down_exps.scale");
      L.msg_query = opt("loop_msg_query.weight");
      static const char* cn[4] = {"q", "k", "v", "ffn"};
      for (int c = 0; c < 4; ++c) {
        L.c_down[c] = opt((std::string("loop_") + cn[c] + "_ctrl_down.weight").c_str());
        L.c_gain[c] = opt((std::string("loop_") + cn[c] + "_ctrl_gain.weight").c_str());
        L.c_forget[c] = opt((std::string("loop_") + cn[c] + "_ctrl_forget.weight").c_str());
      }
    }
  }

  static const float* f32(const Tensor* t) { return (const float*)t->data; }  // F32 tensors only
  static float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

  // rms_norm over the vector, then optional per-channel weight
  Vec rms(const Vec& x, const Tensor* w = nullptr) const {
    double s = 0; for (float v : x) s += (double)v * v;
    float r = 1.f / std::sqrt((float)(s / x.size()) + hp.eps);
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * r * (w ? f32(w)[i] : 1.f);
    return y;
  }

  // attention over the depth: mix earlier block outputs and cur with learned queries; outs[q] is the q-th result
  void mix(const std::vector<Vec>& blocks, const Vec& cur, const Tensor* query, int nq, std::vector<Vec>& outs) const {
    std::vector<Vec> st;
    for (auto& b : blocks) st.push_back(rms(b));
    st.push_back(rms(cur));
    const int E = hp.n_embd, nb = (int)st.size();
    outs.assign(nq, Vec(E, 0.f));
    for (int q = 0; q < nq; ++q) {
      const float* qv = f32(query) + (size_t)q * E;
      std::vector<float> sc(nb); float mx = -1e30f;
      for (int b = 0; b < nb; ++b) { sc[b] = (float)dotf(st[b].data(), qv, E) / std::sqrt((float)E); mx = std::max(mx, sc[b]); }
      float sum = 0; for (auto& s : sc) { s = std::exp(s - mx); sum += s; }
      for (int b = 0; b < nb; ++b) { float w = sc[b] / sum; for (int e = 0; e < E; ++e) outs[q][e] += w * st[b][e]; }
    }
  }

  // mix a cross-pass message into a looped layer input
  Vec control(const Vec& cur, const Vec& msg, const Layer& L, int c) const {
    const int E = hp.n_embd; Vec gate(2 * E);
    Vec a = rms(cur), b = rms(msg);
    std::copy(a.begin(), a.end(), gate.begin()); std::copy(b.begin(), b.end(), gate.begin() + E);
    Vec low(hp.gate_rank); matvec(*L.c_down[c], gate.data(), low.data());
    Vec gain(E), forget(E); matvec(*L.c_gain[c], low.data(), gain.data()); matvec(*L.c_forget[c], low.data(), forget.data());
    Vec out(E);
    for (int e = 0; e < E; ++e) out[e] = cur[e] - std::tanh(forget[e]) * cur[e] + std::tanh(gain[e]) * msg[e];
    return out;
  }

  // causal short convolution with residual. hist: (n_conv-1) x channels, oldest first; updated with cur.
  // dyn: optional input-dependent taps, dyn[lag + head*n_conv], lag 0 = current input.
  Vec convolve(const Vec& cur, const Tensor* w, const Vec* dyn, Vec& hist, int ch) const {
    const int nc = hp.n_conv; const float* wt = f32(w);  // wt[k + c*nc]
    Vec out(ch, 0.f);
    auto x = [&](int k, int c) -> float { return k < nc - 1 ? hist[(size_t)k * ch + c] : cur[c]; };  // k = 0..nc-1 oldest..current
    for (int c = 0; c < ch; ++c) { float s = 0; for (int k = 0; k < nc; ++k) s += x(k, c) * wt[k + (size_t)c * nc]; out[c] = s; }
    if (dyn) {
      const int nh = (int)dyn->size() / nc, per = ch / nh;
      for (int lag = 0; lag < nc; ++lag)
        for (int c = 0; c < ch; ++c) out[c] += x(nc - 1 - lag, c) * (*dyn)[lag + (size_t)(c / per) * nc];
    }
    const float sc = 1.f / std::sqrt((float)((dyn ? 2 : 1) * nc + 1));
    Vec res(ch); for (int c = 0; c < ch; ++c) res[c] = (cur[c] + out[c]) * sc;
    // shift the history: drop the oldest, append cur
    for (int k = 0; k + 1 < nc - 1; ++k) std::copy(hist.begin() + (size_t)(k + 1) * ch, hist.begin() + (size_t)(k + 2) * ch, hist.begin() + (size_t)k * ch);
    std::copy(cur.begin(), cur.end(), hist.begin() + (size_t)(nc - 2) * ch);
    return res;
  }

  // scale each row of `cnt` rows of `d` values down to an L2 norm of at most thr, after the optional per-channel scale
  void clip(float* x, int cnt, int d, const float* scale, float thr) const {
    for (int r = 0; r < cnt; ++r) {
      float* p = x + (size_t)r * d;
      if (scale) for (int i = 0; i < d; ++i) p[i] *= scale[(size_t)r * d + i];
      double s = 0; for (int i = 0; i < d; ++i) s += (double)p[i] * p[i];
      float nrm = (float)std::sqrt(s) + 1e-6f;
      float f = std::min(1.f, std::max(0.f, thr / nrm));
      for (int i = 0; i < d; ++i) p[i] *= f;
    }
  }

  Vec layer(int slot, int ip, Vec cur, std::vector<Vec>& blocks, const std::vector<Vec>* msgs, const Layer* loop) {
    const auto& L = layers_[ip];
    const int E = hp.n_embd, NH = hp.n_head, NKV = hp.n_head_kv, D = hp.d_head, KVD = hp.n_kv_dim();
    const bool swa = hp.is_swa[ip];
    auto& S = slots_[slot];
    Vec inpSA = cur;
    Vec in[3] = {cur, cur, cur};
    if (L.attn_res) {
      std::vector<Vec> mixed; mix(blocks, cur, L.attn_res, swa ? 1 : 3, mixed);
      for (int j = 0; j < 3; ++j) in[j] = mixed[swa ? 0 : j];
    }
    if (msgs) for (int j = 0; j < 3; ++j) in[j] = control(in[j], (*msgs)[j], *loop, j);
    if (swa) { for (int j = 0; j < 3; ++j) in[j] = rms(in[j], L.attn_norm); }
    else { in[0] = rms(in[0], L.q_inp); in[1] = rms(in[1], L.k_inp); in[2] = rms(in[2], L.v_inp); }

    Vec q(NH * D), k0(KVD), v0(KVD), r(NH * hp.n_rel), sinks(NH);
    matvec(*L.wq, in[0].data(), q.data()); matvec(*L.wk, in[1].data(), k0.data()); matvec(*L.wv, in[2].data(), v0.data());
    matvec(*L.rel, in[0].data(), r.data()); matvec(*L.sink, in[0].data(), sinks.data());
    Vec kd, vd; if (!swa) { kd.resize(NKV * hp.n_conv); vd.resize(NKV * hp.n_conv); matvec(*L.kc_proj, in[1].data(), kd.data()); matvec(*L.vc_proj, in[2].data(), vd.data()); }
    Vec K = convolve(k0, L.k_conv, swa ? nullptr : &kd, S.conv.k, KVD);
    Vec V = convolve(v0, L.v_conv, swa ? nullptr : &vd, S.conv.v, KVD);
    clip(q.data(), NH, D, f32(L.q_clip), hp.qk_clip);
    clip(K.data(), NKV, D, nullptr, hp.qk_clip);
    for (int h = 0; h < NKV; ++h) { Vec t(V.begin() + h * D, V.begin() + (h + 1) * D); t = rms(t); std::copy(t.begin(), t.end(), V.begin() + h * D); }
    clip(r.data(), NH, hp.n_rel, f32(L.rel_clip), hp.rel_clip);
    if (!swa) {
      float tau = std::log(std::min(1e30f, std::max(1.f, (float)(pos_ + 1) / hp.log_floor)));
      tau = hp.log_alpha * tau + 1.f;
      for (auto& v : q) v *= tau; for (auto& v : r) v *= tau;
    }
    if ((int)S.K.size() != pos_) throw std::runtime_error("cache/position mismatch");
    S.K.push_back(K); S.V.push_back(V);
    if (trace) { trace("Qcur", slot, q); trace("Kcur", slot, K); trace("Vcur", slot, V); }

    // attention, one query token
    Vec att(NH * D, 0.f); const float scale = 1.f / std::sqrt((float)D);
    const int j0 = swa ? std::max(0, pos_ - hp.n_swa + 1) : 0;
    const float* rb = f32(L.rel_bias);  // [n_rel, extent]: rb[i + d*n_rel]
#pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; ++h) {
      const int kv = h / (NH / NKV);
      std::vector<float> lg(pos_ + 1 - j0); float mx = 0.f;  // the sink logit is 0
      for (int j = j0; j <= pos_; ++j) {
        const float* kj = S.K[j].data() + kv * D;
        double dot = 0; for (int d = 0; d < D; ++d) dot += (double)q[h * D + d] * kj[d];
        const int dist = pos_ - j; float bias = 0.f;
        if (dist >= 0 && dist < hp.rel_extent) { const float* row = rb + (size_t)dist * hp.n_rel; for (int i = 0; i < hp.n_rel; ++i) bias += row[i] * r[h * hp.n_rel + i]; }
        lg[j - j0] = ((float)dot + bias - sinks[h]) * scale; mx = std::max(mx, lg[j - j0]);
      }
      double sum = std::exp(0.f - mx); for (auto& x : lg) { x = std::exp(x - mx); sum += x; }
      for (int j = j0; j <= pos_; ++j) { float w = (float)(lg[j - j0] / sum); const float* vj = S.V[j].data() + kv * D; for (int d = 0; d < D; ++d) att[h * D + d] += w * vj[d]; }
    }
    if (trace) trace("attn_core", slot, att);
    Vec g(NH * D); matvec(*L.gate, in[0].data(), g.data());
    for (int i = 0; i < NH * D; ++i) att[i] *= sigm(g[i]);
    Vec o(E); matvec(*L.wo, att.data(), o.data());
    cur = convolve(o, L.post_conv, nullptr, S.conv.a, E);
    if (trace) trace("attn_out", slot, cur);

    // the first layer of each block starts a new residual stream
    if (ip % hp.block == 0) blocks.push_back(inpSA);
    else for (int e = 0; e < E; ++e) cur[e] += inpSA[e];
    Vec ffn_inp = cur;

    std::vector<Vec> fm; mix(blocks, cur, L.ffn_res, 1, fm);
    Vec x = fm[0];
    if (msgs) x = control(x, (*msgs)[3], *loop, 3);
    x = rms(x, L.ffn_norm);

    // MoE
    const int NE = hp.n_expert, NU = hp.n_expert_used, NF = hp.n_ff_exp;
    Vec logits(NE); matvec(*L.gate_inp, x.data(), logits.data());
    Vec probs(NE), selp(NE); for (int e = 0; e < NE; ++e) { probs[e] = sigm(logits[e]); selp[e] = probs[e] + (L.probs_b ? f32(L.probs_b)[e] : 0.f); }
    std::vector<int> idx(NE); std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return selp[a] > selp[b]; });
    float wsum = 0; for (int i = 0; i < NU; ++i) wsum += probs[idx[i]];
    wsum += 1e-20f;
    Vec moe(E, 0.f), gu(2 * NF), hid(NF), dn(E);
    for (int i = 0; i < NU; ++i) {
      const int e = idx[i]; const float w = probs[e] / wsum * hp.w_scale;
      matvec_expert(*L.gate_up, e, x.data(), gu.data());
      if (L.gate_up_s) { const float s = f32(L.gate_up_s)[e]; for (auto& v : gu) v *= s; }
      for (int f = 0; f < NF; ++f) {
        float ge = gu[f], up = gu[NF + f];
        float act = hp.situ_beta * std::tanh(ge / hp.situ_beta) * sigm(ge);
        hid[f] = act * (hp.situ_lin * std::tanh(up / hp.situ_lin));
      }
      matvec_expert(*L.down, e, hid.data(), dn.data());
      if (L.down_s) { const float s = f32(L.down_s)[e]; for (auto& v : dn) v *= s; }
      for (int c = 0; c < E; ++c) moe[c] += w * dn[c];
    }
    if (trace) trace("moe", slot, moe);
    Vec fo = convolve(moe, L.ffn_post_conv, nullptr, S.conv.f, E);
    for (int e = 0; e < E; ++e) fo[e] += ffn_inp[e];
    if (trace) trace("l_out", slot, fo);
    return fo;
  }

 public:
  Vec forward_impl(int token) {
    const int E = hp.n_embd;
    Vec cur(E); dequant_row(tok_embd_->type, tok_embd_->data + (size_t)token * tok_embd_->row_bytes(), cur.data(), E);
    cur = rms(cur, tok_norm_);
    if (trace) trace("inp_embd", -1, cur);
    std::vector<Vec> blocks;
    for (int il = 0; il < hp.loop_start; ++il) cur = layer(il, il, cur, blocks, nullptr, nullptr);
    Vec prelude = cur; const std::vector<Vec> prelude_blocks = blocks;
    const int core = hp.loop_end - hp.loop_start;
    for (int pass = 0; pass < hp.n_passes; ++pass) {
      std::vector<Vec> bank; Vec bank_out;
      if (pass > 0) {
        bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
        bank_out = cur; cur = prelude; blocks = prelude_blocks;
      }
      for (int ip = hp.loop_start; ip < hp.loop_end; ++ip) {
        const int slot = pass == 0 ? ip : hp.n_layer + (pass - 1) * core + (ip - hp.loop_start);
        if (pass == 0) { cur = layer(slot, ip, cur, blocks, nullptr, nullptr); continue; }
        std::vector<Vec> msgs; mix(bank, bank_out, layers_[ip].msg_query, 4, msgs);
        cur = layer(slot, ip, cur, blocks, &msgs, &layers_[ip]);
      }
    }
    for (int il = hp.loop_end; il < hp.n_layer; ++il) cur = layer(il, il, cur, blocks, nullptr, nullptr);
    std::vector<Vec> om; mix(blocks, cur, out_res_, 1, om);
    Vec h = rms(om[0], out_norm_);
    if (trace) trace("result_norm", -1, h);
    Vec logits(hp.n_vocab_unpadded);
    matvec(output_->type, output_->data, output_->ne[0], hp.n_vocab_unpadded, h.data(), logits.data());
    for (auto& v : logits) v = hp.softcap * std::tanh(v / hp.softcap);
    ++pos_;
    return logits;
  }
};

inline Vec Model::forward(int token) { return forward_impl(token); }

}  // namespace inkling::ref
