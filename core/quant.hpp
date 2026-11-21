// CPU dequantization and matrix-vector products for the formats of Inkling-2-Mini GGUFs.
// Layouts follow ggml (MIT): Q8_0, Q5_K, NVFP4 (see THIRD_PARTY_NOTICES.md).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "gguf.hpp"

namespace inkling {

inline float f16_to_f32(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0) v = std::ldexp((float)m, -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp((float)(m | 1024), (int)e - 25);
  return s ? -v : v;
}
inline float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

// UE4M3, value * 0.5 (kvalues are doubled E2M1)
inline float ue4m3_to_f32(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  int exp = (x >> 3) & 0xF, man = x & 7;
  float raw = exp == 0 ? std::ldexp((float)man, -9) : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
  return raw * 0.5f;
}
static const int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
  else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Dequantize one row of n elements.
inline void dequant_row(uint32_t type, const uint8_t* src, float* y, int64_t n) {
  switch (type) {
    case T_F32: memcpy(y, src, n * 4); return;
    case T_F16: for (int64_t i = 0; i < n; ++i) { uint16_t h; memcpy(&h, src + 2 * i, 2); y[i] = f16_to_f32(h); } return;
    case T_BF16: for (int64_t i = 0; i < n; ++i) { uint16_t h; memcpy(&h, src + 2 * i, 2); y[i] = bf16_to_f32(h); } return;
    case T_Q8_0:
      for (int64_t b = 0; b < n / 32; ++b) {
        const uint8_t* p = src + b * 34; uint16_t h; memcpy(&h, p, 2); float d = f16_to_f32(h);
        for (int i = 0; i < 32; ++i) y[b * 32 + i] = d * (int8_t)p[2 + i];
      }
      return;
    case T_Q5_K:
      for (int64_t b = 0; b < n / 256; ++b) {
        const uint8_t* p = src + b * 176; uint16_t h0, h1; memcpy(&h0, p, 2); memcpy(&h1, p + 2, 2);
        float d = f16_to_f32(h0), mn = f16_to_f32(h1);
        const uint8_t* scales = p + 4; const uint8_t* qh = p + 16; const uint8_t* ql = p + 48;
        float* out = y + b * 256; int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
          uint8_t sc, m;
          get_scale_min_k4(is + 0, scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
          get_scale_min_k4(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
          for (int l = 0; l < 32; ++l) *out++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
          for (int l = 0; l < 32; ++l) *out++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
          ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
      }
      return;
    case T_NVFP4:
      for (int64_t b = 0; b < n / 64; ++b) {
        const uint8_t* p = src + b * 36; const uint8_t* qs = p + 4;
        for (int s = 0; s < 4; ++s) {
          float d = ue4m3_to_f32(p[s]); float* yb = y + b * 64 + s * 16;
          for (int j = 0; j < 8; ++j) {
            uint8_t q = qs[s * 8 + j];
            yb[j] = kFp4[q & 0xF] * d; yb[j + 8] = kFp4[q >> 4] * d;
          }
        }
      }
      return;
  }
  throw std::runtime_error("dequant: unsupported type");
}

inline double dotf(const float* a, const float* b, int64_t n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0; int64_t i = 0;
  for (; i + 4 <= n; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (double)s0 + s1 + s2 + s3;
}

// y[r] = W[r, :] . x for r in [0, n_out). W rows are n_in elements, row-major in `data`.
inline void matvec(uint32_t type, const uint8_t* data, int64_t n_in, int64_t n_out, const float* x, float* y) {
  const size_t rb = (size_t)(n_in / type_block(type)) * type_bytes(type);
#pragma omp parallel
  {
    std::vector<float> buf(n_in);
#pragma omp for schedule(static)
    for (int64_t r = 0; r < n_out; ++r) {
      const uint8_t* row = data + (size_t)r * rb;
      if (type == T_F32) { y[r] = (float)dotf((const float*)row, x, n_in); continue; }
      dequant_row(type, row, buf.data(), n_in);
      y[r] = (float)dotf(buf.data(), x, n_in);
    }
  }
}
inline void matvec(const Tensor& t, const float* x, float* y) { matvec(t.type, t.data, t.ne[0], t.ne[1], x, y); }
// expert e of a 3D tensor [n_in, n_out, n_expert]
inline void matvec_expert(const Tensor& t, int e, const float* x, float* y) {
  matvec(t.type, t.data + (size_t)e * t.ne[1] * t.row_bytes(), t.ne[0], t.ne[1], x, y);
}

}  // namespace inkling
