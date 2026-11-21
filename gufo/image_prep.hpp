// Image preprocessing of the Inkling vision tower (llama.cpp mtmd_image_preprocessor_inkling2v): both sides rounded up to a
// multiple of 40 (patch 20 x merge 2) with padding, pixels normalised with mean = std = 0.5, planar [3][H][W]. Images with more
// than 4096 merged tokens are first resized (bilinear) keeping their ratio.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gufo::models::inkling {

struct PreparedImage { std::vector<float> planar; int width{0}, height{0}; int n_tokens() const { return (width / 40) * (height / 40); } };

inline PreparedImage PrepareImage(const std::uint8_t* rgb, int w, int h) {
  constexpr int align = 40; constexpr long max_pixels = 4096L * align * align;
  const auto up = [&](int v) { return (v + align - 1) / align * align; };
  std::vector<std::uint8_t> resized;
  if ((long)up(w) * up(h) > max_pixels) {
    const double s = std::sqrt((double)max_pixels / ((double)w * h));
    const int nw = std::max(align, (int)(w * s) / align * align), nh = std::max(align, (int)(h * s) / align * align);
    resized.resize((size_t)nw * nh * 3);
    for (int y = 0; y < nh; ++y) {
      const float fy = std::max(0.f, ((float)y + 0.5f) * h / nh - 0.5f); const int y0 = std::min((int)fy, h - 1), y1 = std::min(y0 + 1, h - 1); const float dy = fy - y0;
      for (int x = 0; x < nw; ++x) {
        const float fx = std::max(0.f, ((float)x + 0.5f) * w / nw - 0.5f); const int x0 = std::min((int)fx, w - 1), x1 = std::min(x0 + 1, w - 1); const float dx = fx - x0;
        for (int c = 0; c < 3; ++c) {
          const float a = rgb[((size_t)y0 * w + x0) * 3 + c], b = rgb[((size_t)y0 * w + x1) * 3 + c], cc = rgb[((size_t)y1 * w + x0) * 3 + c], d = rgb[((size_t)y1 * w + x1) * 3 + c];
          resized[((size_t)y * nw + x) * 3 + c] = (std::uint8_t)std::lround(a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + cc * (1 - dx) * dy + d * dx * dy);
        }
      }
    }
    rgb = resized.data(); w = nw; h = nh;
  }
  PreparedImage out; out.width = up(w); out.height = up(h);
  out.planar.assign((size_t)3 * out.width * out.height, 0.f);
  const size_t plane = (size_t)out.width * out.height; const float pad = (-1.0f - 127.5f) / 127.5f;
  for (int y = 0; y < out.height; ++y) for (int x = 0; x < out.width; ++x) for (int c = 0; c < 3; ++c)
    out.planar[c * plane + (size_t)y * out.width + x] = (y < h && x < w) ? ((float)rgb[((size_t)y * w + x) * 3 + c] - 127.5f) / 127.5f : pad;
  return out;
}

}  // namespace gufo::models::inkling
