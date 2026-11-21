// Vision encoder test: the oracle's synthetic "rainbow" image (llama-mtmd-debug -p encode), checksums per stage.
// usage: inkling_vision MMPROJ.gguf [size=640]
#include <cmath>
#include "../engine/vision.hpp"
using namespace inkling::gpu;
int main(int argc, char** argv) {
  const int S = argc > 2 ? atoi(argv[2]) : 640;
  Vision v(argv[1]);
  v.dump = [](const char* name, const float* dev, size_t n) { std::vector<float> h(n); HIP_CHECK(hipMemcpy(h.data(), dev, n * 4, hipMemcpyDeviceToHost)); double s = 0; for (float x : h) s += x; printf("%-14s n=%zu sum=%.4f first=%.5f %.5f %.5f\n", name, n, s, h[0], h[1], h[2]); };
  std::vector<float> pix((size_t)3 * S * S); const float cx = S / 2.f, cy = S / 2.f, md = std::sqrt(cx * cx + cy * cy);
  for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x) {
    float dx = x - cx, dy = y - cy, hue = std::atan2(dy, dx) / (2.f * 3.14159265f); if (hue < 0) hue += 1.f; float sat = std::sqrt(dx * dx + dy * dy) / md; if (sat > 1.f) sat = 1.f;
    float h6 = hue * 6.f; int i6 = (int)h6; float f = h6 - i6, p = 1.f - sat, q = 1.f - sat * f, t = 1.f - sat * (1.f - f), r, g, b;
    switch (i6 % 6) { case 0: r = 1; g = t; b = p; break; case 1: r = q; g = 1; b = p; break; case 2: r = p; g = 1; b = t; break; case 3: r = p; g = q; b = 1; break; case 4: r = t; g = p; b = 1; break; default: r = 1; g = p; b = q; break; }
    const size_t o = (size_t)y * S + x; pix[o] = r; pix[(size_t)S * S + o] = g; pix[(size_t)2 * S * S + o] = b;
  }
  int n; auto out = v.encode(pix, S, S, &n); printf("tokens=%d dim=%d\n", n, v.out_dim());
}
