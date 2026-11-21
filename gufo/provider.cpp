#include <gufo/ext/generated/inkling_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include "probe.hpp"
#include "runner.hpp"

// Provider entry point, run by the host's generated registry only when the provider target is enabled.
void inkling_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_inkling::kProvider_inkling);
  registry.AddTextProvider({
      .id = "inkling",
      .probe = &gufo::models::inkling::ProbeInkling,
      .probe_gguf = &gufo::models::inkling::ProbeInklingGguf,
      .create = &gufo::models::inkling::CreateInklingTextRunner,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
      .sampling_defaults = &gufo::models::inkling::InklingTextSamplingDefaults,
      // NVFP4 is an extension-owned quantization: the host reader cannot size it
      .accepts_unsized_tensors = true,
  });
}
