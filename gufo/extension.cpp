#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include "probe.hpp"

namespace gufo::models::inkling {
#if !defined(ENGINE_ENABLE_HIP)
// A CPU build still discovers the artifact and reports it truthfully; device execution is not registered.
void RegisterInklingCpu(ext::Registry& registry) {
  registry.AddTextProvider({.id = "inkling", .probe = &ProbeInkling, .probe_gguf = &ProbeInklingGguf, .create = nullptr,
                            .bench_driver = nullptr, .prompt_driver = nullptr, .chat_driver = nullptr});
}
#endif
}  // namespace gufo::models::inkling

void inkling_register(gufo::ext::Registry& registry) {
#if !defined(ENGINE_ENABLE_HIP)
  gufo::models::inkling::RegisterInklingCpu(registry);
#else
  (void)registry;
#endif
}
