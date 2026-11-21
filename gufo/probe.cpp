#include "probe.hpp"
#include <filesystem>
#include <string>
namespace gufo::models::inkling {

bool ProbeInklingGguf(const core::GgufReader& reader) {
  const auto a = reader.GetMetadataString("general.architecture");
  return a.has_value() && *a == "inkling2_loop";
}
namespace {
std::optional<std::filesystem::path> ResolveArtifact(std::string_view model_root) {
  std::error_code ec; const std::filesystem::path root(model_root);
  if (std::filesystem::is_regular_file(root, ec)) return root;
  if (!std::filesystem::is_directory(root, ec)) return std::nullopt;
  std::optional<std::filesystem::path> found;
  for (const auto& e : std::filesystem::directory_iterator(root, ec)) {
    if (!e.is_regular_file() || e.path().extension() != ".gguf") continue;
    // mmproj and drafter files live beside the model: only a file that is itself an inkling2_loop model counts
    std::string error; auto reader = core::GgufReader::OpenFile(e.path().string(), &error);
    if (!reader || !ProbeInklingGguf(*reader)) continue;
    if (found.has_value()) return std::nullopt;
    found = e.path();
  }
  return found;
}
}  // namespace

bool ProbeInkling(std::string_view model_root) {
  const auto artifact = ResolveArtifact(model_root); if (!artifact) return false;
  std::string error; auto reader = core::GgufReader::OpenFile(artifact->string(), &error);
  return reader != nullptr && ProbeInklingGguf(*reader);
}

// Recommended sampling of the model card, also stored in the GGUF (general.sampling.*): temperature 1, no truncation.
std::optional<sampling::TextSamplingDefaults> InklingTextSamplingDefaults(const core::GgufReader&) {
  sampling::TextSamplingValues v; v.temperature = 1.0F; v.top_k = 0; v.top_p = 1.0F; v.min_p = 0.0F;
  return sampling::TextSamplingDefaults{v, v, v};
}
}  // namespace gufo::models::inkling
