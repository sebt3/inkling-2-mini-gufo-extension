#pragma once
#include <gufo/ext/text.hpp>
#include <optional>
#include <string_view>
namespace gufo::models::inkling {
bool ProbeInkling(std::string_view model_root);
bool ProbeInklingGguf(const core::GgufReader& reader);
std::optional<sampling::TextSamplingDefaults> InklingTextSamplingDefaults(const core::GgufReader& reader);
}  // namespace gufo::models::inkling
