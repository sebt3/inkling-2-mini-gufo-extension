#pragma once
#include <gufo/ext/text.hpp>
#include <memory>
#include <string>
namespace gufo::models::inkling {
// Loads the model named by options.model_path and returns a text runner; nullptr and *error on failure.
std::shared_ptr<ext::TextModelRunner> CreateInklingTextRunner(const ext::TextRunnerOptions& options, std::string* error);
}  // namespace gufo::models::inkling
