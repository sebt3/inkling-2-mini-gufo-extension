#include <gufo/ext/generated/inkling_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include "probe.hpp"
#include "runner.hpp"
#include "../host/tool_scan.hpp"

namespace {
// Tool calls: <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>. The grammar descriptor
// only lets the host register the dialect: constrained calls are refused by PreparePrompt (the name is repeated inside the JSON,
// which a ToolSyntax cannot express).
void ScanInklingToolCalls(std::string_view text, std::span<const std::string_view>, std::vector<gufo::ext::ScannedToolCall>* calls,
                          std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial) {
  std::vector<inkling::ToolCallOut> found;
  inkling::scan_tool_calls(text, &found, spans, partial);
  for (auto& f : found) {
    gufo::ext::ScannedToolCall c; c.name = std::move(f.name); c.valid = true; c.complete = f.complete;
    for (auto& a : f.args) c.arguments.push_back({std::move(a.name), std::move(a.value), a.is_string});
    if (f.last_open && !c.arguments.empty()) { c.unfinished = std::move(c.arguments.back()); c.arguments.pop_back(); }
    calls->push_back(std::move(c));
  }
}
}  // namespace

// Provider entry point, run by the host's generated registry only when the provider target is enabled.
void inkling_provider_register(gufo::ext::Registry& registry) {
  {
    gufo::ext::ToolDialect d;
    d.id = "inkling";
    // Every assistant block, answers included, starts with <|message_model|>, so a grammar opened by it would turn each answer into a
    // call. The grammar therefore gets an opener the model never writes (generation stays free), and the scanner is started by the
    // real header, which the runner only lets through for tool calls (see InklingTextRunner::Piece).
    constexpr std::string_view kGrammarOpen = "<|inkling-grammar-never-emitted|>";
    d.markers = {"<|message_model|>", kGrammarOpen};
    d.closers = {};
    d.scan_tool_calls = &ScanInklingToolCalls;
    gufo::sampling::JsonConstraint::ToolSyntax syn;
    syn.open = std::string(kGrammarOpen); syn.name_suffix = "<|content_invoke_tool_json|>"; syn.close = "<|end_message|>"; syn.name_delimiters = "<";
    syn.values.whitespace = false; syn.values.sorted_keys = true;
    d.syntax = syn;
    gufo::ext::RegisterToolDialect(std::move(d));
  }
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
      .speculative_defaults = {{.backend = "dspark", .max_draft_tokens = 7, .min_draft_tokens = 1, .draft_policy = "", .draft_policies = {}}},
  });
}
