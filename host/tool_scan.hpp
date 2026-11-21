// Reader of the tool calls in generated text. No dependency on Gufo or llama.cpp (host/json.hpp only).
// Call format, as the template renders it (docs/TOOLS.md):
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{"KEY":VALUE,...}}<|end_message|>
#pragma once
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inkling {

struct ToolArg {
  std::string name;   // decoded key
  std::string value;  // is_string: the decoded content; otherwise the raw JSON text of the value
  bool is_string = false;
};

struct ToolCallOut {
  std::string name;  // from the header (the model's own name for the call)
  std::vector<ToolArg> args;
  bool last_open = false;  // incomplete call whose last argument is a string cut short (its value is the decoded prefix)
  bool complete = true;  // false: the text ends inside this call; args holds what is readable so far
};

// Scans `text` for tool calls, in order. Clears *calls and *spans when given (either may be null).
// spans[i] is the byte range [begin, end) of the markup of calls[i]: from the model header to the end marker, or to
// the closing braces when the end marker is missing, or to text.size() for an unfinished call.
//
// partial = true: the text may stop anywhere (streaming). A header whose name is not yet complete, or a marker cut
// by the end of the text, is ignored. A call whose arguments are not finished is reported last, with
// complete = false; its last argument may be a string cut short.
// partial = false: the text is final. A call whose closing braces are present is complete even without the end
// marker (the model stopped there).
//
// Malformed calls (bad JSON in the arguments, bad escapes) are skipped: they produce no element and no span.
void scan_tool_calls(std::string_view text, std::vector<ToolCallOut>* calls,
                     std::vector<std::pair<size_t, size_t>>* spans, bool partial);

}  // namespace inkling
