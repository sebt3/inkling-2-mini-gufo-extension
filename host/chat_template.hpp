// Prompt rendering for Inkling-2 (the Jinja chat template stored in the GGUF, tokenizer.chat_template).
// The template is reproduced branch by branch; see docs/CHAT-TEMPLATE.md. No dependency on llama.cpp or Gufo.
#pragma once
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace inkling {

struct ContentPart {
  std::string type;  // only "text" is accepted by this renderer
  std::string text;
};

struct Message {
  enum class Content { Absent, Null, Text, Parts };

  std::string role;  // system | user | assistant | tool
  Content content = Content::Absent;
  std::string text;                    // when content == Text
  std::vector<ContentPart> parts;      // when content == Parts
  std::string reasoning_content;       // assistant thinking, empty when none
  std::string name;                    // optional; tool name for tool messages, name suffix for assistant

  static Message make(std::string role, std::string text) {
    Message m;
    m.role = std::move(role);
    m.content = Content::Text;
    m.text = std::move(text);
    return m;
  }
};

struct Options {
  // The oracle server (llama-server default) turns a trailing assistant message into a prefill: the prompt
  // ends with that message's text, without its end-of-message marker. Set false to render it as a closed turn.
  bool prefill_assistant = true;

  // Reasoning effort. At most one of the two is set; neither means the template default (0.9).
  // reasoning_effort is a level name of the template's effort_map (minimal, low, medium, high, xhigh, max,
  // and also none, which the template maps to 0). reasoning_effort_value is a number in [0.0, 0.99].
  std::optional<std::string> reasoning_effort;
  std::optional<double> reasoning_effort_value;
};

// Renders the full prompt. Throws std::invalid_argument where the template calls raise_exception or where
// the server rejects the conversation (same messages as llama.cpp).
std::string render_prompt(const std::vector<Message>& messages, const Options& options);

}  // namespace inkling
