#include "chat_template.hpp"

#include <cstdio>

namespace inkling {
namespace {

const char* const kEnd = "<|end_message|>";
const char* const kText = "<|content_text|>";
const char* const kThink = "<|content_thinking|>";
const char* const kSampling = "<|content_model_end_sampling|>";
const char* const kModel = "<|message_model|>";
const char* const kSystemEffort = "<|message_system|><|content_text|>Thinking effort level: ";

// effort_map of the template
struct EffortLevel {
  const char* name;
  double value;
};
const EffortLevel kEffortMap[] = {
    {"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7},
    {"high", 0.9}, {"xhigh", 0.99}, {"max", 0.99},
};

const char* role_token(const std::string& role) {
  if (role == "user") return "<|message_user|>";
  if (role == "assistant") return "<|message_model|>";
  if (role == "system") return "<|message_system|>";
  if (role == "tool") return "<|message_tool|>";
  return nullptr;
}

// Python/minja style: "%f" then trailing zeros removed, at least one decimal digit ("0.5", "0.99", "0.0")
std::string format_number(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.6f", v);
  std::string s = buf;
  while (s.size() > 1 && s.back() == '0' && s[s.size() - 2] != '.') s.pop_back();
  return s;
}

std::string trim(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

// the effort_macro of the template
std::string render_effort(const Options& opt) {
  if (opt.reasoning_effort && opt.reasoning_effort_value) {
    throw std::invalid_argument("set either reasoning_effort or reasoning_effort_value, not both");
  }
  double num = 0.9;
  if (opt.reasoning_effort) {
    const std::string key = trim(*opt.reasoning_effort);
    bool found = false;
    for (const auto& lvl : kEffortMap) {
      if (key == lvl.name) {
        num = lvl.value;
        found = true;
        break;
      }
    }
    if (!found) throw std::invalid_argument("Unknown reasoning_effort: " + *opt.reasoning_effort);
  } else if (opt.reasoning_effort_value) {
    num = *opt.reasoning_effort_value;
    if (num < 0.0 || num > 0.99) throw std::invalid_argument("reasoning_effort must be in [0.0, 0.99]");
  }
  std::string out = kSystemEffort;
  out += (num == 0.0) ? std::string("0") : format_number(num);
  out += kEnd;
  return out;
}

// the main loop of the template over the messages (no tools)
std::string render_messages(const std::vector<Message>& messages, const Options& opt) {
  std::string out;
  bool effort_emitted = false;

  for (const auto& m : messages) {
    const char* role_tok = role_token(m.role);
    if (!role_tok) throw std::invalid_argument("Unknown message role: " + m.role);
    if (!effort_emitted && m.role != "system") {
      out += render_effort(opt);
      effort_emitted = true;
    }
    const bool assistant = m.role == "assistant";

    // header: role token, followed by the name for tool messages and named assistant messages
    std::string header = role_tok;
    if (!m.name.empty() && (m.role == "tool" || assistant)) header += m.name;

    std::string rendered;
    if (assistant && !m.reasoning_content.empty()) {
      rendered += header + kThink + m.reasoning_content + kEnd;
    }

    const bool is_text = m.content == Message::Content::Text;
    if (is_text && (!m.text.empty() || !(assistant && !m.reasoning_content.empty()))) {
      rendered += header + kText + m.text + kEnd;
    } else if (m.content == Message::Content::Parts && !m.parts.empty()) {
      for (const auto& p : m.parts) {
        if (p.type != "text") throw std::invalid_argument("Unsupported content part type: " + p.type);
        rendered += header + kText + p.text + kEnd;
      }
    }

    if (assistant) rendered += kSampling;
    out += rendered;
  }

  if (!effort_emitted) out += render_effort(opt);
  return out;
}

// Text that continues a trailing assistant message (llama.cpp common_chat_params_init_inkling).
// Typed parts are joined with a blank line (separator added before each part once some text exists), for the
// continuation only: the template itself renders one block per part.
std::string continuation_text(const Message& m) {
  std::string text;
  if (m.content == Message::Content::Text) {
    text = m.text;
  } else if (m.content == Message::Content::Parts) {
    for (const auto& p : m.parts) {
      if (!text.empty()) text += "\n\n";
      text += p.text;
    }
  }
  const bool has_text = !text.empty();
  const bool has_reasoning = !m.reasoning_content.empty();
  if (!has_reasoning && !has_text) return "";

  std::string out;
  if (has_reasoning) out = std::string(kModel) + kThink + m.reasoning_content;
  if (has_text) {
    if (!out.empty()) out += kEnd;
    out += std::string(kModel) + kText + text;
  }
  return out;
}

// The server turns null, absent and empty-array contents into an empty string before the template runs.
Message normalized(Message m) {
  if (m.content == Message::Content::Null || m.content == Message::Content::Absent ||
      (m.content == Message::Content::Parts && m.parts.empty())) {
    m.content = Message::Content::Text;
    m.text.clear();
  }
  return m;
}

}  // namespace

std::string render_prompt(const std::vector<Message>& messages, const Options& options) {
  // server side (llama.cpp tools/server, oaicompat parsing): every message needs a content key, and an
  // assistant message needs content or tool calls
  for (const auto& m : messages) {
    if (m.content != Message::Content::Absent) continue;
    if (m.role == "assistant") {
      throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
    }
    throw std::invalid_argument("All non-assistant messages must contain 'content'");
  }

  std::vector<Message> norm;
  norm.reserve(messages.size());
  for (const auto& m : messages) norm.push_back(normalized(m));

  if (!options.prefill_assistant || norm.empty() || norm.back().role != "assistant") {
    return render_messages(norm, options);
  }
  if (norm.size() >= 2 && norm[norm.size() - 2].role == "assistant") {
    throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
  }
  std::vector<Message> head(norm.begin(), norm.end() - 1);
  return render_messages(head, options) + continuation_text(norm.back());
}

}  // namespace inkling
