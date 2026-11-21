#include "chat_template.hpp"

#include <algorithm>
#include <cstdio>

namespace inkling {
namespace {

const char* const kEnd = "<|end_message|>";
const char* const kText = "<|content_text|>";
const char* const kThink = "<|content_thinking|>";
const char* const kSampling = "<|content_model_end_sampling|>";
const char* const kModel = "<|message_model|>";
const char* const kInvoke = "<|content_invoke_tool_json|>";
const char* const kSystemEffort = "<|message_system|><|content_text|>Thinking effort level: ";
const char* const kToolDeclare = "<|message_system|>tool_declare<|content_xml|>";

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

// The tool block at the top of the template: one declaration per tool, sorted keys, compact.
// Validation follows the server (llama.cpp oaicompat parsing), which rejects what the template could not render.
std::string render_tools(const std::vector<json::Value>& tools) {
  if (tools.empty()) return "";
  json::Value specs = json::Value::array();
  for (const auto& tool : tools) {
    const json::Value* type = tool.find("type");
    if (!type) throw std::invalid_argument("Missing tool type");
    if (!type->is_string() || type->s != "function") throw std::invalid_argument("Unsupported tool type");
    const json::Value* fn = tool.find("function");
    if (!fn) throw std::invalid_argument("Missing tool function");
    if (!fn->is_object()) throw std::invalid_argument("tool function must be an object");
    const json::Value* name = fn->find("name");
    if (!name || !name->is_string()) throw std::invalid_argument("tool name must be a string");

    // defer_loading: the function's value wins; otherwise the tool's value; must be a boolean when present
    if (const json::Value* d = fn->find("defer_loading")) {
      if (!d->is_bool()) throw std::invalid_argument("Expected 'defer_loading' to be a boolean");
    } else if (const json::Value* d2 = tool.find("defer_loading")) {
      if (!d2->is_bool()) throw std::invalid_argument("Expected 'defer_loading' to be a boolean");
    }

    json::Value description = json::Value::string("");
    if (const json::Value* d = fn->find("description")) {
      if (!d->is_string()) throw std::invalid_argument("tool description must be a string");
      if (!d->s.empty()) description = *d;
    }
    json::Value parameters = json::Value::object();
    if (const json::Value* p = fn->find("parameters")) {
      if (json::truthy(*p)) parameters = *p;
    }
    json::Value spec = json::Value::object();
    spec.set("description", description);
    spec.set("name", *name);
    spec.set("parameters", parameters);
    spec.set("type", json::Value::string("function"));
    specs.a.push_back(std::move(spec));
  }
  std::string out = kToolDeclare;
  json::dump(out, specs, /*sort_keys=*/true);
  out += kEnd;
  return out;
}

// The name of a tool message's header: its own name, else the name of every call with the same id.
std::string tool_name_for(const Message& m, const std::vector<Message>& all) {
  if (!m.name.empty()) return m.name;
  std::string out;
  if (m.tool_call_id.empty()) return out;
  for (const auto& prev : all) {
    if (prev.role != "assistant") continue;
    for (const auto& tc : prev.tool_calls) {
      if (!tc.id.empty() && tc.id == m.tool_call_id) out += tc.name;
    }
  }
  return out;
}

// Header of a message: role token, then the name for tool messages and named assistant messages.
std::string header_of(const Message& m, const std::vector<Message>& all) {
  std::string header = role_token(m.role);
  if (m.role == "tool") {
    header += tool_name_for(m, all);
  } else if (m.role == "assistant" && !m.name.empty()) {
    header += m.name;
  }
  return header;
}

// The members of a call's "args" object: sorted by key at top level (dictsort), values in their own order, compact.
// Mirrors the template's checks: a falsy value means no arguments, a string after the server's parse is an error,
// anything but an object is an error.
std::string render_arguments(const ToolCall& tc) {
  if (!tc.arguments) throw std::invalid_argument("Failed to parse messages: key 'arguments' not found");
  json::Value v = *tc.arguments;
  if (v.is_string()) {
    try {
      v = json::parse(v.s);
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string("Failed to parse tool call arguments as JSON: ") + e.what());
    }
  }
  if (!json::truthy(v)) return "";
  if (v.is_string()) {
    throw std::invalid_argument("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream");
  }
  if (!v.is_object()) throw std::invalid_argument("tool call arguments must be an object");

  std::vector<const std::pair<std::string, json::Value>*> members;
  for (const auto& kv : v.o) members.push_back(&kv);
  std::sort(members.begin(), members.end(), [](const auto* x, const auto* y) { return x->first < y->first; });
  std::string out;
  for (size_t i = 0; i < members.size(); ++i) {
    if (i) out += ',';
    out += json::quote(members[i]->first);
    out += ':';
    json::dump(out, members[i]->second, /*sort_keys=*/false);
  }
  return out;
}

// the main loop of the template over messages[0, count); `all` gives the lookups of tool_call_id
std::string render_messages(const std::vector<Message>& all, size_t count, const Options& opt) {
  std::string out;
  bool effort_emitted = false;

  for (size_t i = 0; i < count; ++i) {
    const Message& m = all[i];
    if (!role_token(m.role)) throw std::invalid_argument("Unknown message role: " + m.role);
    if (!effort_emitted && m.role != "system") {
      out += render_effort(opt);
      effort_emitted = true;
    }
    const bool assistant = m.role == "assistant";
    const bool has_reasoning = assistant && !m.reasoning_content.empty();
    const bool has_calls = assistant && !m.tool_calls.empty();
    const std::string header = header_of(m, all);

    std::string rendered;
    if (has_reasoning) {
      rendered += header + kThink + m.reasoning_content + kEnd;
    }

    if (m.content == Message::Content::Text && (!m.text.empty() || !(has_reasoning || has_calls))) {
      rendered += header + kText + m.text + kEnd;
    } else if (m.content == Message::Content::Parts && !m.parts.empty()) {
      for (const auto& p : m.parts) {
        if (p.type == "image") {
          if (m.role != "user" && m.role != "tool") throw std::invalid_argument("Image content is only supported for user and tool messages");
          rendered += header + "<|content_image|><|unused_200054|>" + kEnd;
          continue;
        }
        if (p.type != "text") throw std::invalid_argument("Unsupported content part type: " + p.type);
        rendered += header + kText + p.text + kEnd;
      }
    }

    // tool calls: the call is named by the model header, not by the message header
    for (const auto& tc : m.tool_calls) {
      if (!m.name.empty() && m.name != tc.name) {
        throw std::invalid_argument("assistant name " + m.name + " does not match its tool call " + tc.name);
      }
      rendered += kModel + tc.name + kInvoke + "{\"name\":" + json::quote(tc.name) + ",\"args\":{" +
                  render_arguments(tc) + "}}" + kEnd;
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
    if (m.role == "assistant" && !m.tool_calls.empty()) continue;
    if (m.role == "assistant") {
      throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
    }
    throw std::invalid_argument("All non-assistant messages must contain 'content'");
  }

  std::vector<Message> norm;
  norm.reserve(messages.size());
  for (const auto& m : messages) norm.push_back(normalized(m));

  const std::string tools = render_tools(options.tools);

  if (!options.prefill_assistant || norm.empty() || norm.back().role != "assistant") {
    return tools + render_messages(norm, norm.size(), options);
  }
  if (norm.size() >= 2 && norm[norm.size() - 2].role == "assistant") {
    throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
  }
  if (!norm.back().tool_calls.empty()) {
    throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
  }
  return tools + render_messages(norm, norm.size() - 1, options) + continuation_text(norm.back());
}

}  // namespace inkling
