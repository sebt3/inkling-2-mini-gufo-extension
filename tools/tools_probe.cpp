// tools_probe < cases.jsonl > results.jsonl
// One case per input line (same shape as template_probe, plus tools):
//   {"messages": [...], "tools": [...] | null, "reasoning_effort": "high" | null,
//    "reasoning_effort_value": 0.5 | null, "prefill_assistant": true | false}
// A message: {"role", "content" (string | null | [{"type","text"}] | absent), "reasoning_content", "name",
//             "tool_call_id", "tool_calls": [{"id", "type": "function", "function": {"name", "arguments"}}]}
// "arguments" absent is an error on the template side; "content" absent is Absent.
// One result per line: {"prompt": "..."} or {"error": "..."}
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../host/chat_template.hpp"
#include "../host/json.hpp"

namespace {

using inkling::json::Value;

inkling::ToolCall parse_tool_call(const Value& v) {
  const Value* type = v.find("type");
  if (!type) throw std::runtime_error("Missing tool call type");
  if (!type->is_string() || type->s != "function") throw std::runtime_error("Unsupported tool call type");
  const Value* fn = v.find("function");
  if (!fn) throw std::runtime_error("Missing tool call function");
  const Value* name = fn->find("name");
  if (!name) throw std::runtime_error("Missing tool call name");
  if (!name->is_string()) throw std::runtime_error("type must be string, but is not");
  inkling::ToolCall tc;
  tc.name = name->s;
  if (const Value* id = v.find("id")) {
    if (!id->is_string()) throw std::runtime_error("type must be string, but is not");
    tc.id = id->s;
  }
  if (const Value* args = fn->find("arguments")) tc.arguments = *args;
  return tc;
}

inkling::Message parse_message(const Value& v) {
  inkling::Message m;
  const Value* role = v.find("role");
  if (!role || !role->is_string()) throw std::runtime_error("message without role");
  m.role = role->s;
  if (const Value* c = v.find("content")) {
    if (c->is_null()) {
      m.content = inkling::Message::Content::Null;
    } else if (c->is_string()) {
      m.content = inkling::Message::Content::Text;
      m.text = c->s;
    } else if (c->is_array()) {
      m.content = inkling::Message::Content::Parts;
      for (const auto& p : c->a) {
        const Value* type = p.find("type");
        const Value* text = p.find("text");
        m.parts.push_back({type && type->is_string() ? type->s : "", text && text->is_string() ? text->s : ""});
      }
    } else {
      throw std::runtime_error("bad content");
    }
  }
  if (const Value* r = v.find("reasoning_content"); r && r->is_string()) m.reasoning_content = r->s;
  if (const Value* n = v.find("name"); n && n->is_string()) m.name = n->s;
  if (const Value* t = v.find("tool_call_id"); t && t->is_string()) m.tool_call_id = t->s;
  if (const Value* calls = v.find("tool_calls")) {
    if (!calls->is_array()) throw std::runtime_error("tool_calls must be an array");
    for (const auto& c : calls->a) m.tool_calls.push_back(parse_tool_call(c));
  }
  return m;
}

}  // namespace

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::string out;
    try {
      const Value req = inkling::json::parse(line);
      std::vector<inkling::Message> messages;
      for (const auto& mv : req.find("messages")->a) messages.push_back(parse_message(mv));

      inkling::Options opt;
      if (const Value* t = req.find("tools"); t && t->is_array()) opt.tools = t->a;
      if (const Value* p = req.find("prefill_assistant"); p && p->is_bool()) opt.prefill_assistant = p->b;
      if (const Value* e = req.find("reasoning_effort"); e && e->is_string()) opt.reasoning_effort = e->s;
      if (const Value* v = req.find("reasoning_effort_value"); v && v->is_number()) opt.reasoning_effort_value = v->n;

      out = "{\"prompt\":" + inkling::json::quote(inkling::render_prompt(messages, opt)) + "}\n";
    } catch (const std::exception& e) {
      out = "{\"error\":" + inkling::json::quote(e.what()) + "}\n";
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
  }
  std::fflush(stdout);
  return 0;
}
