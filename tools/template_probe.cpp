// template_probe < cases.jsonl > results.jsonl
// One case per input line:
//   {"messages": [...], "reasoning_effort": "high" | null, "reasoning_effort_value": 0.5 | null,
//    "prefill_assistant": true | false}   (prefill_assistant defaults to true)
// A message: {"role": "user", "content": "text" | null | [{"type": "text", "text": "..."}], "reasoning_content": "...",
//             "name": "..."}; a missing "content" key is Absent.
// One result per line: {"prompt": "..."} or {"error": "..."}
#include <cstdio>
#include <iostream>
#include <string>

#include "../host/chat_template.hpp"
#include "mini_json.hpp"

namespace {

inkling::Message parse_message(const inkling::json::Value& v) {
  inkling::Message m;
  const auto* role = v.get("role");
  if (!role || !role->is_string()) throw std::runtime_error("message without role");
  m.role = role->s;
  if (const auto* c = v.get("content")) {
    if (c->is_null()) {
      m.content = inkling::Message::Content::Null;
    } else if (c->is_string()) {
      m.content = inkling::Message::Content::Text;
      m.text = c->s;
    } else if (c->is_array()) {
      m.content = inkling::Message::Content::Parts;
      for (const auto& p : c->a) {
        const auto* type = p.get("type");
        const auto* text = p.get("text");
        m.parts.push_back({type && type->is_string() ? type->s : "", text && text->is_string() ? text->s : ""});
      }
    } else {
      throw std::runtime_error("bad content");
    }
  }
  if (const auto* r = v.get("reasoning_content"); r && r->is_string()) m.reasoning_content = r->s;
  if (const auto* n = v.get("name"); n && n->is_string()) m.name = n->s;
  return m;
}

}  // namespace

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::string out;
    try {
      const auto req = inkling::json::parse(line);
      std::vector<inkling::Message> messages;
      for (const auto& mv : req.get("messages")->a) messages.push_back(parse_message(mv));

      inkling::Options opt;
      if (const auto* p = req.get("prefill_assistant"); p && p->is_bool()) opt.prefill_assistant = p->b;
      if (const auto* e = req.get("reasoning_effort"); e && e->is_string()) opt.reasoning_effort = e->s;
      if (const auto* v = req.get("reasoning_effort_value"); v && v->is_number()) opt.reasoning_effort_value = v->n;

      out = "{\"prompt\":" + inkling::json::quote(inkling::render_prompt(messages, opt)) + "}\n";
    } catch (const std::exception& e) {
      out = "{\"error\":" + inkling::json::quote(e.what()) + "}\n";
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
  }
  std::fflush(stdout);
  return 0;
}
