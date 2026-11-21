// tokenizer_probe [model.gguf] < cases.jsonl > results.jsonl
// One case per input line: {"text": "...", "parse_special": true|false}
// One result per line:    {"ids": [...], "text": "<decode(ids)>"}
#include <cstdio>
#include <iostream>
#include <string>

#include "../core/gguf.hpp"
#include "../host/tokenizer.hpp"
#include "mini_json.hpp"

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "/models/Inkling-2-Mini-UD-Q4_K_M.gguf";
  try {
    inkling::Gguf gguf(path);
    inkling::Tokenizer tok(gguf);

    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const auto req = inkling::json::parse(line);
      const auto* text = req.get("text");
      if (!text || !text->is_string()) throw std::runtime_error("case without text");
      const auto* ps = req.get("parse_special");
      const bool parse_special = ps && ps->is_bool() && ps->b;

      const auto ids = tok.encode(text->s, parse_special);
      std::string out = "{\"ids\":[";
      for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(ids[i]);
      }
      out += "],\"text\":" + inkling::json::quote(tok.decode(ids)) + "}\n";
      std::fwrite(out.data(), 1, out.size(), stdout);
    }
    std::fflush(stdout);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tokenizer_probe: %s\n", e.what());
    return 1;
  }
  return 0;
}
