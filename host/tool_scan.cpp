#include "tool_scan.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "json.hpp"

namespace inkling {
namespace {

constexpr std::string_view kHeader = "<|message_model|>";
constexpr std::string_view kInvoke = "<|content_invoke_tool_json|>";
constexpr std::string_view kEnd = "<|end_message|>";

enum class St { Ok, Incomplete, Invalid };

struct Cursor {
  std::string_view t;
  size_t i = 0;
  bool at_end() const { return i >= t.size(); }
};

void skip_ws(Cursor& c) {
  while (c.i < c.t.size() && (c.t[c.i] == ' ' || c.t[c.i] == '\t' || c.t[c.i] == '\n' || c.t[c.i] == '\r')) ++c.i;
}

// Matches a fixed token at the cursor. Incomplete when the text ends inside it (cursor not moved).
St literal(Cursor& c, std::string_view lit) {
  const size_t rem = c.t.size() - c.i;
  const size_t n = std::min(rem, lit.size());
  if (c.t.substr(c.i, n) != lit.substr(0, n)) return St::Invalid;
  if (rem < lit.size()) return St::Incomplete;
  c.i += lit.size();
  return St::Ok;
}

int hex_digit(char h) {
  if (h >= '0' && h <= '9') return h - '0';
  if (h >= 'a' && h <= 'f') return h - 'a' + 10;
  if (h >= 'A' && h <= 'F') return h - 'A' + 10;
  return -1;
}

// Reads the 4 hex digits at t[at..at+4). Returns -1 when one of them is not hex.
int64_t hex4_at(std::string_view t, size_t at) {
  int64_t v = 0;
  for (size_t k = 0; k < 4; ++k) {
    int d = hex_digit(t[at + k]);
    if (d < 0) return -1;
    v = v * 16 + d;
  }
  return v;
}

void put_utf8(std::string& out, uint32_t c) {
  if (c < 0x80) {
    out += static_cast<char>(c);
  } else if (c < 0x800) {
    out += static_cast<char>(0xC0 | (c >> 6));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    out += static_cast<char>(0xE0 | (c >> 12));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (c >> 18));
    out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  }
}

// A JSON string at the cursor, decoded. On Incomplete, `out` holds the decoded prefix that is certain (an escape cut
// by the end of the text is not decoded). Lone surrogates decode to U+FFFD.
St json_string(Cursor& c, std::string& out) {
  out.clear();
  if (c.at_end()) return St::Incomplete;
  if (c.t[c.i] != '"') return St::Invalid;
  const std::string_view t = c.t;
  size_t i = c.i + 1;
  while (true) {
    if (i >= t.size()) return St::Incomplete;
    const unsigned char ch = static_cast<unsigned char>(t[i]);
    if (ch == '"') {
      c.i = i + 1;
      return St::Ok;
    }
    if (ch < 0x20) return St::Invalid;
    if (ch != '\\') {
      out += static_cast<char>(ch);
      ++i;
      continue;
    }
    if (i + 1 >= t.size()) return St::Incomplete;
    const char e = t[i + 1];
    switch (e) {
      case '"': out += '"'; i += 2; break;
      case '\\': out += '\\'; i += 2; break;
      case '/': out += '/'; i += 2; break;
      case 'b': out += '\b'; i += 2; break;
      case 'f': out += '\f'; i += 2; break;
      case 'n': out += '\n'; i += 2; break;
      case 'r': out += '\r'; i += 2; break;
      case 't': out += '\t'; i += 2; break;
      case 'u': {
        if (i + 6 > t.size()) {
          // cut inside the escape: fine if the digits seen so far are hex
          for (size_t k = i + 2; k < t.size(); ++k) {
            if (hex_digit(t[k]) < 0) return St::Invalid;
          }
          return St::Incomplete;
        }
        const int64_t v = hex4_at(t, i + 2);
        if (v < 0) return St::Invalid;
        uint32_t cp = static_cast<uint32_t>(v);
        if (cp >= 0xD800 && cp < 0xDC00) {
          // a high surrogate pairs with a following \uDC00-\uDFFF escape; wait for the text when it is cut there
          const size_t next = i + 6;
          if (next >= t.size()) return St::Incomplete;
          if (t[next] == '\\') {
            if (next + 1 >= t.size()) return St::Incomplete;
            if (t[next + 1] == 'u') {
              if (next + 6 > t.size()) return St::Incomplete;
              const int64_t lo = hex4_at(t, next + 2);
              if (lo < 0) return St::Invalid;
              if (lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (static_cast<uint32_t>(lo) - 0xDC00);
                i += 6;
              }
            }
          }
        }
        if (cp >= 0xD800 && cp < 0xE000) cp = 0xFFFD;
        put_utf8(out, cp);
        i += 6;
        break;
      }
      default: return St::Invalid;
    }
  }
}

// A non-string value at the cursor (object, array, number, literal). `end` receives the end of its text.
// Literals and numbers end at a delimiter, or at the end of the text when the text is final.
St raw_value(Cursor& c, bool final, size_t& end) {
  const std::string_view t = c.t;
  if (c.t[c.i] == '{' || c.t[c.i] == '[') {
    int depth = 0;
    bool in_string = false;
    for (size_t i = c.i; i < t.size(); ++i) {
      const char ch = t[i];
      if (in_string) {
        if (ch == '\\') {
          ++i;
        } else if (ch == '"') {
          in_string = false;
        }
        continue;
      }
      if (ch == '"') {
        in_string = true;
      } else if (ch == '{' || ch == '[') {
        ++depth;
      } else if (ch == '}' || ch == ']') {
        if (--depth == 0) {
          end = i + 1;
          return St::Ok;
        }
      }
    }
    return St::Incomplete;
  }
  size_t i = c.i;
  while (i < t.size() && t[i] != ',' && t[i] != '}' && t[i] != ']' && t[i] != ' ' && t[i] != '\t' && t[i] != '\n' &&
         t[i] != '\r') {
    ++i;
  }
  if (i == t.size() && !final) return St::Incomplete;
  end = i;
  return St::Ok;
}

struct CallBody {
  std::vector<ToolArg> args;
  bool last_open = false;     // the last argument is a string cut short
  bool closed = false;        // the outer closing brace was read
  size_t closed_at = 0;       // position after the outer closing brace
  bool end_marker = false;    // the end marker follows the closing brace
  size_t marker_end = 0;      // position after the end marker
};

// The JSON part of a call, from the cursor (just after the invoke marker) to the end marker. Two spellings: the wrapped one the
// template renders and the grammar of the server forces, {"name":NAME,"args":{...}}, and the bare argument object {...} that the
// model writes when nothing constrains it. The wrapper is recognised by its exact prefix with the header's own name.
St call_body(Cursor& c, bool final, CallBody& body, std::string_view header_name) {
  skip_ws(c);
  if (c.at_end()) return St::Incomplete;
  bool wrapped = false;
  {
    // {"name":NAME,"args": with optional spaces between tokens
    Cursor probe = c;
    const auto step = [&](std::string_view lit) { skip_ws(probe); if (probe.at_end()) return St::Incomplete; return literal(probe, lit); };
    St st = step("{");
    if (st == St::Ok) st = step("\"name\"");
    if (st == St::Ok) st = step(":");
    std::string nm;
    if (st == St::Ok) { skip_ws(probe); if (probe.at_end()) return St::Incomplete; st = json_string(probe, nm); }
    if (st == St::Ok) st = step(",");
    if (st == St::Ok) st = step("\"args\"");
    if (st == St::Ok) st = step(":");
    if (st == St::Incomplete) return St::Incomplete;
    if (st == St::Ok) {
      (void)header_name;  // a wrapper that names another tool still calls the header's tool, as before
      wrapped = true; c = probe;
    }
  }
  St st = St::Ok;
  skip_ws(c);
  if (c.at_end()) return St::Incomplete;
  if (c.t[c.i] != '{') return St::Invalid;
  ++c.i;

  bool first = true;
  while (true) {
    skip_ws(c);
    if (c.at_end()) return St::Incomplete;
    if (c.t[c.i] == '}' && first) {
      ++c.i;
      break;
    }
    if (c.t[c.i] != '"') return St::Invalid;
    first = false;
    std::string key;
    st = json_string(c, key);
    if (st == St::Incomplete) return St::Incomplete;
    if (st != St::Ok) return St::Invalid;
    skip_ws(c);
    if (c.at_end()) return St::Incomplete;
    if (c.t[c.i] != ':') return St::Invalid;
    ++c.i;
    skip_ws(c);
    if (c.at_end()) return St::Incomplete;

    if (c.t[c.i] == '"') {
      std::string value;
      st = json_string(c, value);
      if (st == St::Incomplete) {
        body.args.push_back({key, value, true});  // the string read so far
        body.last_open = true;
        return St::Incomplete;
      }
      if (st != St::Ok) return St::Invalid;
      body.args.push_back({key, value, true});
    } else {
      size_t end = 0;
      st = raw_value(c, final, end);
      if (st == St::Incomplete) return St::Incomplete;
      std::string raw(c.t.substr(c.i, end - c.i));
      try {
        json::parse(raw);
      } catch (const std::invalid_argument&) {
        return St::Invalid;
      }
      body.args.push_back({key, raw, false});
      c.i = end;
    }

    skip_ws(c);
    if (c.at_end()) return St::Incomplete;
    if (c.t[c.i] == ',') {
      ++c.i;
      continue;
    }
    if (c.t[c.i] == '}') {
      ++c.i;
      break;
    }
    return St::Invalid;
  }

  if (wrapped) {
    skip_ws(c);
    if (c.at_end()) return St::Incomplete;
    if (c.t[c.i] != '}') return St::Invalid;
    ++c.i;
  }
  body.closed = true;
  body.closed_at = c.i;

  // the end marker is optional here: any other text after the braces just means there is no marker
  Cursor probe = c;
  skip_ws(probe);
  if (literal(probe, kEnd) == St::Ok) {
    body.end_marker = true;
    body.marker_end = probe.i;
  }
  return St::Ok;
}

}  // namespace

void scan_tool_calls(std::string_view text, std::vector<ToolCallOut>* calls,
                     std::vector<std::pair<size_t, size_t>>* spans, bool partial) {
  if (calls) calls->clear();
  if (spans) spans->clear();

  size_t pos = 0;
  while (true) {
    const size_t h = text.find(kHeader, pos);
    if (h == std::string_view::npos) break;
    const size_t name_at = h + kHeader.size();
    const size_t lt = text.find('<', name_at);
    if (lt == std::string_view::npos) break;  // the name is not finished: nothing can follow it yet
    const std::string_view name = text.substr(name_at, lt - name_at);

    if (text.substr(lt, kInvoke.size()) != kInvoke) {
      // a marker cut by the end of the text may still become the invoke marker: wait for more text
      const std::string_view rest = text.substr(lt);
      if (rest.size() < kInvoke.size() && kInvoke.substr(0, rest.size()) == rest) break;
      pos = lt;  // another block (text, thinking, ...)
      continue;
    }
    if (name.empty()) {
      pos = lt;
      continue;
    }

    Cursor c{text, lt + kInvoke.size()};
    CallBody body;
    const bool final = !partial;
    const St st = call_body(c, final, body, name);

    if (st == St::Invalid) {
      pos = lt;
      continue;
    }

    ToolCallOut call;
    call.name = std::string(name);
    if (st == St::Incomplete) {
      call.args = std::move(body.args);
      call.last_open = body.last_open;
      call.complete = false;
      if (calls) calls->push_back(std::move(call));
      if (spans) spans->push_back({h, text.size()});
      break;  // the text ends inside this call
    }

    // closed: complete with its end marker; without it, complete unless partial and the text ends right here
    const bool complete = body.end_marker || !partial || body.closed_at < text.size();
    call.args = std::move(body.args);
    call.complete = complete;
    const size_t stop = body.end_marker ? body.marker_end : body.closed_at;
    if (calls) calls->push_back(std::move(call));
    if (spans) spans->push_back({h, stop});
    pos = stop;
    if (!complete) break;
  }
}

}  // namespace inkling
