// Minimal JSON reader/writer for the probe tools (tokenizer_probe, template_probe).
// Only what the gates need: objects, arrays, strings (with \u escapes), numbers, booleans, null.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inkling::json {

struct Value {
  enum Type { Null, Bool, Number, String, Array, Object } type = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Value> a;
  std::vector<std::pair<std::string, Value>> o;

  bool is_null() const { return type == Null; }
  bool is_string() const { return type == String; }
  bool is_number() const { return type == Number; }
  bool is_bool() const { return type == Bool; }
  bool is_array() const { return type == Array; }
  bool is_object() const { return type == Object; }

  // nullptr when absent (or not an object)
  const Value* get(const std::string& key) const {
    if (type != Object) return nullptr;
    for (auto& kv : o) if (kv.first == key) return &kv.second;
    return nullptr;
  }
};

namespace detail {
struct Parser {
  const std::string& src;
  size_t pos = 0;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("json: ") + what + " at byte " + std::to_string(pos));
  }
  void ws() { while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r')) ++pos; }
  char peek() { ws(); if (pos >= src.size()) fail("unexpected end"); return src[pos]; }
  void expect(char c) { if (peek() != c) fail("unexpected character"); ++pos; }

  static void put_utf8(std::string& out, uint32_t c) {
    if (c < 0x80) out += (char)c;
    else if (c < 0x800) { out += (char)(0xC0 | (c >> 6)); out += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { out += (char)(0xE0 | (c >> 12)); out += (char)(0x80 | ((c >> 6) & 0x3F)); out += (char)(0x80 | (c & 0x3F)); }
    else { out += (char)(0xF0 | (c >> 18)); out += (char)(0x80 | ((c >> 12) & 0x3F)); out += (char)(0x80 | ((c >> 6) & 0x3F)); out += (char)(0x80 | (c & 0x3F)); }
  }
  uint32_t hex4() {
    if (pos + 4 > src.size()) fail("bad \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char h = src[pos++];
      v <<= 4;
      if (h >= '0' && h <= '9') v |= h - '0';
      else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10;
      else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10;
      else fail("bad hex digit");
    }
    return v;
  }
  std::string str() {
    expect('"');
    std::string out;
    while (true) {
      if (pos >= src.size()) fail("unterminated string");
      char c = src[pos++];
      if (c == '"') break;
      if (c != '\\') { out += c; continue; }
      if (pos >= src.size()) fail("bad escape");
      char e = src[pos++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t c1 = hex4();
          if (c1 >= 0xD800 && c1 < 0xDC00 && pos + 1 < src.size() && src[pos] == '\\' && src[pos + 1] == 'u') {
            pos += 2;
            uint32_t c2 = hex4();
            c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
          }
          put_utf8(out, c1);
          break;
        }
        default: fail("unknown escape");
      }
    }
    return out;
  }
  Value value() {
    char c = peek();
    Value v;
    if (c == '{') {
      ++pos; v.type = Value::Object;
      if (peek() == '}') { ++pos; return v; }
      while (true) {
        std::string k = str();
        expect(':');
        v.o.emplace_back(std::move(k), value());
        char d = peek();
        ++pos;
        if (d == '}') break;
        if (d != ',') fail("expected , or }");
      }
    } else if (c == '[') {
      ++pos; v.type = Value::Array;
      if (peek() == ']') { ++pos; return v; }
      while (true) {
        v.a.push_back(value());
        char d = peek();
        ++pos;
        if (d == ']') break;
        if (d != ',') fail("expected , or ]");
      }
    } else if (c == '"') {
      v.type = Value::String; v.s = str();
    } else if (src.compare(pos, 4, "true") == 0) {
      pos += 4; v.type = Value::Bool; v.b = true;
    } else if (src.compare(pos, 5, "false") == 0) {
      pos += 5; v.type = Value::Bool; v.b = false;
    } else if (src.compare(pos, 4, "null") == 0) {
      pos += 4;
    } else {
      size_t start = pos;
      while (pos < src.size() && std::string("+-0123456789.eE").find(src[pos]) != std::string::npos) ++pos;
      if (start == pos) fail("unexpected token");
      v.type = Value::Number;
      v.n = std::strtod(src.substr(start, pos - start).c_str(), nullptr);
    }
    return v;
  }
};
}  // namespace detail

inline Value parse(const std::string& text) {
  detail::Parser p{text};
  Value v = p.value();
  p.ws();
  if (p.pos != text.size()) p.fail("trailing characters");
  return v;
}

// JSON string literal (with quotes). Non-ASCII UTF-8 is written as is.
inline std::string quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += (char)c;
        }
    }
  }
  return out + "\"";
}

}  // namespace inkling::json
