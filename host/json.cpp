#include "json.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace inkling::json {
namespace {

struct Parser {
  std::string_view src;
  size_t pos = 0;

  [[noreturn]] void fail(const std::string& what) const {
    throw std::invalid_argument("json: " + what + " at byte " + std::to_string(pos));
  }
  void ws() {
    while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\n' || src[pos] == '\r')) ++pos;
  }
  char peek() {
    ws();
    if (pos >= src.size()) fail("unexpected end of input");
    return src[pos];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos;
  }
  bool literal(std::string_view word) {
    if (src.substr(pos, word.size()) != word) return false;
    pos += word.size();
    return true;
  }

  static void put_utf8(std::string& out, uint32_t c) {
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

  uint32_t hex4() {
    if (pos + 4 > src.size()) fail("bad \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char h = src[pos++];
      v <<= 4;
      if (h >= '0' && h <= '9') {
        v |= static_cast<uint32_t>(h - '0');
      } else if (h >= 'a' && h <= 'f') {
        v |= static_cast<uint32_t>(h - 'a' + 10);
      } else if (h >= 'A' && h <= 'F') {
        v |= static_cast<uint32_t>(h - 'A' + 10);
      } else {
        fail("bad hex digit");
      }
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
      if (c != '\\') {
        out += c;
        continue;
      }
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
            size_t save = pos;
            pos += 2;
            uint32_t c2 = hex4();
            if (c2 >= 0xDC00 && c2 < 0xE000) {
              c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
            } else {
              pos = save;  // not a low surrogate: decoded on its own below
            }
          }
          if (c1 >= 0xD800 && c1 < 0xE000) c1 = 0xFFFD;  // lone surrogate
          put_utf8(out, c1);
          break;
        }
        default: fail("unknown escape");
      }
    }
    return out;
  }

  static bool digit(char c) { return c >= '0' && c <= '9'; }

  Value number() {
    size_t start = pos;
    if (pos < src.size() && src[pos] == '-') ++pos;
    if (pos >= src.size() || !digit(src[pos])) fail("bad number");
    if (src[pos] == '0') {
      ++pos;
    } else {
      while (pos < src.size() && digit(src[pos])) ++pos;
    }
    if (pos < src.size() && src[pos] == '.') {
      ++pos;
      if (pos >= src.size() || !digit(src[pos])) fail("bad fraction");
      while (pos < src.size() && digit(src[pos])) ++pos;
    }
    if (pos < src.size() && (src[pos] == 'e' || src[pos] == 'E')) {
      ++pos;
      if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) ++pos;
      if (pos >= src.size() || !digit(src[pos])) fail("bad exponent");
      while (pos < src.size() && digit(src[pos])) ++pos;
    }
    Value v;
    v.type = Value::Type::Number;
    v.lexeme = std::string(src.substr(start, pos - start));
    v.n = std::strtod(v.lexeme.c_str(), nullptr);
    return v;
  }

  Value value(int depth) {
    if (depth > 512) fail("nesting too deep");
    char c = peek();
    Value v;
    if (c == '{') {
      ++pos;
      v.type = Value::Type::Object;
      if (peek() == '}') {
        ++pos;
        return v;
      }
      while (true) {
        if (peek() != '"') fail("expected object key");
        std::string k = str();
        expect(':');
        v.set(std::move(k), value(depth + 1));
        char d = peek();
        ++pos;
        if (d == '}') break;
        if (d != ',') fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      ++pos;
      v.type = Value::Type::Array;
      if (peek() == ']') {
        ++pos;
        return v;
      }
      while (true) {
        v.a.push_back(value(depth + 1));
        char d = peek();
        ++pos;
        if (d == ']') break;
        if (d != ',') fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.type = Value::Type::String;
      v.s = str();
    } else if (literal("true")) {
      v.type = Value::Type::Bool;
      v.b = true;
    } else if (literal("false")) {
      v.type = Value::Type::Bool;
      v.b = false;
    } else if (literal("null")) {
      v.type = Value::Type::Null;
    } else {
      return number();
    }
    return v;
  }
};

bool integer_lexeme(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s) {
    if (!(c == '-' || (c >= '0' && c <= '9'))) return false;
  }
  return true;
}

void dump_value(std::string& out, const Value& v, bool sort_keys);

void dump_members(std::string& out, const Value& v, bool sort_keys) {
  std::vector<const std::pair<std::string, Value>*> members;
  members.reserve(v.o.size());
  for (const auto& kv : v.o) members.push_back(&kv);
  if (sort_keys) {
    std::sort(members.begin(), members.end(),
              [](const auto* x, const auto* y) { return x->first < y->first; });
  }
  out += '{';
  bool first = true;
  for (const auto* kv : members) {
    if (!first) out += ',';
    first = false;
    out += quote(kv->first);
    out += ':';
    dump_value(out, kv->second, sort_keys);
  }
  out += '}';
}

void dump_value(std::string& out, const Value& v, bool sort_keys) {
  switch (v.type) {
    case Value::Type::Null: out += "null"; break;
    case Value::Type::Bool: out += v.b ? "true" : "false"; break;
    case Value::Type::Number: out += number_text(v); break;
    case Value::Type::String: out += quote(v.s); break;
    case Value::Type::Array: {
      out += '[';
      for (size_t i = 0; i < v.a.size(); ++i) {
        if (i) out += ',';
        dump_value(out, v.a[i], sort_keys);
      }
      out += ']';
      break;
    }
    case Value::Type::Object: dump_members(out, v, sort_keys); break;
  }
}

}  // namespace

Value parse(std::string_view text) {
  Parser p{text};
  Value v = p.value(0);
  p.ws();
  if (p.pos != text.size()) p.fail("trailing characters");
  return v;
}

bool truthy(const Value& v) {
  switch (v.type) {
    case Value::Type::Null: return false;
    case Value::Type::Bool: return v.b;
    case Value::Type::Number: return v.n != 0.0;
    case Value::Type::String: return !v.s.empty();
    case Value::Type::Array: return !v.a.empty();
    case Value::Type::Object: return !v.o.empty();
  }
  return false;
}

std::string quote(std::string_view s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
  return out;
}

void dump(std::string& out, const Value& v, bool sort_keys) { dump_value(out, v, sort_keys); }

std::string number_text(const Value& v) {
  if (integer_lexeme(v.lexeme)) {
    errno = 0;
    long long i = std::strtoll(v.lexeme.c_str(), nullptr, 10);
    if (errno == 0) {
      if (i == 0) return "0";  // "-0" is an integer 0
      return std::to_string(i);
    }
  }
  const double x = v.n;
  if (x == 0.0) return std::signbit(x) ? "-0.0" : "0.0";
  const bool neg = x < 0;
  const double ax = std::fabs(x);

  // shortest decimal digits that round-trip
  char buf[64];
  for (int p = 1; p <= 17; ++p) {
    std::snprintf(buf, sizeof buf, "%.*e", p - 1, ax);
    if (std::strtod(buf, nullptr) == ax) break;
  }
  // buf: d[.ddd]e[+-]XX
  std::string digits;
  const char* q = buf;
  for (; *q && *q != 'e'; ++q) {
    if (*q != '.') digits += *q;
  }
  int e10 = std::atoi(q + 1);
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  const int len = static_cast<int>(digits.size());

  std::string out = neg ? "-" : "";
  if (e10 >= -4 && e10 <= 15) {
    if (e10 >= 0) {
      std::string ip;
      for (int i = 0; i <= e10; ++i) ip += i < len ? digits[static_cast<size_t>(i)] : '0';
      std::string fp = len > e10 + 1 ? digits.substr(static_cast<size_t>(e10 + 1)) : "0";
      out += ip + "." + fp;
    } else {
      out += "0." + std::string(static_cast<size_t>(-e10 - 1), '0') + digits;
    }
  } else {
    out += digits[0];
    if (len > 1) out += "." + digits.substr(1);
    out += e10 < 0 ? "e-" : "e+";
    const int ae = e10 < 0 ? -e10 : e10;
    if (ae < 10) out += '0';
    out += std::to_string(ae);
  }
  return out;
}

}  // namespace inkling::json
