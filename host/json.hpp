// Minimal JSON values for the host tier (no dependency on Gufo or llama.cpp).
// Objects keep their insertion order, like the llama.cpp server (ordered_json): the chat template relies on it
// for tool-call arguments, and sorts keys itself where it asks for it (tool declarations). Output follows
// nlohmann::json::dump: strings escape only quote, backslash, control characters; numbers keep integer
// lexemes and print floats in the shortest round-trip form with the exponent style of nlohmann.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inkling::json {

struct Value {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool b = false;
  double n = 0.0;
  std::string lexeme;  // numbers: the JSON text as written (integer lexemes are printed from it)
  std::string s;       // strings: decoded content
  std::vector<Value> a;
  std::vector<std::pair<std::string, Value>> o;  // object members, insertion order, unique keys

  static Value null() { return Value{}; }
  static Value boolean(bool v) { Value x; x.type = Type::Bool; x.b = v; return x; }
  static Value number(double v) { Value x; x.type = Type::Number; x.n = v; return x; }
  static Value string(std::string v) { Value x; x.type = Type::String; x.s = std::move(v); return x; }
  static Value array() { Value x; x.type = Type::Array; return x; }
  static Value object() { Value x; x.type = Type::Object; return x; }

  bool is_null() const { return type == Type::Null; }
  bool is_bool() const { return type == Type::Bool; }
  bool is_number() const { return type == Type::Number; }
  bool is_string() const { return type == Type::String; }
  bool is_array() const { return type == Type::Array; }
  bool is_object() const { return type == Type::Object; }

  // member lookup (objects only); nullptr when absent
  const Value* find(std::string_view key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& kv : o) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }

  // replaces the value of an existing key in place, or appends the member
  void set(std::string key, Value v) {
    for (auto& kv : o) {
      if (kv.first == key) {
        kv.second = std::move(v);
        return;
      }
    }
    o.emplace_back(std::move(key), std::move(v));
  }
};

// Whole-text parser. Throws std::invalid_argument with a "json: ..." message.
Value parse(std::string_view text);

// Python / Jinja truthiness as the template sees it: null, false, 0, "", [] and {} are false.
bool truthy(const Value& v);

// JSON string literal with quotes, escaped like nlohmann::json::dump (non-ASCII written as is).
std::string quote(std::string_view s);

// Compact serialization. sort_keys sorts object members by key (byte order, which is code point order for UTF-8).
void dump(std::string& out, const Value& v, bool sort_keys);

// Number text as nlohmann::json::dump prints it: integers from their lexeme, floats in shortest round-trip form,
// fixed notation for decimal exponents -4..15, otherwise d.ddde+XX (two-digit exponent).
std::string number_text(const Value& v);

}  // namespace inkling::json
