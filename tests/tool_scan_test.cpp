// Tests of host/tool_scan: explicit cases (expected results written by hand from the call format), then two
// property checks on generated calls: a round trip (write a call, read it back) and a prefix check (every cut of a
// valid text, read in streaming mode, must agree with the full read).
//   tool_scan_test        exit status 0 only when every check passes
#include <cstdio>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../host/json.hpp"
#include "../host/tool_scan.hpp"

namespace {

const std::string H = "<|message_model|>";
const std::string I = "<|content_invoke_tool_json|>";
const std::string E = "<|end_message|>";

int checks = 0;
int failures = 0;

void fail(const std::string& where, const std::string& what) {
  ++failures;
  std::printf("FAIL %s: %s\n", where.c_str(), what.c_str());
}

// the call as the template writes it: body is the members text (already in the output form)
std::string call_text(const std::string& name, const std::string& body) {
  return H + name + I + "{\"name\":\"" + name + "\",\"args\":{" + body + "}}" + E;
}

using Arg = std::tuple<std::string, std::string, bool>;  // name, value, is_string

struct Want {
  std::string name;
  std::vector<Arg> args;
  bool complete;
};

std::string describe(const std::vector<inkling::ToolCallOut>& calls) {
  std::string s;
  for (const auto& c : calls) {
    s += "[" + c.name + (c.complete ? "" : " INCOMPLETE");
    for (const auto& a : c.args) {
      s += " " + a.name + "=" + (a.is_string ? "\"" + a.value + "\"" : a.value);
    }
    s += "] ";
  }
  return s.empty() ? "(none)" : s;
}

void expect(const std::string& where, const std::string& text, bool partial, const std::vector<Want>& want) {
  ++checks;
  std::vector<inkling::ToolCallOut> calls;
  std::vector<std::pair<size_t, size_t>> spans;
  inkling::scan_tool_calls(text, &calls, &spans, partial);
  bool ok = calls.size() == want.size() && spans.size() == calls.size();
  for (size_t i = 0; ok && i < want.size(); ++i) {
    const auto& c = calls[i];
    const auto& w = want[i];
    ok = c.name == w.name && c.complete == w.complete && c.args.size() == w.args.size();
    for (size_t k = 0; ok && k < w.args.size(); ++k) {
      ok = c.args[k].name == std::get<0>(w.args[k]) && c.args[k].value == std::get<1>(w.args[k]) &&
           c.args[k].is_string == std::get<2>(w.args[k]);
    }
  }
  if (!ok) {
    std::string wants;
    for (const auto& w : want) {
      wants += "[" + w.name + (w.complete ? "" : " INCOMPLETE");
      for (const auto& a : w.args) wants += " " + std::get<0>(a) + "=\"" + std::get<1>(a) + "\"";
      wants += "] ";
    }
    fail(where, "got " + describe(calls) + " want " + (wants.empty() ? "(none)" : wants));
    return;
  }
  // every span starts at a model header
  for (size_t i = 0; i < spans.size(); ++i) {
    if (spans[i].second > text.size() || spans[i].first >= spans[i].second ||
        text.compare(spans[i].first, H.size(), H) != 0) {
      fail(where, "bad span " + std::to_string(i));
      return;
    }
  }
}

void explicit_cases() {
  // one call
  expect("ping no args", call_text("ping", ""), false, {{"ping", {}, true}});
  expect("one string", call_text("get_weather", "\"city\":\"Paris\""), false,
         {{"get_weather", {{"city", "Paris", true}}, true}});
  expect("number arg raw", call_text("set", "\"n\":12"), false, {{"set", {{"n", "12", false}}, true}});
  expect("arg order kept", call_text("f", "\"a\":1,\"b\":\"x\""), false,
         {{"f", {{"a", "1", false}, {"b", "x", true}}, true}});
  expect("empty string arg", call_text("f", "\"s\":\"\""), false, {{"f", {{"s", "", true}}, true}});
  // several calls and text around
  const std::string c1 = call_text("a", "\"x\":1");
  const std::string c2 = call_text("b", "\"y\":\"z\"");
  expect("two calls", c1 + c2, false, {{"a", {{"x", "1", false}}, true}, {"b", {{"y", "z", true}}, true}});
  expect("text around", "Let me check. " + c1 + " Done.", false, {{"a", {{"x", "1", false}}, true}});
  expect("text block before", "<|content_text|>hi<|end_message|>" + c1, false, {{"a", {{"x", "1", false}}, true}});
  expect("thinking block before", "<|content_thinking|>plan<|end_message|>" + c1, false,
         {{"a", {{"x", "1", false}}, true}});
  expect("header of a text block only", "<|message_model|><|content_text|>hi<|end_message|>", false, {});
  expect("no markup", "hello world, no call here", false, {});
  expect("empty text", "", false, {});
  // JSON inside strings
  expect("braces in string", call_text("code", "\"c\":\"fn main() { if x { } }\""), false,
         {{"code", {{"c", "fn main() { if x { } }", true}}, true}});
  expect("escaped quotes", call_text("say", "\"q\":\"say \\\"hi\\\"\""), false,
         {{"say", {{"q", "say \"hi\"", true}}, true}});
  expect("slash and backslash", call_text("p", "\"p\":\"a\\/b\\\\c\""), false, {{"p", {{"p", "a/b\\c", true}}, true}});
  expect("unicode escapes", call_text("u", "\"u\":\"\\u00e9t\\u00e9\""), false,
         {{"u", {{"u", "\xc3\xa9t\xc3\xa9", true}}, true}});
  expect("surrogate pair", call_text("e", "\"e\":\"\\ud83d\\ude00\""), false,
         {{"e", {{"e", "\xf0\x9f\x98\x80", true}}, true}});
  expect("lone high surrogate", call_text("e", "\"e\":\"\\ud83dx\""), false,
         {{"e", {{"e", "\xef\xbf\xbdx", true}}, true}});
  expect("raw utf8 kept", call_text("e", "\"e\":\"\xc3\xa9\xf0\x9f\x98\x80\""), false,
         {{"e", {{"e", "\xc3\xa9\xf0\x9f\x98\x80", true}}, true}});
  expect("end marker text in a string", call_text("e", "\"e\":\"<|end_message|>\""), false,
         {{"e", {{"e", "<|end_message|>", true}}, true}});
  // non-string values: raw JSON text
  expect("object arg", call_text("trip", "\"legs\":{\"from\":\"A\",\"to\":[\"B\",\"C\"]}"), false,
         {{"trip", {{"legs", "{\"from\":\"A\",\"to\":[\"B\",\"C\"]}", false}}, true}});
  expect("array with bracket in string", call_text("xs", "\"xs\":[1,\"]\",{\"a\":\"}\"}]"), false,
         {{"xs", {{"xs", "[1,\"]\",{\"a\":\"}\"}]", false}}, true}});
  expect("bool and null", call_text("f", "\"f\":false,\"n\":null,\"t\":true"), false,
         {{"f", {{"f", "false", false}, {"n", "null", false}, {"t", "true", false}}, true}});
  expect("float exponent", call_text("f", "\"x\":-1.5e3"), false, {{"f", {{"x", "-1.5e3", false}}, true}});
  expect("zero", call_text("f", "\"n\":0"), false, {{"f", {{"n", "0", false}}, true}});
  expect("string array", call_text("f", "\"tags\":[\"a\",\"b\"]"), false,
         {{"f", {{"tags", "[\"a\",\"b\"]", false}}, true}});
  expect("empty object arg", call_text("f", "\"o\":{}"), false, {{"f", {{"o", "{}", false}}, true}});
  // names
  expect("unicode name", call_text("m\xc3\xa9t\xc3\xa9o", "\"city\":\"Lyon\""), false,
         {{"m\xc3\xa9t\xc3\xa9o", {{"city", "Lyon", true}}, true}});
  expect("header and json names differ", "<|message_model|>a<|content_invoke_tool_json|>{\"name\":\"b\",\"args\":{}}<|end_message|>",
         false, {{"a", {}, true}});
  expect("empty header name", "<|message_model|><|content_invoke_tool_json|>{\"name\":\"\",\"args\":{}}<|end_message|>",
         false, {});
  // malformed calls are skipped
  expect("bad json value", call_text("f", "\"x\":{\"a\":}"), false, {});
  expect("bad escape", call_text("f", "\"s\":\"\\q\""), false, {});
  expect("raw control char", "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"s\":\"a\nb\"}}<|end_message|>",
         false, {});
  expect("spaces between tokens (not the template form)", "<|message_model|>f<|content_invoke_tool_json|>{\"name\": \"f\"}<|end_message|>",
         false, {});
  expect("malformed then valid", "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"x\":}}<|end_message|>" +
         call_text("g", "\"y\":2"), false, {{"g", {{"y", "2", false}}, true}});
  // closing without the end marker
  expect("no end marker, final", "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"x\":1}}",
         false, {{"f", {{"x", "1", false}}, true}});
  expect("no end marker, other text follows", "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"x\":1}}<|content_model_end_sampling|>",
         true, {{"f", {{"x", "1", false}}, true}});
  // streaming: cut anywhere
  const std::string full = call_text("get_weather", "\"city\":\"Paris\",\"unit\":\"c\"");
  expect("prefix: cut inside a string", full.substr(0, full.find("Pa") + 2), true,
         {{"get_weather", {{"city", "Pa", true}}, false}});
  expect("prefix: cut after the opening brace", H + "get" + I + "{\"name\":\"get\",\"args\":{", true,
         {{"get", {}, false}});
  expect("prefix: cut after a comma", H + "get" + I + "{\"name\":\"get\",\"args\":{\"a\":1,", true,
         {{"get", {{"a", "1", false}}, false}});
  expect("prefix: cut inside a key", H + "get" + I + "{\"name\":\"get\",\"args\":{\"a\":1,\"b", true,
         {{"get", {{"a", "1", false}}, false}});
  expect("prefix: cut inside an escape", H + "get" + I + "{\"name\":\"get\",\"args\":{\"s\":\"ab\\u00", true,
         {{"get", {{"s", "ab", true}}, false}});
  expect("prefix: cut after a high surrogate", H + "get" + I + "{\"name\":\"get\",\"args\":{\"s\":\"\\ud83d", true,
         {{"get", {{"s", "", true}}, false}});
  expect("prefix: cut inside a raw value", H + "get" + I + "{\"name\":\"get\",\"args\":{\"o\":{\"a\":[1,2", true,
         {{"get", {}, false}});
  expect("prefix: cut in the name", H + "ge", true, {});
  expect("prefix: cut at the name end", H + "get", true, {});
  expect("prefix: cut in the invoke marker", H + "get<|content_inv", true, {});
  expect("prefix: cut in the header", "text <|message_mo", true, {});
  expect("prefix: braces closed, no marker, streaming", H + "get" + I + "{\"name\":\"get\",\"args\":{\"a\":1}}", true,
         {{"get", {{"a", "1", false}}, false}});
  expect("prefix: braces closed, no marker, final", H + "get" + I + "{\"name\":\"get\",\"args\":{\"a\":1}}", false,
         {{"get", {{"a", "1", false}}, true}});
  expect("prefix: cut in the end marker", call_text("f", "\"a\":1").substr(0, call_text("f", "\"a\":1").size() - 4), true,
         {{"f", {{"a", "1", false}}, true}});
  expect("final text cut inside a string", H + "get" + I + "{\"name\":\"get\",\"args\":{\"s\":\"ab", false,
         {{"get", {{"s", "ab", true}}, false}});
  expect("complete then cut call", call_text("a", "\"x\":1") + H + "b" + I + "{\"name\":\"b\",\"args\":{\"y\":\"z", true,
         {{"a", {{"x", "1", false}}, true}, {"b", {{"y", "z", true}}, false}});
  // several calls, the last one unfinished
  expect("two calls, second cut", call_text("a", "\"x\":1") + H + "b" + I + "{\"name\":\"b\",\"args\":{\"y\":2", true,
         {{"a", {{"x", "1", false}}, true}, {"b", {}, false}});
  // many arguments
  std::string many;
  std::vector<Arg> many_want;
  for (int k = 0; k < 20; ++k) {
    if (k) many += ",";
    many += "\"k" + std::to_string(k) + "\":" + std::to_string(k * 7);
    many_want.emplace_back("k" + std::to_string(k), std::to_string(k * 7), false);
  }
  expect("twenty arguments", call_text("big", many), false, {{"big", many_want, true}});
  // the text after the last call is ignored; a call with no end marker in the middle of text is complete
  expect("trailing noise", call_text("a", "\"x\":1") + " \n<|content_model_end_sampling|>", false,
         {{"a", {{"x", "1", false}}, true}});
}

// ---- property checks -------------------------------------------------------------------------------

std::mt19937 rng(20261121);

std::string utf8(uint32_t c) {
  std::string out;
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
  return out;
}

uint32_t random_cp() {
  static const uint32_t pool[] = {0x41, 0x7a, 0x20, 0x22, 0x5c, 0x2f, 0x0a, 0x09, 0x01, 0x1f, 0x7f, 0xe9,
                                  0x4e2d, 0x2028, 0x1f600, 0x10ffff, 0x3c, 0x7b, 0x7d, 0x5d};
  return pool[rng() % (sizeof pool / sizeof pool[0])];
}

// a JSON string literal for codepoints `cps`, with random escapes; the decoded text goes to `decoded`
std::string json_string_text(const std::vector<uint32_t>& cps, std::string& decoded) {
  std::string out = "\"";
  decoded.clear();
  for (uint32_t c : cps) {
    decoded += utf8(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += static_cast<char>(c);
    } else if (c < 0x20) {
      static const char* short_forms[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                          "\\b", "\\t", "\\n", nullptr, nullptr, "\\r"};
      if (c < 14 && short_forms[c] && rng() % 2) {
        out += short_forms[c];
      } else {
        char buf[8];
        std::snprintf(buf, sizeof buf, "\\u%04x", c);
        out += buf;
      }
    } else if (c == '/' && rng() % 2) {
      out += "\\/";
    } else if (c >= 0x80) {
      if (rng() % 2) {
        out += utf8(c);
      } else if (c < 0x10000) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "\\u%04x", c);
        out += buf;
      } else {
        const uint32_t v = c - 0x10000;
        char buf[16];
        std::snprintf(buf, sizeof buf, "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
        out += buf;
      }
    } else {
      out += static_cast<char>(c);
    }
  }
  return out + "\"";
}

std::string random_string(std::string& decoded) {
  std::vector<uint32_t> cps;
  const size_t n = rng() % 9;
  for (size_t i = 0; i < n; ++i) cps.push_back(random_cp());
  return json_string_text(cps, decoded);
}

// a random non-string JSON value: raw text (compact) and nothing else needed
std::string random_raw(int depth) {
  const int kind = static_cast<int>(rng() % (depth > 2 ? 4 : 7));
  switch (kind) {
    case 0: return std::to_string(static_cast<int>(rng() % 2001) - 1000);
    case 1: return "-0.25e-3";
    case 2: return (rng() % 2) ? "true" : "false";
    case 3: return "null";
    case 4: {
      std::string out = "[";
      const size_t n = rng() % 3;
      for (size_t i = 0; i < n; ++i) {
        if (i) out += ",";
        out += random_raw(depth + 1);
      }
      return out + "]";
    }
    default: {
      std::string out = "{";
      const size_t n = rng() % 3;
      for (size_t i = 0; i < n; ++i) {
        if (i) out += ",";
        std::string d;
        out += random_string(d) + ":" + random_raw(depth + 1);
      }
      return out + "}";
    }
  }
}

void round_trip_checks(int rounds) {
  for (int r = 0; r < rounds; ++r) {
    std::string text;
    std::vector<Want> want;
    const int ncalls = 1 + static_cast<int>(rng() % 3);
    for (int c = 0; c < ncalls; ++c) {
      std::string noise;
      const size_t nn = rng() % 12;
      for (size_t i = 0; i < nn; ++i) noise += "ab .,\n"[rng() % 6];
      text += noise;
      // names are written raw in the header and in the JSON: take them from a pool of plain names
      static const char* const names[] = {"get", "set_time", "m\xc3\xa9t\xc3\xa9o", "a-b", "x.y"};
      const std::string name = names[rng() % 5];
      std::string body;
      std::vector<Arg> args;
      const size_t na = rng() % 4;
      for (size_t k = 0; k < na; ++k) {
        std::string kd;
        const std::string key_text = random_string(kd);
        const bool is_string = rng() % 2;
        std::string value_text;
        std::string value_decoded;
        if (is_string) {
          value_text = random_string(value_decoded);
        } else {
          value_text = random_raw(0);
          value_decoded = value_text;
        }
        if (k) body += ",";
        body += key_text + ":" + value_text;
        args.emplace_back(kd, value_decoded, is_string);
      }
      text += H + name + I + "{\"name\":\"" + name + "\",\"args\":{" + body + "}}" + E;
      want.push_back({name, args, true});
    }

    std::vector<inkling::ToolCallOut> calls;
    std::vector<std::pair<size_t, size_t>> spans;
    inkling::scan_tool_calls(text, &calls, &spans, false);
    ++checks;
    bool ok = calls.size() == want.size();
    for (size_t i = 0; ok && i < want.size(); ++i) {
      ok = calls[i].name == want[i].name && calls[i].complete && calls[i].args.size() == want[i].args.size();
      for (size_t k = 0; ok && k < want[i].args.size(); ++k) {
        ok = calls[i].args[k].name == std::get<0>(want[i].args[k]) &&
             calls[i].args[k].value == std::get<1>(want[i].args[k]) &&
             calls[i].args[k].is_string == std::get<2>(want[i].args[k]);
      }
    }
    if (!ok) {
      fail("round trip #" + std::to_string(r), "text " + text + " got " + describe(calls));
      return;
    }
  }
}

void prefix_checks(int rounds) {
  for (int r = 0; r < rounds; ++r) {
    // one or two calls with string and raw values, with noise between them
    std::string text;
    for (int c = 0; c < 1 + static_cast<int>(rng() % 2); ++c) {
      std::string d;
      std::string body = "\"v\":" + random_string(d) + ",\"w\":" + random_raw(0);
      text += "x y " + call_text("tool" + std::to_string(c), body);
    }
    std::vector<inkling::ToolCallOut> full;
    inkling::scan_tool_calls(text, &full, nullptr, false);
    for (size_t cut = 0; cut <= text.size(); ++cut) {
      ++checks;
      const std::string prefix = text.substr(0, cut);
      std::vector<inkling::ToolCallOut> part;
      std::vector<std::pair<size_t, size_t>> spans;
      inkling::scan_tool_calls(prefix, &part, &spans, true);
      if (part.size() > full.size() || spans.size() != part.size()) {
        fail("prefix #" + std::to_string(r), "cut " + std::to_string(cut) + ": too many calls");
        return;
      }
      for (size_t i = 0; i < part.size(); ++i) {
        const bool last = i + 1 == part.size();
        if (part[i].name != full[i].name) {
          fail("prefix #" + std::to_string(r), "cut " + std::to_string(cut) + ": name differs");
          return;
        }
        if (!last || part[i].complete) {
          if (!part[i].complete || part[i].args.size() != full[i].args.size()) {
            fail("prefix #" + std::to_string(r), "cut " + std::to_string(cut) + ": finished call differs");
            return;
          }
        }
        // the readable arguments are a prefix of the full ones; only the last may be a string cut short
        for (size_t k = 0; k < part[i].args.size(); ++k) {
          if (k >= full[i].args.size()) {
            fail("prefix #" + std::to_string(r), "cut " + std::to_string(cut) + ": extra argument");
            return;
          }
          const auto& pa = part[i].args[k];
          const auto& fa = full[i].args[k];
          bool same_arg = pa.name == fa.name && pa.is_string == fa.is_string;
          if (!same_arg) {
            fail("prefix #" + std::to_string(r), "cut " + std::to_string(cut) + ": argument differs");
            return;
          }
          const bool exact = pa.value == fa.value;
          const bool string_prefix = pa.is_string && pa.value.size() <= fa.value.size() &&
                                     fa.value.compare(0, pa.value.size(), pa.value) == 0;
          if (!(exact || (string_prefix && last && !part[i].complete))) {
            fail("prefix #" + std::to_string(r), "cut " + std::to_string(cut) + ": value differs for " + pa.name);
            return;
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  explicit_cases();
  const size_t explicit_count = static_cast<size_t>(checks);
  round_trip_checks(3000);
  prefix_checks(150);
  std::printf("tool_scan_test: %d checks (%zu explicit cases), %d failures\n", checks, explicit_count, failures);
  return failures == 0 ? 0 : 1;
}
