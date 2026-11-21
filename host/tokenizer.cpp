#include "tokenizer.hpp"

#include <algorithm>
#include <stdexcept>

#include <unicode/uchar.h>

namespace inkling {
namespace {

// ---- UTF-8 helpers -------------------------------------------------------------------------------

std::vector<uint32_t> utf8_to_cpts(const std::string& s) {
  std::vector<uint32_t> out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (size_t k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
    out.push_back(cp);
    i += len;
  }
  return out;
}

std::string cpt_to_utf8(uint32_t c) {
  std::string out;
  if (c < 0x80) out += (char)c;
  else if (c < 0x800) { out += (char)(0xC0 | (c >> 6)); out += (char)(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { out += (char)(0xE0 | (c >> 12)); out += (char)(0x80 | ((c >> 6) & 0x3F)); out += (char)(0x80 | (c & 0x3F)); }
  else { out += (char)(0xF0 | (c >> 18)); out += (char)(0x80 | ((c >> 12) & 0x3F)); out += (char)(0x80 | ((c >> 6) & 0x3F)); out += (char)(0x80 | (c & 0x3F)); }
  return out;
}

// ---- Unicode classes for the pre-tokenizer -------------------------------------------------------
// llama.cpp "collapses" the text before applying the regex: every non-ASCII code point becomes one byte
// that stands for its general category (letter, number, punctuation, mark, symbol), non-ASCII white space
// becomes 0x0B (\s), any other non-ASCII code point becomes 0xD0. ASCII is kept as is.
constexpr char kCatLetter = '\322';   // 0xD2
constexpr char kCatNumber = '\321';   // 0xD1
constexpr char kCatPunct = '\323';    // 0xD3
constexpr char kCatMark = '\324';     // 0xD4
constexpr char kCatSymbol = '\325';   // 0xD5
constexpr char kCatOther = '\320';    // 0xD0
constexpr char kSpaceFallback = 0x0B;

char collapse(uint32_t c) {
  if (c < 0x80) return (char)c;
  if (u_hasBinaryProperty((UChar32)c, UCHAR_WHITE_SPACE)) return kSpaceFallback;
  switch (u_charType((UChar32)c)) {
    case U_UPPERCASE_LETTER: case U_LOWERCASE_LETTER: case U_TITLECASE_LETTER:
    case U_MODIFIER_LETTER: case U_OTHER_LETTER:
      return kCatLetter;
    case U_DECIMAL_DIGIT_NUMBER: case U_LETTER_NUMBER: case U_OTHER_NUMBER:
      return kCatNumber;
    case U_CONNECTOR_PUNCTUATION: case U_DASH_PUNCTUATION: case U_START_PUNCTUATION:
    case U_END_PUNCTUATION: case U_INITIAL_PUNCTUATION: case U_FINAL_PUNCTUATION: case U_OTHER_PUNCTUATION:
      return kCatPunct;
    case U_NON_SPACING_MARK: case U_ENCLOSING_MARK: case U_COMBINING_SPACING_MARK:
      return kCatMark;
    case U_MATH_SYMBOL: case U_CURRENCY_SYMBOL: case U_MODIFIER_SYMBOL: case U_OTHER_SYMBOL:
      return kCatSymbol;
    default:
      return kCatOther;
  }
}

// The GPT4O pattern of llama.cpp (src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_GPT4O, used by inkling2),
// with \p{L}, \p{N}, \p{M}, \p{S} replaced by their collapsed form, as unicode_regex_split does.
std::string collapsed_gpt4o_pattern() {
  const std::string L = std::string(1, kCatLetter) + "A-Za-z";
  const std::string N = std::string(1, kCatNumber) + "0-9";
  const std::string LM = L + std::string(1, kCatMark);
  const std::string contractions = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?";
  return "[^\\r\\n" + L + N + "]?"
         "((?=[" + LM + "])([^a-z]))*"
         "((?=[" + LM + "])([^A-Z]))+" + contractions + "|"
         "[^\\r\\n" + L + N + "]?"
         "((?=[" + LM + "])([^a-z]))+"
         "((?=[" + LM + "])([^A-Z]))*" + contractions + "|"
         "[" + N + "]{1,3}"
         "| ?[^\\s" + L + N + "]+[\\r\\n/]*"
         "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
}

// GPT-2 byte -> code point table (bytes_to_unicode)
uint32_t gpt2_code_point(int byte, int& extra) {
  if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255)) return (uint32_t)byte;
  return 256u + (uint32_t)extra++;
}

}  // namespace

Tokenizer::Tokenizer(const Gguf& gguf) {
  if (gguf.m("tokenizer.ggml.model").str != "gpt2") throw std::runtime_error("tokenizer: model is not gpt2 BPE");
  if (gguf.m("tokenizer.ggml.pre").str != "inkling2") throw std::runtime_error("tokenizer: pre-tokenizer is not inkling2");

  tokens_ = gguf.m("tokenizer.ggml.tokens").sarr;
  const auto& types = gguf.m("tokenizer.ggml.token_type").arr;
  types_.reserve(types.size());
  for (double t : types) types_.push_back((uint8_t)t);
  if (types_.size() != tokens_.size()) throw std::runtime_error("tokenizer: token_type size mismatch");

  for (int id = 0; id < (int)tokens_.size(); ++id) {
    text_to_id_[tokens_[id]] = id;  // llama.cpp keeps the last id for a duplicated text
    const uint8_t t = types_[id];
    if (t == CONTROL || t == USER_DEFINED || t == UNKNOWN) special_ids_.push_back(id);
  }
  std::stable_sort(special_ids_.begin(), special_ids_.end(),
                   [&](int a, int b) { return tokens_[a].size() > tokens_[b].size(); });

  const auto& merges = gguf.m("tokenizer.ggml.merges").sarr;
  for (int rank = 0; rank < (int)merges.size(); ++rank) merge_rank_.emplace(merges[rank], rank);  // first rank wins

  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    uint32_t cp = gpt2_code_point(b, extra);
    byte_to_char_[b] = cpt_to_utf8(cp);
    cpt_to_byte_[cp] = (uint8_t)b;
  }

  pre_regex_ = std::regex(collapsed_gpt4o_pattern(), std::regex_constants::ECMAScript | std::regex_constants::optimize |
                                                         std::regex_constants::nosubs);
}

int Tokenizer::token_id(const std::string& text) const {
  auto it = text_to_id_.find(text);
  return it == text_to_id_.end() ? -1 : it->second;
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
  const auto cpts = utf8_to_cpts(text);
  std::string collapsed(cpts.size(), '\0');
  for (size_t i = 0; i < cpts.size(); ++i) collapsed[i] = collapse(cpts[i]);

  // same loop as unicode_regex_split_stl over one segment: matches and the gaps between them become segments
  std::vector<size_t> offsets;
  int64_t start_idx = 0;
  for (auto it = std::sregex_iterator(collapsed.begin(), collapsed.end(), pre_regex_); it != std::sregex_iterator(); ++it) {
    const auto& m = *it;
    if (m.position() > start_idx) offsets.push_back((size_t)(m.position() - start_idx));
    offsets.push_back((size_t)m.length());
    start_idx = m.position() + m.length();
  }
  if (start_idx < (int64_t)cpts.size()) offsets.push_back((size_t)((int64_t)cpts.size() - start_idx));

  std::vector<std::string> words;
  size_t pos = 0;
  for (size_t len : offsets) {
    std::string word;
    for (size_t i = pos; i < pos + len; ++i) word += cpt_to_utf8(cpts[i]);
    words.push_back(std::move(word));
    pos += len;
  }
  return words;
}

int Tokenizer::bpe_rank(const std::string& left, const std::string& right) const {
  auto it = merge_rank_.find(left + " " + right);
  return it == merge_rank_.end() ? -1 : it->second;
}

std::vector<int> Tokenizer::bpe_word(const std::string& word) const {
  // one symbol per UTF-8 character of the byte-encoded word
  std::vector<std::string> sym;
  for (size_t i = 0; i < word.size();) {
    unsigned char c = word[i];
    size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    sym.push_back(word.substr(i, len));
    i += len;
  }

  // repeatedly merge the adjacent pair with the lowest rank (leftmost on ties)
  while (sym.size() > 1) {
    int best_rank = -1;
    size_t best = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      int r = bpe_rank(sym[i], sym[i + 1]);
      if (r >= 0 && (best_rank < 0 || r < best_rank)) {
        best_rank = r;
        best = i;
      }
    }
    if (best_rank < 0) break;
    sym[best] += sym[best + 1];
    sym.erase(sym.begin() + (long)best + 1);
  }

  std::vector<int> ids;
  for (const auto& s : sym) {
    int id = token_id(s);
    if (id >= 0) {
      ids.push_back(id);
      continue;
    }
    // not a token: fall back to the single-byte tokens of the symbol (as llama.cpp does)
    for (char b : s) {
      int bid = token_id(std::string(1, b));
      if (bid >= 0) ids.push_back(bid);
    }
  }
  return ids;
}

std::vector<int> Tokenizer::bpe_encode_fragment(const std::string& text) const {
  std::vector<int> ids;
  for (const auto& word : pre_tokenize(text)) {
    std::string encoded;
    encoded.reserve(word.size() * 2);
    for (unsigned char b : word) encoded += byte_to_char_[b];
    auto part = bpe_word(encoded);
    ids.insert(ids.end(), part.begin(), part.end());
  }
  return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
  struct Fragment {
    bool is_token;
    int id;
    std::string text;
  };
  std::vector<Fragment> frags;
  if (!text.empty()) frags.push_back({false, -1, text});

  // special tokens split the raw fragments, longest first (llama.cpp tokenizer_st_partition)
  for (int sid : special_ids_) {
    if (!parse_special && (types_[sid] == CONTROL || types_[sid] == UNKNOWN)) continue;
    const std::string& st = tokens_[sid];
    std::vector<Fragment> next;
    for (auto& f : frags) {
      if (f.is_token) {
        next.push_back(f);
        continue;
      }
      size_t pos = 0;
      while (true) {
        size_t m = f.text.find(st, pos);
        if (m == std::string::npos) {
          if (pos < f.text.size()) next.push_back({false, -1, f.text.substr(pos)});
          break;
        }
        if (m > pos) next.push_back({false, -1, f.text.substr(pos, m - pos)});
        next.push_back({true, sid, {}});
        pos = m + st.size();
      }
    }
    frags.swap(next);
  }

  std::vector<int> out;
  for (const auto& f : frags) {
    if (f.is_token) {
      out.push_back(f.id);
    } else {
      auto part = bpe_encode_fragment(f.text);
      out.insert(out.end(), part.begin(), part.end());
    }
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for (int id : ids) {
    if (id < 0 || id >= (int)tokens_.size()) throw std::out_of_range("tokenizer: bad token id " + std::to_string(id));
    const uint8_t t = types_[id];
    if (t == UNUSED) continue;
    if (t == NORMAL) {
      // byte-level text: map each code point back to its byte
      for (uint32_t cp : utf8_to_cpts(tokens_[id])) {
        auto it = cpt_to_byte_.find(cp);
        if (it != cpt_to_byte_.end()) out += (char)it->second;
        else out += cpt_to_utf8(cp);
      }
    } else {
      out += tokens_[id];  // control, user-defined and unknown tokens are printed as their text
    }
  }
  return out;
}

}  // namespace inkling
