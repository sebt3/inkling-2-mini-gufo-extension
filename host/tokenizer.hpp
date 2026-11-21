// Byte-level BPE tokenizer for the Inkling-2 GGUF (tokenizer.ggml.model = gpt2, tokenizer.ggml.pre = inkling2).
// Behaviour follows llama.cpp (pre-tokenizer GPT4O branch, special-token partition, BPE merges by rank).
// No Gufo or llama.cpp dependency. Unicode general categories come from ICU (u_charType).
#pragma once
#include <array>
#include <cstdint>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/gguf.hpp"

namespace inkling {

class Tokenizer {
 public:
  explicit Tokenizer(const Gguf& gguf);

  // Token ids for text. parse_special=true: control and user-defined tokens written in the text
  // (e.g. "<|message_user|>") become their ids; parse_special=false: control tokens are plain text.
  std::vector<int> encode(const std::string& text, bool parse_special) const;

  // Inverse of encode: special tokens are printed as their text (llama.cpp detokenize with special=true).
  std::string decode(const std::vector<int>& ids) const;

  int size() const { return (int)tokens_.size(); }
  // -1 when the text is not a token
  int token_id(const std::string& text) const;

 private:
  enum TokenType : uint8_t { NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, UNUSED = 5, BYTE = 6 };

  std::vector<int> bpe_encode_fragment(const std::string& text) const;
  std::vector<std::string> pre_tokenize(const std::string& text) const;
  std::vector<int> bpe_word(const std::string& byte_encoded_word) const;
  int bpe_rank(const std::string& left, const std::string& right) const;

  std::vector<std::string> tokens_;
  std::vector<uint8_t> types_;
  std::unordered_map<std::string, int> text_to_id_;
  std::unordered_map<std::string, int> merge_rank_;  // key: "left right"
  std::vector<int> special_ids_;                      // control/user/unknown tokens, longest text first

  std::array<std::string, 256> byte_to_char_;          // GPT-2 byte -> UTF-8 of the mapped code point
  std::unordered_map<uint32_t, uint8_t> cpt_to_byte_;  // inverse mapping

  std::regex pre_regex_;  // GPT4O pattern in the collapsed-category form used by llama.cpp
};

}  // namespace inkling
