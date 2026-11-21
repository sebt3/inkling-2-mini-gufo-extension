# Third-party notices

## llama.cpp (MIT, The ggml authors)

`host/tokenizer.cpp` and `host/chat_template.cpp` reproduce the behaviour of llama.cpp (b12617, commit 44c20b1c0)
that the oracle relies on, and reuse its pieces of logic:

* the GPT4O pre-tokenizer pattern of `src/llama-vocab.cpp` (the `inkling2` pre-tokenizer uses it), applied on the
  collapsed-category text of `src/unicode.cpp` (`unicode_regex_split`, `unicode_regex_split_stl`);
* the special-token partition (`tokenizer_st_partition`) and the BPE merge loop of `llm_tokenizer_bpe_session`;
* the continuation of a trailing assistant message (`common/parsers/inkling.cpp`, `common/chat.cpp`) and the
  server rules of `tools/server/server-common.cpp` that the rendering follows.

The MIT permission notice of llama.cpp applies to these parts.

## ICU (ICU License, Unicode, Inc.)

`host/tokenizer.cpp` takes the Unicode general category and the White_Space property of a code point from ICU
(`u_charType`, `u_hasBinaryProperty`). The build links `libicuuc`.

## Reference-only sources

`/home/coder/projets/external/flint-gufo-extension` (MIT) was only listed while looking for a template renderer
example. Nothing from it is used in this milestone.
