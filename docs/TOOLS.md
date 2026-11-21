# Tool calls (M9)

Scope: rendering of tools in the prompt (`host/chat_template.cpp`), and reading the calls in generated text
(`host/tool_scan.cpp`). Neither depends on Gufo or llama.cpp. Gate G9a compares the rendering with the oracle.

## Format of a call

From the template (`tokenizer.chat_template`, see `docs/CHAT-TEMPLATE.md`) and the parser of llama.cpp b12617
(`common/parsers/inkling.cpp`, `common_chat_params_init_inkling`):

```
<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{"KEY":VALUE,...}}<|end_message|>
```

* The header is `<|message_model|>` followed directly by the tool name, then the invoke marker.
* The JSON is compact. `name` is the same name again, JSON-escaped. `args` is an object.
* Each call is its own block ending with `<|end_message|>`. A turn may hold several calls, one after another.
  The grammar of the server allows several calls only at the end of the turn (`zero_or_more(header + tool_choice)`).
* Rendering: the tool name is written raw in the header and JSON-escaped in `name`. Arguments are sorted by key at
  top level only (`args | dictsort`); nested values keep their order. Numbers and strings use the JSON escapes
  described below.

Tool declarations: a single system block at the top of the prompt, before the effort line and the messages:

```
<|message_system|>tool_declare<|content_xml|>[{"description":...,"name":...,"parameters":{...},"type":"function"},...]<|end_message|>
```

The JSON array is compact with every object's keys sorted (nested too).

## What the oracle does

Rendering (verified by G9a against `/apply-template` of the oracle server):

* `tools` accepted as in the OpenAI shape. A tool without `type: "function"`, without `function`, with a non-string
  `name`, or with a non-string `description` is rejected (server error). `description` and `parameters` default to
  `""` and `{}`; falsy `parameters` (`null`, `false`, `[]`) render as `{}`.
* Tool-call `arguments` given as a JSON string are parsed by the server first, then the template applies its checks.
  A string that is not JSON is rejected; a JSON string after the parse is rejected; a non-object (array, number) is
  rejected; a falsy value renders `{}`.
* A tool message with `name` uses it; without `name` it uses the `name` of the call with the same `tool_call_id`.
* A trailing assistant message with tool calls is rejected when prefilled (the server cannot continue it).

Parsing of generated text (from the source, not measured live: the oracle was not asked to generate, so no GPU
time was spent):

* The grammar lets a call start on a header followed by a character other than `<`, and only for the tools
  declared in the request. Calls are the last blocks of the turn; `tool_choice` and `parallel_tool_calls` apply.
* The server turns each call into a `tool_calls` entry (name, and `arguments` as a JSON string).
* Thinking blocks (`<|content_thinking|>` ... `<|end_message|>`) are extracted as reasoning when the request asks for
  it; they never produce calls.

## Decisions

Rendering:

* Assistant reasoning is rendered before the content, as in the template; the content (or one block per part) comes
  next, then the calls, then the sampling marker. A message with tool calls and empty content renders no text block.
* A tool call's arguments are validated where the template validates them: missing (`nullopt`) is an error, string
  arguments are JSON text, falsy means no arguments.
* The lookup of a tool message's name by `tool_call_id` follows the template: every call with that id contributes
  its name (a duplicated id gives a concatenated header; the oracle does the same).
* `Options::tools` empty: the prompt is byte-identical to the prompt without the field (G2 unchanged, 113/113).

Number and string output (`host/json.cpp`, follows `nlohmann::json::dump`):

* Integers print from their lexeme (`-0` prints `0`). Floats print in shortest round-trip form; fixed notation for
  decimal exponents from -4 to 15, otherwise `d.ddde+XX` / `d.ddde-XX` with two exponent digits (`1e-05`, `1e+16`).
* Strings escape `"`, `\`, control characters (`\b \f \n \r \t`, others `\u00xx`). `/` and non-ASCII are written as is.
* Objects keep insertion order, except where the template sorts (declarations, top-level arguments).

Scanner (`host/tool_scan.cpp`):

* `scan_tool_calls(text, calls, spans, partial)`: `calls[i]` and `spans[i]` are parallel. A span runs from the model
  header to the end marker; without the marker, to the closing braces; for an unfinished call, to the end of text.
* `partial = true` (streaming): a header whose name is not closed, or a marker cut by the end of the text, is ignored.
  An unfinished call is reported last with `complete = false`. Its arguments are the ones already complete, plus a
  string argument cut short (its value is the decoded prefix, which the caller must not treat as final).
  A number or literal at the very end is not reported yet (it could still grow).
* `partial = false` (final text): a call whose closing braces are present is complete, even without the end marker.
* Malformed calls (invalid JSON value, bad escape, raw control character in a string, header name missing) are
  skipped: no element, no span. The scan then continues after the header.
* `\uXXXX` escapes decode, surrogate pairs combine; a lone surrogate decodes to U+FFFD.
* The name is taken from the header. Names are not checked against the declared tools (the template does not check
  them either, the grammar of the server does).

## Limits

* The scanner expects the compact form the template writes. A call with spaces between its JSON tokens is skipped.
* A `<|message_model|>` followed by a tool name, typed inside a text block, is read as a call header: the scanner
  cannot tell it apart from a real one. A header followed by `<|content_text|>` or `<|content_thinking|>` is not a
  call, so ordinary blocks do not trigger this.
* An assistant message with `tool_calls: []` and no `content` key is rejected by the renderer; the server accepts it
  (the message type cannot tell an empty list from a missing key).
* Numbers beyond the int64 range print in float form. The oracle prints the unsigned maximum `18446744073709551615`
  as `-1` (a server quirk), which is not reproduced.
* JSON text of a tool argument with a lone surrogate is accepted (U+FFFD); the server rejects it.
* The server's `reasoning_format` and its handling of thinking inside the response are not reproduced: the scanner
  simply reads no calls from thinking blocks.

## Results

* G9a (`gates/tools_gate.py`): 130 of 130 checked cases identical to the oracle, 20 of them rejected on both sides.
  The corpus has 131 conversations: hand-written cases (one or several tools, nested declarations, calls with and
  without content, reasoning, parts, several calls, tool messages by name and by id, argument edge cases, errors)
  and 60 seeded random conversations. One case has `prefill_assistant = false` and is not compared.
* Scanner (`tests/tool_scan_test.cpp`): 56 explicit cases with hand-written expectations, 3000 random round trips
  (write a call, read it back), and a prefix sweep (every cut of 150 texts, streamed, checked against the full read):
  33524 checks, 0 failures.
* G2 (`gates/template_gate.py`): still 113 of 113.

## Reproduce

```
cd <worktree> && TREE=m9 tools/m9_gates.sh          # G9a, G2 and the scanner tests
TREE=m9 tools/m9_gates.sh tools                     # G9a only
TREE=m9 tools/m9_gates.sh scan                      # scanner tests only
```

The C++ runs in the dev pod (`tools/dev.sh`); the oracle is queried from `infra-m1`.

## Gufo integration (M9b)

* `PreparePrompt` maps `ChatRequest` tools, assistant `tool_calls` and tool messages onto `host/chat_template`; the rendered prompt is
  byte-identical to the oracle's (checked on a tool round trip). `INKLING_LOG_PROMPT=1` prints each rendered prompt, `INKLING_LOG_TOKENS=1`
  each sampled token (debugging).
* A `ToolDialect` with id `inkling` (`gufo/provider.cpp`) wires `host/tool_scan` into the host's parsers (buffered and streamed). The
  scanner also accepts the **bare** form `<|message_model|>NAME<|content_invoke_tool_json|>{ARGS}`: unconstrained, the model writes the
  arguments object directly, the `{"name":...,"args":...}` wrapper only appears under the oracle's grammar.
* Markup visibility (`InklingTextRunner::Piece`, a four-phase machine): answer headers (`<|message_model|><|content_text|>`) stay hidden;
  a thinking block opens with the whole `<|message_model|><|content_thinking|>` (this is also the reasoning opener declared to the host);
  a tool call keeps its entire markup, `<|message_model|>` being released with the first token of the call header.
* **No constrained calls.** The host's grammar for a call opened by `<|message_model|>` would also swallow every answer block, so the
  dialect declares a dummy opener the model never writes (generation stays free; the scanner starts on the real header).
  `tool_choice: required` and a forced tool are refused with a clear error; `auto` and `none` work, with reasoning, streaming and
  multiple calls. Constrained calls would need a grammar that can tell `<|message_model|>NAME` from `<|message_model|><|content_text|>`.
