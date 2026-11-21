# Chat template of Inkling-2-Mini

Source: `tokenizer.chat_template` of `Inkling-2-Mini-UD-Q4_K_M.gguf` (6867 characters, Jinja, no `add_generation_prompt`
branch). Reference rendering: llama.cpp b12617 (`44c20b1c0`), `/apply-template` of the oracle server. Implementation:
`host/chat_template.cpp`, gate `gates/template_gate.py` (G2, 113/113 identical). Tools: see `docs/TOOLS.md`
(gate G9a, `gates/tools_gate.py`).

## Variables the template reads

| Variable | Use |
|----------|-----|
| `messages` | the conversation: `role`, `content` (string, list of parts, or null), `reasoning_content`, `name`, `tool_calls`, `tool_call_id` |
| `reasoning_effort` | level name (`effort_map` key) or number; absent or null gives `0.9` |
| `tools` | tool declarations: `Options::tools`, rendered in the `tool_declare` block (M9) |
| `bos_token`, `eos_token`, `enable_thinking`, `add_generation_prompt` | not read by the template |

## Tokens

Role tokens: `user` `<|message_user|>`, `assistant` `<|message_model|>`, `system` `<|message_system|>`,
`tool` `<|message_tool|>`. Block tokens: `<|content_text|>`, `<|content_thinking|>`, `<|content_image|>`,
`<|content_invoke_tool_json|>`, every block ends with `<|end_message|>`. An assistant message ends with
`<|content_model_end_sampling|>` after its blocks.

## Effort

`effort_map`: none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99, max 0.99.

* A string is trimmed then looked up; an unknown string raises `Unknown reasoning_effort: <value>`.
* A number must be in [0.0, 0.99], otherwise raises. It is printed with `%f` and trailing zeros removed, `0` for zero:
  `0.123456789` gives `0.123457`, `1e-5` gives `0.00001`, `0.5` gives `0.5`.
* The rendered line is `<|message_system|><|content_text|>Thinking effort level: X<|end_message|>`.
* The line is emitted once, before the first non-system message. With no non-system message it is emitted at the end.

## Branches of the message loop

1. Role not in {system, user, assistant, tool} raises `Unknown message role`.
2. Assistant with `reasoning_content` non-empty: `<thinking block>`.
3. Text content (a null, absent or empty-array content is an empty string) is rendered as one block, unless the
   message is an assistant with reasoning and the text is empty.
4. List content: one block per part, each block with the same header. Only `text` parts are accepted.
5. Header of a tool message is `<|message_tool|>` + `name`, or, without `name`, the name of every earlier tool call with
   the same `tool_call_id`. A named assistant message is `<|message_model|>` + `name`; its name must equal the name of
   each of its tool calls.
6. Assistant `tool_calls` are rendered after the content blocks: `<|message_model|>NAME<|content_invoke_tool_json|>` +
   `{"name":"NAME","args":{...}}` + `<|end_message|>`, arguments sorted at top level. See `docs/TOOLS.md`.
7. Assistant messages end with the sampling marker.

## Server behaviour around the template (llama.cpp tools/server)

* Every message needs a `content` key. A missing key raises `All non-assistant messages must contain 'content'`,
  and for an assistant `Assistant message must contain either 'content' or 'tool_calls'!`.
* A trailing assistant message is a **prefill** (server default `prefill_assistant = true`). The template renders the
  messages before it, then the text of the prefill is appended with no closing marker:
  * reasoning: `<|message_model|><|content_thinking|>` + reasoning;
  * if there is both reasoning and text, `<|end_message|>` between them;
  * text: `<|message_model|><|content_text|>` + text. Typed parts are joined with `\n\n` (a separator is added before
    each part once some text exists, so an empty first part adds nothing);
  * a prefill with neither reasoning nor text adds nothing: the trailing message is dropped.
* Two assistant messages at the end raise `Cannot have 2 or more assistant messages at the end of the list.`
* The server maps `reasoning_effort: "none"` in the request body to "no effort" and removes the variable, so the template
  uses 0.9. Through `chat_template_kwargs` the string `"none"` reaches the template and gives `0`. `render_prompt`
  follows the template: `reasoning_effort = "none"` gives `0`. A caller that wants the server's body semantics must
  not pass `"none"`.

## Limits of this milestone

* An assistant message with `tool_calls: []` and no `content` key is rejected here; the server accepts it. The
  message type cannot tell an empty list from a missing key (the gate corpus does not contain this case).
* Images (`<|content_image|>` with `<|unused_200054|>`): rejected with an error, the multimodal path is M6.
* Audio is rejected by the template; the renderer rejects every non-text part.
* `prefill_assistant = false` is not compared with the oracle (the server always prefills).
* A trailing assistant message with tool calls is rejected when prefilled (server error, same text as the oracle).
