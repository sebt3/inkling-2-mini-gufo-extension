# Progress log (working notes, newest last)

## 2026-11-21 evening
* Repo, plan, GGUF layout, `tools/gguf_dump.py` pushed. Oracle (lemonade b1384, build 12617) validated: PPL 2.55 on
  a 51 KB corpus, coherent French output.
* Dev loop: `tools/dev.sh sync|run|build-ref` (tar over ssh to worker04, build in the `inkling-dev` pod with the
  nix toolchain, see `tools/env.sh`). HIP hello-world runs on gfx1151 (`hipcc` flags in `tools/env.sh`).
* `core/gguf.hpp`, `core/quant.hpp` (Q8_0, Q5_K, NVFP4, BF16, F16), `ref/ref_model.hpp` (CPU F32 forward, 0.26 s/token)
  and `tools/inkling_ref.cpp` written. First run on 24 tokens: **wrong** (predictions copy the input), so a bug in
  the port remains. Debugging with `llama-eval-callback` built CPU-only from the oracle sources (`/src/llama.cpp`
  on worker04 = /var/tmp/inkling-ext/llama.cpp, build dir `/tmp/llbuild` in the pod) to compare per-layer tensors.
* Oracle top-20 data: `gates/oracle_topk.py` (run on infra-m1, python3 there) against `http://<oracle pod IP>:8001`.
* gufo prod deployment is scaled to 0 (replicas were 1) while the oracle and dev pods hold the GPU.

## M1 host tier (tokenizer, chat template)
* `host/tokenizer.{hpp,cpp}`: byte-level BPE from the GGUF, GPT4O-style pre-tokenizer as in llama.cpp (collapsed
  categories, ICU for the general category), special-token partition. G1: 340/340 identical to the oracle, decode
  round trip included (`gates/tokenizer_gate.py`).
* `host/chat_template.{hpp,cpp}`: render_prompt, branch by branch, with the server's prefill of a trailing assistant
  message. G2: 113/113 identical to the oracle (`gates/template_gate.py`). Variables and quirks in
  `docs/CHAT-TEMPLATE.md`. Tools and images are not rendered yet.
* Known limit: the pre-tokenizer uses std::regex as llama.cpp does, which is roughly quadratic on long runs of a
  single character class (16k letters: about 2.5 s; runs of 100k+ did not finish within minutes). Real text is not
  affected; a base64 blob of tens of kilobytes would be. A linear hand-written matcher is the fix; the oracle has the
  same cost.
* Run both gates: `tools/m1_gates.sh` (the oracle pod must be up on infra-m1).

## 2026-11-21 night: Gufo integration works
* M1 merged (tokenizer 340/340, chat template 113/113 against the oracle; re-run and confirmed).
* `gufo extension install` builds the module (host/HIP/text tiers, ICU linked); `gufo serve llm -m <Q4_K_M gguf>` loads in
  2.2 s and answers `/v1/chat/completions`: greedy output (reasoning + content) is identical to the oracle's on a first
  prompt; 26.7 tok/s decode with the unoptimised kernels; the component cache captures checkpoints (K/V row components +
  private conv/logits state).
* Not yet: tools, vision, DSpark, batching (`DecodeBatch`), long-context validation, kernel optimisation, package.

## 2026-11-21 evening: tools, batching validated, state of the numbers
* M9 tool calls merged (template + scanner, 130/130 against the oracle's `/apply-template`, scanner 33.5k checks) and wired into Gufo:
  tool dialect `inkling`, prompt tools/tool_calls/tool messages, context-aware markup (`InklingTextRunner::Piece`), streaming, reasoning
  plus calls, tool-result round trip answering like the oracle. Details and the grammar limitation: `docs/TOOLS.md`.
* Batching: 4 concurrent requests byte-identical to the sequential run; 56 tok/s aggregate vs 30 tok/s.
* Decode: ~29-30 tok/s (oracle ~25), 26.5 at 20k depth; prefill ~650 tok/s. Wave-shuffle reductions in rmsnorm/mix/control gate.
* DSpark: lossless and functional, but acceptance is 10-30 % and speculative decoding is slower than plain decoding (17.6 vs 30 tok/s):
  verifying 8 tokens touches ~5x more experts, and the drafter head is unoptimised. Do not enable `--speculative` for speed yet.
  No reference acceptance rate: building the unmerged llama.cpp PR (#32218) to get one was not run (sandbox policy); the question stays open.
