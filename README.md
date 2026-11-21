# inkling-2-mini: Gufo extension

[Gufo](https://github.com/gufo-org/gufo) text-generation extension for Thinking Machines' **Inkling-2-Mini**
(26B total, 3.4B active, looped MoE, GGUF architecture `inkling2_loop`), written for AMD gfx1151 (Strix Halo, HIP).

What it does:

* correct execution of the architecture, including the **48-slot KV cache** (40 sliding-window-1024 slots, 8 global slots, per-slot
  convolution state) and its integration with Gufo's continuation cache (prefix reuse, RAM/disk checkpoints);
* **vision** (ViT encoder + projector of the `mmproj` file, images in chat messages);
* **DSpark** speculative decoding (block drafter + Markov head, lossless verification);
* **continuous batching** (`--sessions N`, one token per session in a single engine pass);
* **tool calls** (auto mode, parallel calls, streaming, reasoning; no constrained calls yet): see `docs/TOOLS.md`.

Status and measurements are in [PLAN.md](PLAN.md) and [PROGRESS.md](PROGRESS.md); the design in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install and run

Requirements: Gufo 0.33.0 (extension API 0.2.21) with its extension SDK, Linux, an AMD gfx1151 GPU.

```sh
gufo extension install /path/to/inkling-2-mini-gufo-extension --yes        # builds the module (HIP, ICU)
gufo serve llm -m Inkling-2-Mini-UD-Q4_K_M.gguf \
    --mmproj Inkling-2-Mini-mmproj-BF16.gguf \                              # optional: images
    --speculative dspark --dspark-model Inkling-2-Mini-dspark-Q8_0.gguf \   # optional: DSpark
    -c 65536 --sessions 4
```

Weights: `unsloth/Inkling-2-Mini-GGUF`. Use the files of the **`Mainline_Support/`** folder (rename `*.gguf_file` to `.gguf`): that is
the `inkling2_loop` layout this extension reads (the root files use another tensor layout meant for Unsloth's own build). The
`mmproj-*.gguf` and the `dspark-*.gguf` drafter are at the root of the repository.

Sampling defaults come from the model card (temperature 1, no truncation). Reasoning arrives in `reasoning_content`
(`<|content_thinking|> ... <|end_message|>` is split by the host); the effort level maps to the template's `effort_map`.

## Layout

| Path | Content |
|------|---------|
| `engine/` | HIP engine: `kernels.hpp` (scalar kernels), `wmma.hpp` (matrix-core GEMM, grouped MoE), `flash.hpp` (prefill attention), `decode.hpp` (fused single-token kernels), `vision.hpp`, `dspark.hpp`, `engine.hpp`, C++ facade `engine_api.*` |
| `gufo/` | Gufo adapter: runner, session, continuation-cache adapter, provider, probe |
| `host/` | tokenizer and chat template (no Gufo or llama.cpp dependency) |
| `core/`, `ref/` | standalone GGUF reader, quantization decoders, **CPU reference forward** |
| `tools/` | test harnesses (`inkling_ref`, `inkling_gpu`, `inkling_vision`, `inkling_spec`, `oracle_dump`, benchmarks) and `dev.sh` |
| `gates/` | validation scripts against the llama.cpp oracle |

## Validation

Every stage is checked against llama.cpp (lemonade build b1384, which contains the NVFP4 fix of PR #32704) through the scripts of
`gates/`: tokenizer and chat template (G1, G2: 340/340 and 113/113 identical), CPU reference vs oracle layer by layer (<= 1.1 % with
the oracle's expert choice forced), GPU engine vs CPU reference (<= 7e-4 on all 48 slots), vision encoder vs `llama-mtmd-debug`, and end to
end chat comparisons (`gates/serve_cmp.py`, `gates/vision_cmp.py`).

License: MIT (see LICENSE and THIRD_PARTY_NOTICES.md).
