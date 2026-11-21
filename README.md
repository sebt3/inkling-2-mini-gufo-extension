# inkling-2-mini: Gufo extension

[Gufo](https://github.com/gufo-org/gufo) text-generation extension for Thinking Machines' **Inkling-2-Mini**
(26B total, 3.4B active, looped MoE, GGUF architecture `inkling2_loop`).

Planned scope: correct 48-slot KV cache (40 sliding-window-1024 slots, 8 global slots), vision, the DSpark
speculative drafter, and continuous batching, on AMD gfx1151 (HIP).

**Status: work in progress, nothing runs yet.** See [PLAN.md](PLAN.md) for the design, the milestones and the
validation gates, and [docs/GGUF-LAYOUT.md](docs/GGUF-LAYOUT.md) for the exact tensor layout the extension reads.

* Target: Gufo 0.33.0 (extension API 0.2.21), Linux.
* Weights: `unsloth/Inkling-2-Mini-GGUF`, `Mainline_Support/` layout (`general.architecture = inkling2_loop`).
* Oracle: llama.cpp with the `inkling2_loop` support and the NVFP4 fix (PR #32694 + #32704).

`tools/gguf_dump.py` prints the metadata and tensor table of a GGUF from its first bytes.

License: MIT (see LICENSE).
