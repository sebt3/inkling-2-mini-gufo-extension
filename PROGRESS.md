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
