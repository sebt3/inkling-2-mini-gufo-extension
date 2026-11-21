# Inkling-2-Mini Gufo extension: plan

Goal: serve `thinkingmachines/Inkling-2-Mini` (GGUF architecture `inkling2_loop`) from Gufo on gfx1151
(Strix Halo), with a correct KV cache, vision, the DSpark drafter, and continuous batching.

Reference implementation (the **oracle**): llama.cpp `src/models/inkling2.cpp` + `inkling2-loop.cpp`
(PR #32694, with the NVFP4 scale fix #32704), run as `lemonade-sdk/llamacpp-rocm` **b1384**
(llama.cpp build 12617, commit `44c20b1c0`). b1383 is NOT a valid oracle: it predates #32704 and
produces garbage (perplexity ~130 instead of 7.65). DSpark is not in mainline; its oracle is the
Unsloth prebuilt `b12588-mix-c0c3de8` with the root-layout GGUF.

Target API: Gufo 0.33.0, extension API 0.2.21 (`TextModelRunner`, `DecodeBatch`/`AdvanceBatch`,
`PreparePrompt` with images, `TextPromptContext`, speculative `draft_backend`).

## What the model is (from the oracle source and the GGUF, see docs/GGUF-LAYOUT.md)

24 physical layers, hidden 2304, 24 query heads / 8 KV heads of 128, MoE 8 of 128 experts (FFN 1152, no
shared expert), vocab 201024 padded / 200058 real. Layers 6..17 run 3 times with shared weights, so a token
makes **48 layer applications** and the engine needs **48 cache slots**:

* 40 local (sliding window 1024) slots: layers 0-4 and 18-22 once (10), the 10 local layers of the core
  (6-10, 12-16) three times (30).
* 8 global slots: layer 5 and 23 once, layers 11 and 17 three times.
* Every slot owns its own K/V and its own convolution state, even when the weights are shared.

Mechanisms with no equivalent in the existing extensions:

1. No RoPE. A learned **relative-position bias** (`attn_rel`, `attn_rel_bias[16, 1024]`, `attn_rel_clip`)
   added to the attention logits by distance, plus a per-query **sink** (`attn_sink_proj`). Global layers
   multiply q and r by `tau = 1 + 0.1*log(max(1, (pos+1)/8192))`.
2. **Short causal convolutions with a recurrent state** (kernel 4) on K, V, the attention output and the FFN
   output. On global layers K and V use input-dependent taps (`attn_k_conv_proj`, `attn_v_conv_proj`).
   State per slot: 3 x 6656 floats.
3. q/k/r **L2 clipping** (`qk_clip` 22.63, `relative_clip` 8), V rms-norm without gamma, sigmoid gate on the
   attention output (`attn_gate`).
4. **Block attention residuals**: every 6 layers a new residual stream starts; each layer attends over the
   earlier block outputs with learned queries (`attn_res_query`, `ffn_res_query`, `output_res_query`).
5. **Loop controllers**: passes 2 and 3 receive a message from the previous pass, mixed in with tanh-gated
   low-rank controllers (`loop_{q,k,v,ffn}_ctrl_{down,gain,forget}`, `loop_msg_query`, rank 64).
6. MoE: sigmoid routing with bias, scale 8, activation SITU (beta 4 / linear beta 25), experts in **NVFP4**
   (blocks of 64: 4 UE4M3 scales + 32 bytes of E2M1) with an extra per-expert F32 `.scale` tensor. Gufo core
   does not know NVFP4: the extension declares `accepts_unsized_tensors` and ships its own decode and GEMV.
7. Final logit softcap 30, padded vocabulary cropped to 200058.

Memory per sequence (fp16 KV): local slots 40 x 4 MB (ring of 1024) = 160 MB; global slots 8 x 4 KB/token =
32 KB/token (8 GB at 262144); conv states ~4 MB. The 17.6 GB weights stay resident once.

## Milestones and gates

Each gate is a script in `gates/` that exits non-zero on failure. Goldens live outside git.

| M | Content | Gate |
|---|---------|------|
| M0 | Repo scaffold, oracle pod, golden capture (`llama-perplexity --kl-divergence-base`, `/tokenize`, `/apply-template`, `/completion` with `n_probs`) | G0: oracle reproduces ppl ~7.65 on the corpus |
| M1 | Host tier: GGUF config, NVFP4 reader, tokenizer (`inkling2` pre-tokenizer), Jinja chat template with effort levels, reasoning delimiters, tool dialect | G1 tokenizer and G2 template: 100 % identical to the oracle on the fixed corpus |
| M2 | **CPU reference forward** (F32, one token at a time, straight port of the oracle graph) **DONE** | G3: with the oracle's expert choice forced, every layer of all 48 slots within 2 % relative error of the oracle (measured: <= 1.1 %, the floor is the oracle's Q8_0 activation quantization); free-running top-1 identical on all tested positions, top-20 KL <= 0.02 |
| M3 | HIP kernels: attention with relative bias and sinks, conv, block mixing, loop controllers, NVFP4 MoE; batched over tokens and sequences **functional DONE** (naive kernels, 12 tok/s decode); optimisation and long context pending | G4: GPU vs CPU reference with forced experts <= 1e-3 relative on every tensor of the 48 slots (measured <= 4e-4 for decode, prefill batch, and chunks of 5/7 crossing state boundaries) |
| M4 | KV cache: 40 rings + 8 global slots, conv state, checkpoints, prefix reuse, 262144 context | G5: needle at 32k/128k/250k, snapshot restore bit-identical, ppl at 32k equal to oracle |
| M5 | **Batching**: `DecodeBatch` over N sequences with per-sequence rings/states | G8: concurrent identical requests byte-identical; N=4 aggregate throughput > 2.5x single |
| M6 | Vision: ViT (23 layers) + projector from `mmproj-BF16.gguf`, image spans in `TextPromptContext` | G6: image embeddings within 1e-2 of `llama-mtmd-cli`, same caption on 5 images |
| M7 | DSpark drafter (`dflash` GGUF, block 8, markov head, taps on applications 5/23/41/48) | G7: greedy output unchanged, acceptance rate and speedup reported |
| M8 | Package option in `think/gufo` (kydah/home), bench vs oracle, docs | Bench table in docs/ |

Order: M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M8. M2 is the pivotal one: it separates "the model is
misunderstood" from "a kernel is wrong". Every kernel in M3 is tested against the M2 code layer by layer.

## Working setup

* Development tree: this repo, built on worker04 inside a `gufo-runtime:0.33.0` pod (nix toolchain, hipcc,
  cmake, ninja; the 1.1 MB extension SDK is in `/lib/gufo/sdk`). Sources are synced to `/var/tmp/inkling-ext`.
* Oracle pod and the dev pod share the GPU (96 GB unified); production `gufo` must be at 0 replicas.
* Weights: `Inkling-2-Mini-UD-Q4_K_M.gguf` (Mainline layout) and `Inkling-2-Mini-mmproj-BF16.gguf` in the
  `gufo-model` PVC. The Unsloth root layout is only needed for DSpark.
* Never build in the code-server pod (6 GiB limit): build on the node.

## Open questions

* Exact meaning of DSpark `target_layer_ids [5, 23, 41, 48]` (0- or 1-based application indices); read the
  drafter code of llama.cpp PR #32218 / Unsloth's build.
* Whether sparse attention is worth porting for the 8 global slots at 250k (decode attention cost grows with
  context on those slots only).
* Reasoning budget behavior: the card says minimum effort does not fully disable reasoning.

## Findings from M2 (what the port got wrong first)

* The relative-position bias minus the sink is folded into the **attention mask** by the oracle, so it is added after
  the 1/sqrt(d) scaling of q.k. Scaling it too gave attention outputs ~500x too large.
* Expert selection flips on near-ties (8th vs 9th expert) because the oracle quantizes activations to Q8_0: with the
  router logits within 0.15 % the top-8 set still differs on a few tokens per layer. Validation therefore uses teacher
  forcing (`inkling_ref --force-experts DUMPDIR`); free-running agreement is measured on logits/top-k instead.
* Tools: `tools/oracle_dump` (oracle graph on CPU, F32 KV, no flash attention, dumps named tensors),
  `tools/inkling_ref --dump`, `gates/compare_dumps.py` (relative error per tensor), `gates/compare_topk.py`.

## Findings from M3

* The conv history of a token that is not the first of its batch needs a shift by its rank in the batch
  (`k_conv`); the first version indexed the stored history as if every token started the batch (wrong at chunk
  boundaries only, invisible in a single prefill batch).
* GPU vs CPU free-running routing differs on near-ties (a token or two per layer); validation forces the CPU
  reference's experts (`inkling_gpu --force-experts DUMPDIR`).
* The engine uploads the 17 GB of weights in 4 s and decodes at 12 tok/s with deliberately simple kernels
  (one wave per output row, 4 tokens per tile, thread-per-key attention): the optimisation pass comes after the
  Gufo integration.
