# Architecture

## Model (see PLAN.md for the full description)

24 physical layers, the middle 12 run three times: 48 *layer applications*, hence 48 cache slots (40 local with a 1024 window, 8 global),
each slot owning its K/V and its four convolution histories. Relative-position bias with sinks instead of RoPE, short causal
convolutions with a recurrent state, block attention-residuals, loop controllers, sigmoid-routed MoE (NVFP4 experts).

## Engine (`engine/`)

* One `Engine` per model: weights uploaded once (Q8_0 and NVFP4 rows are repacked to a *planar* layout, quantized values then block
  scales, 16-byte aligned, so lanes load 16 bytes at a time), sequences created on demand (`create_seq`): per slot K/V rings
  (local: 2048 rows, global: the configured context) in f16 plus the convolution histories. A step takes a batch of tokens
  `(token, sequence, position)`; kernels read the sequence and position per token, so prefill chunks, single-token decode of several
  sessions and speculative verification batches share the same code.
* Prefill (T >= 16): WMMA GEMMs (64x64 tile, weights dequantized to f16 in LDS), grouped MoE GEMMs on the per-expert token lists
  (deterministic order), flash-style attention on matrix cores with the relative bias read from a table built by a WMMA GEMM.
* Decode (T = 1): fused kernels (all projections of a layer in one launch, convolution + history + residual, clip/norm/tau/KV store,
  loop controllers, only the 8 chosen experts); the step can be replayed as a HIP graph. Decoding is bound by the number of dependent
  launches (~2000 per token, ~18 us latency each), not by memory traffic.
* Convolution state and speculation: a verification batch saves the old history and its inputs, `rollback_conv(seq, keep)` rebuilds
  the history after the accepted prefix. KV rows past the accepted position are simply overwritten later.

## Gufo integration (`gufo/`)

* `InklingTextRunner` (a `HipTextModelRunner`): prefill in chunks of 1024, sampling through the host's sampler, `DecodeBatch` /
  `AdvanceBatch` for plain sessions, `DecodeStep` with the DSpark loop for speculative ones.
* Continuation cache (`continuation_adapter.*`, `layout.hpp`): per slot a K and a V component (append rows for global slots,
  *window rows* of 1024 for local ones, 256 rows per chunk) plus one private component = the convolution histories of the 48 slots and
  the last logits. Row transfers copy between the device rings and cache pages; restoring a checkpoint is exact.
* Images: `PreparePrompt` splits message content at the image offsets, renders `<|content_image|><|unused_200054|><|end_message|>`
  blocks, expands the placeholder to one token per embedding row and returns a prompt context holding the embeddings; prefill replaces
  the token embedding by the image row (no embedding norm). The cache identity includes the image hashes.
* Reasoning: the runner hides the structural markers of the answer and keeps `<|content_thinking|>` / `<|end_message|>` of the thinking
  block visible so that the host splits reasoning from content.

## DSpark

`engine/dspark.hpp`: taps of the target at layer applications 5, 23, 41 (inputs of those applications) and the final normalized state
feed, per drafter layer, a softmax-weighted fusion -> keys/values (RoPE NEOX, base 1e6) kept in the drafter's own caches. A block is the
anchor token (target embedding) plus 7 mask rows run through 3 non-causal layers; the Markov head picks the 7 tokens one by one using the
target's lm_head. Verification is lossless: the tokens are the target's own samples.
