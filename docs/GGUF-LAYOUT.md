# GGUF layout (Mainline_Support, UD-Q4_K_M)

Generated with `tools/gguf_dump.py` from `unsloth/Inkling-2-Mini-GGUF` `Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file` (architecture `inkling2_loop`) and `mmproj-BF16.gguf` (architecture `clip`, projector `inkling2v`).

## Text model metadata
```
# GGUF v3, 744 tensors, 59 kv
general.architecture = 'inkling2_loop'
general.type = 'model'
general.name = 'Inkling-2-Mini'
general.size_label = '128x1.8B'
inkling2_loop.block_count = 24
inkling2_loop.context_length = 1048576
inkling2_loop.embedding_length = 2304
inkling2_loop.feed_forward_length = 1152
inkling2_loop.attention.head_count = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
inkling2_loop.attention.head_count_kv = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
inkling2_loop.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
inkling2_loop.expert_count = 128
inkling2_loop.expert_used_count = 8
inkling2_loop.attention.key_length = 128
inkling2_loop.attention.value_length = 128
inkling2_loop.loop.count = 3
inkling2_loop.loop.start = 6
inkling2_loop.loop.end = 18
inkling2_loop.loop.gate_rank = 64
inkling2_loop.vocab_size = 201024
inkling2_loop.vocab_size_unpadded = 200058
inkling2_loop.expert_feed_forward_length = 1152
inkling2_loop.attention.key_length_swa = 128
inkling2_loop.attention.value_length_swa = 128
inkling2_loop.rope.dimension_count = 0
inkling2_loop.attention.sliding_window = 1024
inkling2_loop.attention.sliding_window_pattern = [True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, False]
inkling2_loop.shortconv.l_cache = 4
inkling2_loop.expert_weights_scale = 8.0
inkling2_loop.activation.situ_beta = 4.0
inkling2_loop.activation.situ_linear_beta = 25.0
inkling2_loop.attn_res.block_size = 6
inkling2_loop.final_logit_softcapping = 30.0
inkling2_loop.attention.relative_length = 16
inkling2_loop.attention.relative_extent = 1024
inkling2_loop.attention.qk_clip = 22.627416610717773
inkling2_loop.attention.relative_clip = 8.0
inkling2_loop.attention.log_scale_floor = 8192
inkling2_loop.attention.log_scale_alpha = 0.10000000149011612
tokenizer.ggml.model = 'gpt2'
tokenizer.ggml.pre = 'inkling2'
tokenizer.ggml.tokens = <200058 items>
tokenizer.ggml.token_type = <200058 items>
tokenizer.ggml.merges = <424413 items>
tokenizer.ggml.eos_token_id = 200006
tokenizer.ggml.eot_token_id = 200006
tokenizer.ggml.add_bos_token = False
tokenizer.ggml.add_eos_token = False
tokenizer.ggml.clean_spaces = False
general.quantization_version = 2
general.file_type = 15
general.basename = 'Inkling-2-Mini'
tokenizer.ggml.padding_token_id = 199999
general.sampling.temp = 1.0
general.sampling.top_k = 0
general.sampling.top_p = 1.0
general.sampling.min_p = 0.0
inkling2_loop.loop.varying = False
```

## Non-layer tensors
```
output.weight	[2304, 201024]	Q8_0	0
output_norm.weight	[2304]	F32	492106752
output_res_query.weight	[2304]	F32	492115968
token_embd.weight	[2304, 201024]	Q8_0	492125184
token_embd_norm.weight	[2304]	F32	984231936
```

## Layer 0 (local), layer 5 (global), layer 6 (first loop-core layer)
```
blk.0.attn_gate.weight	[2304, 3072]	Q8_0
blk.0.attn_k.weight	[2304, 1024]	Q8_0
blk.0.attn_k_conv.weight	[4, 1024]	F32
blk.0.attn_norm.weight	[2304]	F32
blk.0.attn_output.weight	[3072, 2304]	Q8_0
blk.0.attn_post_conv.weight	[4, 2304]	F32
blk.0.attn_q.weight	[2304, 3072]	Q8_0
blk.0.attn_q_clip.weight	[128, 24]	F32
blk.0.attn_rel.weight	[2304, 384]	Q8_0
blk.0.attn_rel_bias.weight	[16, 1024]	F32
blk.0.attn_rel_clip.weight	[16, 24]	F32
blk.0.attn_sink_proj.weight	[2304, 24]	BF16
blk.0.attn_v.weight	[2304, 1024]	Q8_0
blk.0.attn_v_conv.weight	[4, 1024]	F32
blk.0.exp_probs_b.bias	[128]	F32
blk.0.ffn_down_exps.scale	[128]	F32
blk.0.ffn_down_exps.weight	[1152, 2304, 128]	NVFP4
blk.0.ffn_gate_inp.weight	[2304, 128]	F32
blk.0.ffn_gate_up_exps.scale	[128]	F32
blk.0.ffn_gate_up_exps.weight	[2304, 2304, 128]	NVFP4
blk.0.ffn_norm.weight	[2304]	F32
blk.0.ffn_post_conv.weight	[4, 2304]	F32
blk.0.ffn_res_query.weight	[2304]	F32
blk.5.attn_gate.weight	[2304, 3072]	Q8_0
blk.5.attn_k.weight	[2304, 1024]	Q8_0
blk.5.attn_k_conv.weight	[4, 1024]	F32
blk.5.attn_k_conv_proj.weight	[2304, 32]	BF16
blk.5.attn_k_inp_norm.weight	[2304]	F32
blk.5.attn_output.weight	[3072, 2304]	Q8_0
blk.5.attn_post_conv.weight	[4, 2304]	F32
blk.5.attn_q.weight	[2304, 3072]	Q8_0
blk.5.attn_q_clip.weight	[128, 24]	F32
blk.5.attn_q_inp_norm.weight	[2304]	F32
blk.5.attn_rel.weight	[2304, 384]	Q8_0
blk.5.attn_rel_bias.weight	[16, 1024]	F32
blk.5.attn_rel_clip.weight	[16, 24]	F32
blk.5.attn_res_query.weight	[2304, 3]	F32
blk.5.attn_sink_proj.weight	[2304, 24]	BF16
blk.5.attn_v.weight	[2304, 1024]	Q8_0
blk.5.attn_v_conv.weight	[4, 1024]	F32
blk.5.attn_v_conv_proj.weight	[2304, 32]	BF16
blk.5.attn_v_inp_norm.weight	[2304]	F32
blk.5.exp_probs_b.bias	[128]	F32
blk.5.ffn_down_exps.scale	[128]	F32
blk.5.ffn_down_exps.weight	[1152, 2304, 128]	NVFP4
blk.5.ffn_gate_inp.weight	[2304, 128]	F32
blk.5.ffn_gate_up_exps.scale	[128]	F32
blk.5.ffn_gate_up_exps.weight	[2304, 2304, 128]	NVFP4
blk.5.ffn_norm.weight	[2304]	F32
blk.5.ffn_post_conv.weight	[4, 2304]	F32
blk.5.ffn_res_query.weight	[2304]	F32
blk.6.loop_q_ctrl_down.weight	[4608, 64]	F32
blk.6.loop_q_ctrl_forget.weight	[64, 2304]	F32
blk.6.loop_q_ctrl_gain.weight	[64, 2304]	F32
blk.6.loop_k_ctrl_down.weight	[4608, 64]	F32
blk.6.loop_k_ctrl_forget.weight	[64, 2304]	F32
blk.6.loop_k_ctrl_gain.weight	[64, 2304]	F32
blk.6.loop_ffn_ctrl_down.weight	[4608, 64]	F32
blk.6.loop_ffn_ctrl_forget.weight	[64, 2304]	F32
blk.6.loop_ffn_ctrl_gain.weight	[64, 2304]	F32
blk.6.loop_v_ctrl_down.weight	[4608, 64]	F32
blk.6.loop_v_ctrl_forget.weight	[64, 2304]	F32
blk.6.loop_v_ctrl_gain.weight	[64, 2304]	F32
blk.6.loop_msg_query.weight	[2304, 4]	F32
```

Tensor types: 518 F32, 150 Q8_0, 40 NVFP4 (all MoE experts), 32 BF16, 4 Q5_K (the 24 physical layers only; looped passes reuse the weights).

## mmproj metadata
```
# GGUF v3, 422 tensors, 19 kv
general.architecture = 'clip'
general.type = 'mmproj'
general.name = 'Inkling-2-Mini'
general.size_label = '318M'
general.file_type = 32
clip.has_vision_encoder = True
clip.vision.projection_dim = 2304
clip.vision.image_size = 640
clip.vision.patch_size = 20
clip.vision.embedding_length = 1024
clip.vision.feed_forward_length = 4096
clip.vision.block_count = 23
clip.vision.attention.head_count = 16
clip.vision.image_mean = [0.5, 0.5, 0.5]
clip.vision.image_std = [0.5, 0.5, 0.5]
clip.vision.projector_type = 'inkling2v'
clip.vision.attention.layer_norm_epsilon = 9.999999747378752e-06
clip.vision.spatial_merge_size = 2
general.quantization_version = 2
```

## mmproj tensors (non-block + block 0)
```
v.class_embd	[1024]	F32
v.class_position_embd.weight	[1024]	F32
v.patch_embd.weight	[20, 20, 3, 1024]	F32
v.position_embd.weight	[1024, 1024]	F32
v.pre_ln.bias	[1024]	F32
v.pre_ln.weight	[1024]	F32
mm.1.weight	[4096, 4096]	BF16
mm.2.weight	[4096, 2304]	BF16
v.blk.0.ln1.bias	[1024]	F32
v.blk.0.ln1.weight	[1024]	F32
v.blk.0.ln2.bias	[1024]	F32
v.blk.0.ln2.weight	[1024]	F32
v.blk.0.ls1.weight	[1024]	F32
v.blk.0.ls2.weight	[1024]	F32
v.blk.0.ffn_up.bias	[4096]	F32
v.blk.0.ffn_up.weight	[1024, 4096]	BF16
v.blk.0.ffn_down.bias	[1024]	F32
v.blk.0.ffn_down.weight	[4096, 1024]	BF16
v.blk.0.attn_k.bias	[1024]	F32
v.blk.0.attn_k.weight	[1024, 1024]	BF16
v.blk.0.attn_out.bias	[1024]	F32
v.blk.0.attn_out.weight	[1024, 1024]	BF16
v.blk.0.attn_q.bias	[1024]	F32
v.blk.0.attn_q.weight	[1024, 1024]	BF16
v.blk.0.attn_v.bias	[1024]	F32
v.blk.0.attn_v.weight	[1024, 1024]	BF16
```
