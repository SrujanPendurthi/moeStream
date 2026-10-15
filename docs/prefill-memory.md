# Prefill memory: what grows, and how to measure it

**Update 2026-10-15, after the main pass (`results/prefill_mem_main.json`): cause found and fixed
(f357884).** The transient grew ~0.4 GB per prefill *chunk* (0.20 GB per 1k tokens at step 2048,
0.39 at step 1024), not per token: halving the step doubled the growth. mlx-lm's prefill evaluates
only the caches after each chunk; nothing reads the last layer's expert output, so that layer's
slot writes (one layer of routed experts, ~0.4 GB) stayed pending across chunks until the end of
the prompt. `StreamingMoE` now `async_eval`s its slot tensors at the end of every multi-token call.
Reproduced and pinned at tiny scale
(`test_prefill_chunks_evaluated_as_mlx_lm_does_do_not_pile_up_slot_writes`). The analysis below
was written before this and is kept as the record; its "unexplained ~3.3 GB" is this.

| step | 5k | 15k | 30k |
|---|---|---|---|
| 2048: transient / top MEM / tok/s | 1.97 GB / 11.0 GB / 81 | 4.14 / 13.3 / 82 | 6.98 / 16.0 / 72 |
| 1024: transient / top MEM / tok/s | 2.65 / 12.3 / 75 | 6.52 / 16.0 / 67 | watchdog (17.4 GB footprint) |

Status 2026-10-15. Off-GPU analysis after two real-weight attempts of `tools/prefill_mem.py`
were stopped by the 2 GB swap guard with the machine in use (`results/prefill_mem_attempt2*`,
`results/prefill_mem_attempt3*`). No fitted context term yet; this page says what the code
predicts and how the next run should be built.

## What was observed

| attempt | memory_gb | point | MLX active before | MLX peak | transient | top MEM | tok/s |
|---|---|---|---|---|---|---|---|
| 2 | 12 | step 512, 5k tokens | 1.16 GB (no warm-up) | 13.34 GB | not usable | 15.0 GB | 61.9 |
| 3 | 10 | step 512, 5k tokens | 7.55 GB (after 2k warm-up) | 11.62 GB | 4.06 GB | 13.3 GB | 48.1 |
| 3 | 10 | step 512, 15k tokens | | | stopped: swap 2084 MB | footprint 16.4 GB (watchdog) | |

The 7.55 GB active after the warm-up equals slots (6.40 GB from the load banner) plus resident
weights (1.16 GB), so the warm-up did fill every layer's slots: slot tensors are allocated in full
the first time a layer is used. Attempt 3's 4.06 GB transient is therefore real, not slot growth.

## What the code allocates during a prefill chunk

**Attention is not the cost.** Laguna calls `mx.fast.scaled_dot_product_attention`. Full-attention
layers get the string mask `"causal"` (mlx-lm `create_attention_mask`, N > 1, no window), so the
fused kernel runs and no chunk x context score matrix is built. Sliding layers get a boolean mask
of chunk x (window 512 + slack 64 + chunk), a few MB. What does grow with context, per full layer
and only while that layer runs:

- `DequantKVCache` dequantizes the layer's whole KV to bf16 for every chunk:
  8 heads x 128 x 2 bytes x (K, V) = 4096 bytes per token: 61 MB at 15k, 123 MB at 30k.
- `QuantizedKVCache` grows in steps of 256 by concatenation, briefly holding two copies of one
  layer's q8 KV: 2176 bytes per token, 33 MB at 15k, 65 MB at 30k.
- The KV itself stays: 21,760 bytes per token over the 10 full layers, 0.022 GB per 1k tokens.

Together that is about 0.2 GB at 30k: the attention path alone cannot produce the 0.21 GB per 1k
tokens seen in the Task 18 memory log (12.07 GB at 5.4k to 14.16 GB at 15.2k).

**Expert slot writes are.** A prefill chunk routes each layer's tokens to most of the 256 experts.
From the recorded routing (`data/routing`, 11 sessions), distinct experts per layer per chunk:

| step | distinct experts (mean) | upper bound read per 1k prompt tokens |
|---|---|---|
| 512 | 200 | 26.9 GB |
| 1024 | 215 | 14.5 GB |
| 2048 | 230 | 7.8 GB |

That is more than the 64-192 slots a layer has, so `StreamingMoE.__call__` runs the layer in
groups of at most `slots` experts, filling slots before each group. All of it is lazy until the
next layer's `mx.eval`. Measured on Metal with a 64 MB slot tensor
(scratch check, 64 slots refilled per group):

| groups in one layer | transient |
|---|---|
| 1 | 64 MB (the new expert arrays) |
| 2 | 192 MB |
| 3 | 275 MB |
| 4 | 338 MB |

Each extra group copies the whole slot tensor: group g's matmul still holds the old version when
group g+1 writes, so MLX cannot write in place. Per layer, the transient is about
`distinct x 1.77 MB + (groups - 1) x slots x 1.77 MB`:

| memory_gb (ctx 32k) | slots per layer | step 512 | step 2048 |
|---|---|---|---|
| 10 | 64-128 | 0.64 GB mean, 0.74 max | 0.73 GB mean, 0.78 max |
| 12 | 96-192 | 0.58 GB mean, 0.78 max | 0.67 GB mean, 0.79 max |

So about 0.8 GB at worst if only one layer's graph is alive at a time, which `mx.eval(inds)` at
the start of every layer should guarantee. That covers a fifth of attempt 3's 4.06 GB.
**The other ~3.3 GB is not explained by reading the code**; the next run has to split it.

## The gap between top MEM and MLX's peak (2-4 GB)

top's MEM is the whole process; MLX's peak counts only its active buffers. Outside them:

| what | worst case |
|---|---|
| MLX's free-buffer pool (`mlx_cache_gb`) | 1.0 GiB |
| prefetch staging buffers kept free (`keep = 2 x top_k = 16` per layer x 39 layers x 1.77 MB) | 1.10 GB |
| staging buffers in flight during one group (up to `slots` x 1.77 MB, freed after the copy) | 0.34 GB |
| Python, tokenizer, numpy routing arrays | about 0.3-0.5 GB |
| SSD tier restore buffer (serve only, tier off in this study) | one chain's KV, 0.3-0.7 GB |

About 2.7-2.9 GB without the tier, against 1.7 GB observed at point 1 (13.3 - 11.6). Point 2's
MLX peak was never logged (stopped mid-point), so its 16.4 GB footprint cannot be split. The staging buffers and the pool
are not in `memory_gb`.

## Why swap rose

The smaller the prefill step, the more expert bytes are read per prompt token (table above:
27 GB per 1k tokens at step 512 against 7.8 GB at 2048, before slot hits). The packed experts are
17.7 GB, larger than the memory left on this machine, so a 512-step prefill streams the whole pack
through the page cache every ~0.7k tokens. That pressure, not only the process footprint, pushed
other programs into swap. Step 512 first was the worst order for a shared machine.

## Corrected study design

Built (`tools/prefill_mem.py`, `moestream.layer.EVAL_GROUPS`): items 1-4 below; item 4 is
`--eval-groups`. Run plan: `.gpu/gpu-run moe tools/watch.sh 4 16 python3 -m tools.prefill_mem
--config results/prefill.toml`, then the same with `--eval-groups --steps 2048 --ctx 15000`.

1. **Order:** step 2048 first, then 1024, then 512 (least disk traffic first); lengths 5k, 15k, 30k.
2. **Baseline:** keep the 2k-token warm-up; also log `active_gb` before the warm-up and check that
   active after it equals slots + resident (it did in attempt 3).
3. **Per point, add:** bytes read (`stream_stats` misses x stride) and pageins (`vm_stat`), so disk
   pressure and transient memory are separated; MLX `get_cache_memory` at peak.
4. **Split the unexplained 3.3 GB:** one extra pass of step 2048 x 15k with `mx.eval(yg)` after each
   group inside `StreamingMoE.__call__` (diagnostic flag, like `EVAL_WRITES`). If the transient falls
   by the slot-copy term, the per-group copies are the cost and the fix is that eval (or one group
   per eval); if not, look across layers (are layer graphs alive past the next `mx.eval`?).
5. **When:** only with the user away, with the watchdog's own limits (`watch.sh 4 16`), no 2 GB guard.
6. **Report:** per step, the fit `transient = a + b x tokens`; if `b x 128k` is material, the budget
   gets a context-dependent activation term in place of the flat 1 GB.

**Prediction from this page:** `b` is small (about 0.01 GB per 1k tokens from the attention path),
`a` is set by the slot-write term (0.6-0.8 GB) plus whatever the split in step 4 finds; and the
budget's flat 1 GB plus the 1.1 GB of staging buffers outside `memory_gb` should be accounted for
before a 128k session is trusted on 24 GB.

## Design alternative: one group per layer in prefill (sketch, not built)

**Idea.** During a prefill chunk, do not run a layer's experts through its slots. Gather the
chunk's distinct routed experts (200-230 of 256) into a temporary, prefill-only expert set: the
ones already in slots are copied from the slot tensors on the GPU, the rest are read from disk.
Run the layer in one group against that set, then let it go at the next layer's `mx.eval`. Slots
are left as decode tuned them; optionally refill the few hottest experts of the chunk afterwards.

**Memory, per layer, one layer alive at a time** (stride 1.77 MB; distinct from recorded routing):

| step | now: fills + slot copies (memory_gb 10) | one group: temporary set |
|---|---|---|
| 512 | 0.64 GB mean, 0.74 max | 200 x 1.77 = 0.35 GB |
| 2048 | 0.73 GB mean, 0.78 max | 230 x 1.77 = 0.41 GB |

About 0.3-0.4 GB lower peak, and the term no longer depends on the slot count, so `memory_gb` can
give slots more without raising the prefill peak. It also removes the in-slot copy that
`--eval-groups` would only serialize.

**Time.** Disk reads stay the same for experts not in slots (upper bound 7.8 GB per 1k tokens at
step 2048). Saved: the (groups - 1) slot-tensor copies per layer (0.1-0.3 GB of GPU copies, a few
ms per layer at GPU bandwidth, about 0.1-0.2 s per 2048-token chunk) and the per-group dispatch.
The larger gain is after prefill: today a long prefill evicts decode's hot experts from every layer
(each chunk cycles 200+ experts through 64-192 slots), so the first decode tokens miss; with slots
left alone, decode starts at its steady hit rate. Expected prefill speed change: small (+0-10%,
not measured); decode after a long prompt: back to the steady ~90% hit rate at once.

**Costs.** A second code path in `StreamingMoE.__call__` (prefill vs decode), and a gather from the
slot tensors for resident experts. The prefill transient becomes about 0.4 GB per layer
regardless of slots, which the budget's flat 1 GB already covers.
