# Long context on real weights (2026-10-15)

`tools/longctx_check.sh` (results in `results/longctx/`): lagunaLab's longest captured session
(django__django-11039) replayed through the moeStream server from four turns before its end, so the
first request is one 80,583-token prompt (thinking on renders the recorded reasoning), then two more
turns, a server restart, and one turn restored from the SSD tier. Config: `memory_gb = 14`,
`context_tokens = 131072`, q8 KV, `mlx_cache_gb = 1`, prefill step 2048, tier on, tuner off,
the pending-slot-write fix (f357884) in, `PREFILL_SET` off. Watchdog `watch.sh 4 17`. One clean run.

| turn | prompt tokens | reused | MLX peak | hit rate | time | tokens out |
|---|---|---|---|---|---|---|
| 0 | 80,583 | 0 | 10.44 GB | 0.23 | 1,451 s | 340 |
| 1 | 81,054 | 80,923 (RAM) | 11.66 GB | 0.80 | 124 s | 378 |
| 2 | 81,636 | 81,432 (RAM) | 11.72 GB | 0.80 | 143 s | 438 |
| restart, 3 | 82,202 | 82,074 (SSD, 0.75 s) | 9.83 GB | 0.77 | 112 s | 340 |

- **Prefill:** 80,583 tokens in 1,347 s, **59.8 tok/s** averaged over the prompt (about 43 tok/s per
  chunk near the end).
- **Decode at ~81k:** about **3.1-3.3 tok/s** (turns 0 and 3: 340 tokens in ~104-105 s after prefill).
- **Memory:** MLX peak 11.72 GB against the 12-13 GB predicted; footprint 13.0 GB (watchdog) against
  14-15 predicted; swap 0.52-0.54 GB throughout. RAM hits peak ~1.8 GB higher than the SSD restore
  (mlx-lm deep-copies a RAM hit; a restore takes its place).
- **SSD tier:** after the restart, 82,074 tokens restored in **0.75 s** instead of a ~1,350 s
  prefill; 1.8 GB on disk for the conversation.

**Against lagunaLab:** its README records the same session (pruned pack, streamed) at 4.7 tok/s
decode and 8.04 GB peak; its streaming-64-slot check measured 14.5 tok/s at 32k. moeStream at 81k
(full pack, 14 GB) decodes slower than both. Not yet split: how much of the ~300 ms per token is
the full-KV dequantize per decode step on the 10 full-attention layers (`DequantKVCache`) and how
much is expert misses (hit rate 0.80).

## Decode profile at 32k (2026-10-15, `tools/decode_prof.py`)

64 slots on every layer (lagunaLab's comparison row), pool 1 GB, 32,000-token prefill, 64 greedy
tokens, one KV width per process (`results/decode_prof_kv8_and_tainted_kv0.json` kv8 row,
`results/decode_prof_kv0.json`). Split from the instrumented pass (an eval after every attention and
layer), plain tok/s from an uninstrumented pass over the same prefix (same tokens both ways).

| ms per token | kv_bits 8 (DequantKVCache) | kv_bits 0 (bf16) | lagunaLab 64 slots, bf16 |
|---|---|---|---|
| plain step | 186.5 (5.36 tok/s) | 192.4 (5.20 tok/s) | 72.2 (13.9 tok/s) |
| full attention, 10 layers | 81.2 | 47.8 | |
| sliding attention, 30 layers | 23.6 | 37.6 | |
| MoE (sync / read / write) | 77.9 (14 / 15 / 8) | 113.4 (20 / 18 / 13) | |
| other | 1.9 | 2.8 | |
| hit rate / MLX peak | 0.77 / 8.3 GB | 0.77 / 9.5 GB | |

- **Dequantizing q8 KV costs ~33 ms per token at 32k** (full attention 81 to 48 ms), close to
  lagunaLab's synthetic ~26 ms. Scaled to 81k: ~60-85 ms of the measured ~310 ms per token.
- **The two runs differ by 14 ms in sliding attention and 35 ms in MoE**, parts the KV width does
  not touch: run-to-run variance (disk, page cache) of that size. Plain totals are level
  (186 vs 192 ms), so at 32k bf16 KV buys nothing measurable overall.
- **The gap to lagunaLab is not mainly KV:** at the same 64 slots and hit rate, moeStream's MoE alone
  (78-113 ms) and its attention (105-85 ms) each match or exceed lagunaLab's whole step (72 ms).

**bf16 KV at 81k, estimate (input for the default-KV decision, not applied):** removing the
dequantize saves ~60-85 ms of ~310 ms, so ~4.0-4.3 tok/s instead of ~3.2. Memory: bf16 KV is
40,960 bytes per token against 21,760 at q8, +1.56 GB at 81k per session; at `context_tokens =
131072` and `memory_gb = 14` the 2-session reserve would leave ~1.1 GB of slots (unusable), a
1-session reserve ~6.5 GB (today, q8 with 2 sessions: 6.14).

## Decode gap to lagunaLab, bisected (2026-10-15)

Same harness for every row (`tools/decode_prof.py`): 32,000-token prompt, 64 slots on every layer,
MLX pool 1 GB, 64 greedy tokens, one configuration per process. Plain = uninstrumented step;
the split is from the instrumented pass (ms per token).

| run | plain ms (tok/s) | full attn | sliding attn | MoE | other | file |
|---|---|---|---|---|---|---|
| lagunaLab `load_streaming`, bf16 | 108.8 (9.19) | 31.7 | 20.9 | 70.5 | 1.7 | `decode_prof_laguna.json` |
| moeStream kv8, lookahead on | 186.5 (5.36) | 81.2 | 23.6 | 77.9 | 1.9 | `decode_prof_kv8_and_tainted_kv0.json` |
| moeStream kv0, lookahead on | 192.4 (5.20) | 47.8 | 37.6 | 113.4 | 2.8 | `decode_prof_kv0.json` |
| moeStream kv8, lookahead off | 166.0 (6.02) | 74.2 | 22.1 | 79.6 | 1.7 | `decode_prof_noprefetch.json` |
| same, repeat | 169.0 (5.92) | 75.7 | 22.9 | 80.0 | 1.7 | `decode_prof_noprefetch_repeat.json` |
| **moeStream kv0, lookahead off** | **113.3 (8.83)** | **31.7** | 20.7 | 75.1 | 1.8 | `decode_prof_kv0_noprefetch.json` |

- **The gap is two settings, not the engine.** With bf16 KV and lookahead off, moeStream is within
  4% of lagunaLab (113.3 vs 108.8 ms); full attention is identical (31.7 ms), MoE 4.6 ms slower.
- **q8 KV costs ~42-44 ms per token at 32k** (full attention 74-76 to 32 ms): `DequantKVCache`
  rewrites the whole layer's KV every step. More than lagunaLab's synthetic ~26 ms; there is no
  other difference in the attention path (same model classes, caches and masks at kv0).
- **Lookahead costs ~20 ms per token at 32k with 64 slots** (186.5 to 166.0 at kv8) and makes runs
  noisy: with it on, parts it does not touch moved 14-35 ms between runs; with it off, a repeat
  landed within 2% (166.0 / 169.0) and every part within 2 ms.
- lagunaLab's own 72.2 ms row is not reproduced in this harness (108.8); the comparison above is
  like for like.

**Defaults this suggests (not applied; the user decides):**

1. **Lookahead off** (`io.next = None`): -20 ms per token at 32k / 64 slots (11%), and steadier
   timing. The earlier 128-slot bench found it about break-even; nowhere was it measured as a win.
2. **bf16 KV instead of q8**, paired with reserving KV for one session instead of two (the SSD tier
   holds other conversations): -53 ms per token at 32k (166 to 113, 1.47x). At 81k the dequantize
   share grows with context (~2.5x), so decode would go from ~3.2 to roughly 5 tok/s (estimate,
   not measured). Memory: +19,200 bytes per token (1.56 GB at 81k, 2.5 GB at 128k). At
   `memory_gb = 14`, `context_tokens = 131072`: today (q8, 2 sessions) 6.14 GB of slots; bf16 with
   2 sessions ~1.1 GB (unusable); bf16 with 1 session ~6.5 GB. With the prefill budget proposal
   (activations 1.3 GB + 0.021 GB per 1k) on top: bf16, 1 session, ~3.4 GB.
   An alternative that keeps q8's memory: quantized SDPA at decode (lagunaLab's
   `quantized_decode=True`); its synthetic numbers put that at 12.05 against 12.98 ms per layer at
   82k, so it would recover little of the 42 ms.
