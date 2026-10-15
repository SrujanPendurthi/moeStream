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
