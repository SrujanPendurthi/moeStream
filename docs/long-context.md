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

## 81k with the recommended settings (2026-10-15, `results/longctx_bf16/`)

Same replay and harness as the first 81k run; set by config only: `kv_bits = 0`, `lookahead = false`,
`kv_sessions = 1`, tier on, step 2048, pool 1 GB, watchdog 4 GB / 17 GB. `memory_gb = 12`, not 14:
at 14 a RAM-hit turn's two bf16 KV copies (~3.3 GB each at 81k) predicted a footprint over the
17 GB watchdog. So slots are 4.47 GB here against 6.14 GB in the q8 run: two things changed.

| | q8, lookahead on, 14 GB (first run) | bf16, lookahead off, 1 session, 12 GB |
|---|---|---|
| prefill 80,583 tokens | 59.8 tok/s | 55.3 tok/s |
| decode at ~81k | 3.1-3.3 tok/s | **4.3-5.0 tok/s** |
| decode hit rate (turns 1-2) | 0.80 | 0.71 / 0.63 |
| MLX peak | 11.72 GB | **15.81 GB** |
| footprint (watchdog) / swap | 13.0 GB / 0.54 GB | 15.0 GB / 0.46 GB |
| SSD restore after restart | 82,074 tokens in 0.75 s | 82,062 tokens in 1.28 s |
| disk | 1.8 GB | 3.3 GB |

- **Decode +40-50% at 81k** (turn 0: 340 tokens in ~68 s after prefill, 5.0 tok/s; turn 3 after
  the restore ~4.7), with fewer slots and a lower hit rate.
- **bf16 goes through mlx-lm's batched path** (its caches can merge), and the SSD tier works there
  on real weights: the restore after restart took 1.28 s, no errors, no fallback. This closes the
  gap the final review flagged.
- **Memory is the cost.** On the batched path mlx-lm stores two cache entries per turn (the prompt,
  then prompt + reply), and the peaks came at the second insert of each turn (15.73, 15.81 GB). At
  81k that is ~3 bf16 KV copies live at once, not the one `kv_sessions = 1` reserves. With
  `memory_gb = 12` it fits under a 17 GB watchdog with ~1 GB to spare; at 128k it would not.

## Recommended defaults (for the user's decision; nothing applied)

(a) **Lookahead off** (`lookahead = false`): ~20 ms per token faster at 32k / 64 slots (186.5 to
166.0 ms) and steadier timing (a repeat within 2%); it was never measured as a win. Low risk.
(b) **bf16 KV with a 1-session reserve and the SSD tier** (`kv_bits = 0`, `kv_sessions = 1`): at 32k
decode within 4% of lagunaLab (113.3 vs 108.8 ms, against 166-169 ms at q8); at 81k 4.3-5.0 against
3.1-3.3 tok/s. But the MLX peak at 81k was 15.8 GB even at `memory_gb = 12`, because mlx-lm's
batched path keeps the prompt's and the reply's caches and deep-copies a RAM hit. So (b) needs one
more change before it is safe as a default at 128k: cap the RAM prompt cache to one entry on that
path (or skip the prompt-only insert), and re-measure. Until then: (b) at `memory_gb = 12` for
sessions up to ~80k, and q8 for longer ones.

## The three 81k runs, and where the extra KV copy was (2026-10-15)

| | 1: q8, lookahead on, 14 GB | 2: bf16, lookahead off, 1 session, 12 GB | 3: as 2, RAM cache 1 entry |
|---|---|---|---|
| results | `results/longctx/` | `results/longctx_bf16/` | `results/longctx_bf16_cap1/` |
| prefill 80,583 tokens | 59.8 tok/s | 55.3 tok/s | 56.2 tok/s |
| decode at ~81k | 3.1-3.3 tok/s | 4.3-5.0 tok/s | 4.1-5.0 tok/s |
| decode hit rate | 0.80 | 0.71 / 0.63 | 0.71 / 0.66 / 0.70 |
| MLX active between turns | 10.9 GB | 12.4 GB | 12.4 GB |
| MLX peak | 11.72 GB | 15.81 GB | 15.81 GB |
| footprint (watchdog) / swap | 13.0 / 0.54 GB | 15.0 / 0.46 GB | 16.0 / 0.46 GB |
| SSD restore after restart | 0.75 s | 1.28 s | 1.29 s |

Run 3 capped mlx-lm's RAM prompt cache to one entry (`prompt_cache_entries = 1`) and changed
nothing: the new log fields show the RAM cache held one entry (3.4 GB) throughout, and its byte cap
had already limited it to one session. Between turns, active memory was resident (1.16) + slots
(4.47) + two KV copies (2 x 3.4). The second copy was the SSD tier's writer: a snapshot's arrays
share the live cache's buffers, and the writer loop kept the last job bound until the next one
arrived. Fixed in c8f5b72 (test: `test_the_writer_lets_go_of_a_written_snapshot`). The same copy
was in run 1 (10.9 = 1.16 + 6.14 + 2 x 1.8). The peaks (+3.3 GB over the rest level, at the end of
a turn) are one more transient KV copy, likely the final insert or KV growth; not yet split.

**Expected with the fix (not yet measured):** bf16 at 81k ~9.1 GB between turns, peak ~12.4 GB,
footprint ~14 GB at `memory_gb = 12`; so `memory_gb = 14` (slots ~6.5 GB) should fit 81k under the
17 GB watchdog. At 128k, KV is 5.4 GB per copy: rest ~1.16 + 6.5 + 5.4 = 13.1 GB, peak ~18.5 GB:
does not fit at `memory_gb = 14` with bf16 unless the transient copy goes too; q8 at 128k
(2.85 GB per copy) would, with room.

## Fourth 81k run: the writer fix, and what still holds a copy (2026-10-15, `results/longctx_bf16_fix12/`)

Same config as run 2 (bf16, lookahead off, 1 session, `memory_gb = 12`) with the tier writer fix
(c8f5b72). The restart phase's server also ran the live-KV inventory in the memory log (5d5ea61);
the long phase did not, so the run mixes two code versions.

| | run 2 (no fix) | run 4 (writer fix) |
|---|---|---|
| MLX peak | 15.81 GB | **14.31 GB** |
| peaks at the end of each turn | 15.73, 15.81 | gone (12.75-12.85) |
| active at the next request's start | 12.4 GB | 12.4 GB |
| active at the next prompt's end | 12.4 GB | **9.03-9.06 GB** |
| footprint / swap | 15.0 / 0.46 GB | 15.0 / 0.45 GB |

The fix removed the end-of-turn peaks: the pinned snapshot had kept the old buffer alive while the
cache grew. One copy still lives from the end of a turn to the next request's prefill. The live-KV
inventory at the restart turn's end shows 20 full-attention `KVCache` objects (6.75 GB): the
prompt cache's one entry plus another full set. That set is the cache `fetch_nearest_cache` returned:
on mlx-lm's batched path the server merges it into a `BatchKVCache` (a copy) but keeps its local
`cache` bound until the next request's fetch. With a restore it is the restored cache; with a RAM
hit, the deep copy. So at 81k bf16, a turn holds ~2 KV copies (and briefly 3 at the next fetch's
deep copy: the 14.3 GB peaks).

**Not done, proposed:** release the fetched cache's arrays once mlx-lm has merged it (wrap
`BatchKVCache.merge` in `patch_server` to empty the source caches; safe because the batched path
never reads them after the merge and the prompt cache holds its own copy). Expected: ~9.1 GB at rest
and ~12.4 GB peak at 81k bf16, which would let `memory_gb = 14` fit. q8 takes the single path,
which merges nothing, so it is unaffected. The `memory_gb = 14` run was not done: its condition
(peak <= 13 GB) was not met.

## Fifth 81k run: `release_merged_caches` (2026-10-15, `results/longctx_bf16_rel12/`)

Run 4's config plus `release_merged_caches = true`. **Long phase only:** I edited
`tools/longctx_check.sh` (a new knob) while this run's copy was executing; bash reads a script as
it goes, so it skipped the end of the long phase and started a second server without stopping the
first. Everything was stopped by PID at 07:02; the restart phase did not run. Lesson: never edit a
script while a run of it is going (copy it per run).

Long phase (3 turns): MLX peak **14.31 GB, unchanged from run 4**; active after each turn 12.4-12.5 GB;
footprint 15.0 GB, swap 0.43 GB. So emptying merged caches had no measurable effect. The live-KV
inventory (all turns this time) shows the copy that remains: after each turn's final insert, 20
full-attention `KVCache` objects (6.6-6.7 GB): the prompt cache's one entry plus the end-of-prompt
cache, which the final insert evicts but mlx-lm's batched path still holds in the loop variable of
its segment-insert loop (`for uid, (cache, cache_key) in caches.items()`), until the next turn.
It is there after a cold turn too (turn 0: no fetched cache, nothing merged).

`release_evicted_caches` (f09564c, default false) empties an entry's arrays when the prompt cache
drops it, which would free that copy. Built and tested on the tiny model; not yet run on real weights.
