# moeStream

Streams MoE expert weights from disk on Apple Silicon, with a decayed-LFU slot cache, next-layer lookahead prefetch and a per-layer slot budget calibrated offline from recorded routing. Phase 1 compares it against lagunaLab's streaming loader on Laguna-XS-2.1.

Documentation: [docs/](docs/README.md): architecture, configuration, serving and pi, the self-tuning runtime, tools, development.

## Run

```bash
python3 -m pytest tests -v
PYTHONPATH=.:../lagunaLab python3 -m tools.record      # routing over lagunaLab/traces -> data/routing (loads model)
PYTHONPATH=.:../lagunaLab python3 -m tools.calibrate   # offline -> budget.json
PYTHONPATH=.:../lagunaLab python3 -m tools.bench       # warm -> results/bench.json
PYTHONPATH=.:../lagunaLab python3 -m tools.bench --cold  # cold page cache -> results/bench_cold.json
```

`record` and `bench` load real weights; check no other heavy job is running first.

## Where it stands (2026-10-15)

Measured, and what the gate now is. Every speed number taken with MLX's buffer pool uncapped is
superseded: the uncapped pool evicted expert files from the page cache (lagunaLab 9c35d6e) and
handicapped whichever engine it hit. moeStream now caps the pool at 1 GB by default
(`mlx_cache_gb`).

**Speed: parity at equal memory.** 14.4 GB cap, both engines' pools capped at 1 GB
(`results/bench_cap14.4_pool1.json`):

| | lagunaLab (157 slots) | moeStream | ratio |
|---|---|---|---|
| short-prompt decode | 17.20 tok/s | 17.31 tok/s | 1.01x |
| 16k decode | 13.20 tok/s | 13.74 tok/s | 1.04x |
| 16k prefill | 131.6 tok/s | 106.9 tok/s | 0.81x |

moeStream is not faster than lagunaLab per token; prefill is 19% slower.

**Multi-turn: the SSD tier.** On three interleaved agent sessions (15 turns, `results/kvtier/`),
restoring each conversation's KV from SSD instead of re-processing it cut prompt work 3.7x
(135,132 to 36,358 tokens, the minimum possible) and wall time 2.8x (1,478 s to 519 s). The tier
survives a server restart (3 of 3 sessions restored) and `kill -9` mid-write (torn snapshot dropped,
nothing orphaned), and a disk restore gives the same output as a RAM hit. Details below.

**Long sessions (2026-10-15, `docs/long-context.md`).** An 81k-token agent session runs at
`memory_gb = 14` with q8 KV: 13.0 GB footprint, prefill 59.8 tok/s, decode 3.1-3.3 tok/s, and after a
restart the SSD tier restores the 82k-token conversation in 0.75 s instead of a ~22-minute prefill.
Fixed on the way: prefill memory grew ~0.4 GB per chunk (pending slot writes, f357884), and the
tier writer pinned a turn's KV (c8f5b72). At 32k with 64 slots, the decode gap to lagunaLab was two
settings, not the engine: q8 KV's dequantize (~42 ms per token) and lookahead prefetch (~20 ms); with
bf16 KV and lookahead off moeStream is within 4% of lagunaLab (113.3 vs 108.8 ms).

**Defaults since 2026-10-15** (the user's decision): lookahead off; KV stays q8; the activation
reserve is 1.3 GB + 0.021 GB per 1k context tokens instead of a flat 1 GB; KV is reserved for one
session (`kv_sessions = 1`), the SSD tier holding the other conversations, so the default keeps 5.94 GB
of slots at 128k (was 6.12 GB with a flat reserve and two sessions); bf16 KV at 81k fits at `memory_gb = 12`
with `release_evicted_caches = true` (opt-in).

## Results (warm, `results/bench.json`, superseded: uncapped pool, Phase 1)

| | lagunaLab slots=32 | moeStream |
|---|---|---|
| decode tok/s | 14.99 | 5.94 |
| hit rate (decode) | 0.547 | 0.905 |
| peak GPU GB | 5.91 | 15.50 |
| long-ctx (16000) decode tok/s | 8.23 | 6.88 |

moeStream time split per token: sync 145.1 ms, read 10.3 ms, write 4.1 ms. Prediction accuracy 0.706. Calibration chose half-life 64 (`results/calibrate.log`).

| criterion | value | pass |
|---|---|---|
| 1. `identical_to_lagunaLab` | false | no, see note |
| 2. `speedup_vs_lagunaLab` >= 2.0 | 0.396 | **no** |
| 3. `ctx_decode_ratio` >= 0.75 | 1.158 | yes |
| 4. peak GPU <= budget + resident + KV + 1 GB | 15.50 GB | not evaluated |

Note on 1: on 3 prompts x 128 tokens, outputs first diverge at tokens 19, 104 and 26, where the top-2 logit gap is 0 to 0.375 (about one bf16 step). Consistent with the bf16 near-tie explanation in lagunaLab's `streaming.json`; not verified per-step.

## Results (cold page cache, `results/bench_cold.json`, superseded: uncapped pool, Phase 1)

| | lagunaLab slots=32 | moeStream |
|---|---|---|
| decode tok/s | 14.37 | 4.39 |
| hit rate (decode) | 0.547 | 0.932 |
| peak GPU GB | 5.92 | 17.29 |
| long-ctx (16000) decode tok/s | 7.77 | 3.08 |

moeStream time split per token: sync 206.4 ms, read 7.9 ms, write 3.2 ms. `speedup_vs_lagunaLab` 0.306 (fail), `ctx_decode_ratio` 0.702 (fail, warm passes), `identical_to_lagunaLab` false.

## Speed gate (sub-project 0, `results/bench_slots*.json`, `results/bench_evalwrites.json`, superseded: uncapped pool)

The Phase 1 conclusion above ("build zero-copy Metal") was wrong. `sync_ms_per_tok` timed `mx.eval` on the lazy graph, so it included the slot writes queued before it. With writes forced into their own timer (`--eval-writes`) at the calibrated 128-256 slots and a 16.08 GB peak, writes took 140 ms per token, about 5.5 ms per fill. The same write path on a synthetic real-shaped layer costs 0.10-0.28 ms per fill at 32, 128 and 256 slots, so writes do not copy the slot tensor. The slowdown was memory pressure: swap was in use, and `budget.probe` counts inactive pages as free.

Each row below comes from a single run with its own lagunaLab baseline. The baseline varies 10.3-15.5 tok/s between runs, so compare within a row.

| run | moeStream tok/s | lagunaLab tok/s | ratio | 16k-ctx decode (moeStream / lagunaLab) | moeStream peak GB |
|---|---|---|---|---|---|
| `--slots 32` | 8.10 | 10.28 | 0.79 | 4.01 / 7.45 | 5.91 |
| `--eval-writes` (128-256, before reboot) | 4.12 | 11.15 | 0.37 | 2.57 / 7.07 | 16.08 |
| `--slots 128` | 12.25 | 13.65 | 0.90 | 11.58 / 8.45 | 12.80 |
| `--slots 128 --no-prefetch` | 14.58 | 15.51 | 0.94 | 11.40 / 7.96 | 12.80 |

At 128 slots and 12.8 GB, moeStream is 0.90-0.94x lagunaLab on short-prompt decode, 1.37-1.43x on 16k-context decode, and about 1.5x on prefill. Lookahead prefetch roughly breaks even. It hides about 10 ms of reads but adds about 18 ms of sync per token.

## Gate history

The gate is now the measured result in "Where it stands" above. The text below is how it got there.

The original gate ("decode >= lagunaLab at equal or lower memory" against lagunaLab at 32 slots) is replaced. New gate, at the 20 GB cap, with lagunaLab given the same slot memory (`tools.bench --cap-gb 20`):

1. Short-prompt decode within about 10% of lagunaLab.
2. 16k-context decode faster than lagunaLab.
3. 16k prefill faster than lagunaLab.

Data so far was measured at **unequal memory** (lagunaLab 32 slots, 5.9 GB; moeStream 128 slots, 12.8 GB), so it does not decide the gate: short-prompt decode 0.90x and 0.94x, 16k decode 1.37x and 1.43x, prefill 1.55x.

### Equal-memory verdict (2026-10-14)

**20 GB cap (`results/bench_cap20.log`): aborted, thrashing after about 35 min, no moeStream result.** lagunaLab's half finished but collapsed: 239 slots, 15.7 tok/s short decode, 16k prefill 51.9 tok/s, 16k decode **0.16 tok/s**, peak GPU 20.9 GB. moeStream then loaded 9280 slots (16.4 GB) and sat paging (process state stuck, about 8% CPU, 86M pageins, footprint about 18 GB) without tripping the watchdog (swap 2.4 GB, under its 8 GB limit), so it was killed by PID. Watchdog peaks for the run: swap 3510 MB, footprint 20480 MB. A 20 GB cap is not usable on this 24 GB machine.

**14.4 GB cap (`results/bench_cap14.4.json`), superseded: lagunaLab ran with its pool uncapped:**

| | lagunaLab (157 slots) | moeStream (128-256 slots per layer) | ratio |
|---|---|---|---|
| short-prompt decode | 14.09 tok/s | 12.68 tok/s | 0.90x |
| 16k prefill | 93.1 tok/s | 97.6 tok/s | 1.05x |
| 16k decode | 10.31 tok/s | 12.83 tok/s | 1.24x |
| peak GPU | 14.52 GB | 14.19 GB | |
| decode hit rate | 91.6% | 90.5% | |

Watchdog peaks for the whole run (both engines, one process): swap 160 MB, footprint 15360 MB.

Gate (superseded): passed at 14.4 GB, with short-prompt decode at the edge of item 1 (10.0% slower). Items 2 and 3 pass. The footprint is about 1.2 GB above moeStream's peak GPU: the prefetcher's staging buffers (numpy, up to 16 free expert-sized buffers per layer, plus those in flight) live outside the GPU budget and are not counted in `memory_gb`. moeStream decode time per token: sync 53 ms, read 8.7 ms, write 3.9 ms.

**14.4 GB cap, both buffer pools capped at 1 GB (`results/bench_cap14.4_pool1.json`, 2026-10-14):** lagunaLab found that MLX's uncapped free-buffer pool evicts expert files from the page cache (its 9c35d6e), which had handicapped its side of the run above. Re-run with `--cache-limit-gb 1` for both engines:

| | lagunaLab (157 slots) | moeStream (128-256 slots per layer) | ratio | uncapped ratio |
|---|---|---|---|---|
| short-prompt decode | 17.20 tok/s | 17.31 tok/s | 1.01x | 0.90x |
| 16k prefill | 131.6 tok/s | 106.9 tok/s | **0.81x** | 1.05x |
| 16k decode | 13.20 tok/s | 13.74 tok/s | 1.04x | 1.24x |
| peak GPU | 14.52 GB | 14.19 GB | | |
| decode hit rate | 91.6% | 90.5% | | |

Watchdog peaks: swap 533 MB, footprint 15360 MB. moeStream decode time per token: sync 39 ms, read 8.1 ms, write 2.6 ms.

Gate with both pools capped: **not met as written**. Short decode passes (1.01x), 16k decode is a small win (1.04x), 16k prefill loses (0.81x). The earlier long-context margins came mostly from lagunaLab's uncapped pool, not from moeStream. The pool cap helps both engines (moeStream short decode 12.68 to 17.31 tok/s), and is now moeStream's default (`mlx_cache_gb = 1.0`). Gate rewritten to this result (2026-10-14, the user): equal-memory parity plus the SSD tier, see "Where it stands".

Consequence (done: `memory_gb = 14.0`): the `memory_gb = 20.0` default in `moestream.toml` should come down to about 14. At 14 GB with q8 KV at 128k, the budget reserves KV for two sessions (2 x 2.85 GB), so the slots get 6.12 GB (64-128 per layer).


Sub-project 1 (`docs/specs/2026-10-13-engine-config-serve-design.md`) adds the explicit memory cap in place of the free-page probe.

## pi check (engine Task 6, 2026-10-14, `tools/verify_pi.sh`, `results/serve_verify.log`)

Served with `memory_gb = 14.0`, q8 KV, 128k context and tuning off, under the watchdog with a 4 GB swap and 17 GB footprint limit.

| run | slots | curl (512 tokens max) | peak footprint | peak swap |
|---|---|---|---|---|
| base (LFU, q8) | 3456 (6.12 GB) | 36 s, with reasoning | 11.0 GB | 289 MB |
| `eviction = "lru"` | 3456 | 42 s | 8.5 GB | 281 MB |
| `kv_bits = 0` | 624 (1.10 GB, 16 per layer) | 78 s | 3.9 GB | 273 MB |
| `thinking = false` | 3456 | 20 s, no reasoning, stopped by itself at 268 tokens | 8.5 GB | 273 MB |

The 3-turn pi session on a small repo (run the tests, fix the bug, add an empty-list test) completed with tool calls. pytest at the end shows 2 passed. The turns took 67, 21 and 71 s. Prefix caching works between turns: after the first 1388-token prompt, requests processed only 22-259 new tokens. bf16 KV at 128k leaves almost no room for slots, which is why q8 is the default.

## SSD tier for the prompt cache: simulation (2026-10-14, `tools/kv_tier_sim.py`, `results/kv_tier_sim.json`)

Question: when a session's KV is no longer in the RAM prompt cache (another session pushed it out, or
the server restarted), would restoring it from SSD beat re-processing the prompt?

Method: token-level bookkeeping, no model load. The 11 captured agent sessions (14-98 turns, 43k-83k
tokens at the end, 690 turns in all) were rendered through Laguna's chat template turn by turn, as pi
sends them. The RAM cache is mlx-lm's LRU prompt cache with `tools/serve.py`'s cap of one session of
q8 KV (131,072 tokens). The SSD tier holds each session's latest turn-end snapshot (q8, 20 GB cap). An
entry is reused only when it is an exact prefix of the new prompt; that held at every turn of every
session (no template-induced breaks). Times assume 100 tok/s prefill (generous at 60k+ context) and
2 GB/s SSD reads.

| Workload | RAM only: recomputed | RAM + SSD: recomputed | RAM only, est. prompt time | RAM + SSD |
|---|---|---|---|---|
| one session at a time | 0.46 M tokens | 0.46 M | 77 min | 77 min |
| one session, server restart midway | 0.51 M | 0.46 M | 85 min | 77 min |
| 2 sessions interleaved | 3.75 M | 0.46 M | 10.4 h | 78 min |
| 3 sessions interleaved | 12.87 M | 0.46 M | 35.8 h | 79 min |
| 3 interleaved, restart midway | 12.93 M | 0.46 M | 35.9 h | 79 min |

Totals are over all 11 sessions (29.3 M prompt tokens served). 0.46 M is the floor: first prompts plus
each turn's new tool output, which must be processed anyway. With the SSD tier every workload reaches
it.

- **One pi session at a time:** the RAM cache already reuses everything. The SSD tier only saves the
  re-prefill after a restart (one session's context, up to about 80k tokens, about 13 minutes).
- **Interleaved sessions** (parallel agents, harness evals, GRPO rollouts): the one-session RAM cap
  thrashes. Two sessions recompute 8x more prompt tokens, three sessions 28x more. The SSD tier
  brings both back to the floor.
- **Cost:** 60 GB written for the 11 sessions, about 87 MB per turn. Most of it is the sliding-window
  layers' state (63 MB, rewritten whole at every turn end); the full-attention KV is append-only.
  Live storage: about 1.6 GB per 75k-token session at q8.

Recommendation: **build it**, before the harness and GRPO work that will run sessions side by side.
For single-session pi use alone it would only save restarts. Spec first; mlx-lm's
`save_prompt_cache` and `load_prompt_cache` are the starting point.

## SSD tier on real weights (Task 18, 2026-10-14, `tools/kvtier_check.sh`, `results/kvtier/`)

Run 3, after two stopped attempts (`results/kvtier_run2/`: a driver bug, then a watchdog stop that led
to keeping two sliding snapshots per chain). A functional check, not a speed claim: `memory_gb = 12.0`,
watchdog 4 GB swap / 16 GB footprint while the machine was in use, `MOESTREAM_SHARED=1` (no wired
limit, 1 GB MLX buffer pool), `context_tokens = 24576` so RAM holds about one of the three sessions
(as 128k holds one 75k session in real use), thinking off, tuner off. Three captured sessions
(astropy-12907, django-10924, astropy-6938) interleaved turn by turn, keeping the model's own replies.

| Phase | Result |
|---|---|
| off vs on, same 15 turns | prompt tokens processed **135,132 vs 36,358** (the floor exactly: first prompts plus each turn's new tokens); 1478 s vs 519 s |
| restores during on | 12, of 5.5k-13k tokens, **0.15-0.73 s** each (prefill of the same tokens: about 55-130 s) |
| restart | fresh server: **3 of 3** sessions resumed from SSD (15,460 / 8,893 / 13,404 tokens in 0.28 / 0.72 / 0.76 s) |
| kill -9 mid-write | caught a `.tmp` file; nothing left running; the torn chain was dropped at startup; the next turn restored from the turn before (15,460 tokens, 0.27 s) |
| output equality | greedy 48-token continuation after an SSD restore equals the one after a RAM hit, **3 of 3** (tool-call ids are random per response and are ignored) |
| disk | 1.7 GB for 4 chains, under the 10 GB cap; per turn about 63 MB of sliding state plus 21.8 KB per new token |

Peaks per phase (watchdog footprint / largest MLX request peak): off 16.4 GB / 14.16 GB, on 14.3 / 12.55,
restart 13.3 / 11.42, kill9 13.3 / 11.10, equality 13.3 / 11.57; swap 0.42-0.48 GB throughout.

Memory attribution (`results/kvtier/mem.log`): MLX active memory at rest is 10.2-10.4 GB (slots
8.72 GB, resident 1.16 GB, KV and the RAM prompt cache). The per-request peak grows with the prompt
that is prefilled: 12.07 GB at 5.4k tokens, 12.79 at 8.9k, 13.62 at 12.1k, 14.16 at 15.2k, about 1.8 GB
plus 0.21 GB per 1000 tokens above active, against the budget's flat 1 GB for activations. The
footprint adds about 1.2 GB of staging buffers on top. With the tier on, long prefills (and these peaks)
mostly disappear. Not yet measured: whether the per-chunk attention scores (prefill chunk 2048 ×
context) are the transient; at 128k it would matter.

Findings: a reply cut off at `max_tokens` does not re-render as a prefix, so the next turn restores
from the snapshot before it (the second kept snapshot is what makes that work) and its snapshot starts
a new chain, rewriting that conversation's full-attention KV (about 370 MB here). Reusing the earlier
chain's segments for a branch would avoid the rewrite.

## Self-tuning on real weights (Task 10, 2026-10-14, `results/tune_*`)

The server tuned itself while serving captured agent sessions (`tools.drive`: turns up to about 4k
tokens, replies up to 1024 tokens, 5 s pauses), at `memory_gb = 14`, q8 KV, 128k context, under the
watchdog at 4 GB swap / 17 GB footprint. How it works: [docs/self-tuning.md](docs/self-tuning.md).

**Run 1 (13:45-15:23):** no change kept. A read-threads A/B trial gained a pair just often enough to
never time out, and the replay knobs waited behind it. Led to the fixes in `a23f39c`: replay knobs
first, live trials capped at 15 minutes, 32-token A/B windows, replay workers exit with the server.

**Run 2 (15:55-16:56, restart check 17:03-17:14), with those fixes:**

| Trial | Time | Result |
|---|---|---|
| prefetch off on 10 layers | 16:06 | rejected: faster in 6 of 8 pairs (7 needed) |
| eviction policy | 16:12 | no better candidate: LFU 15.0% simulated miss rate, LRU and FIFO not lower |
| half-life | 16:23 | no better candidate: 64 |
| read threads 8 to 16 | 16:31 | **kept**: faster in 8 of 8 pairs, median window time 0.86x |
| eviction policy (retry) | 16:45 | no better candidate |
| half-life (retry) | 16:56 | no better candidate |
| slot split | | never ran: the quality guard's noise check failed (`math_ok` false in `tuning/reference.npz`), so the only output-changing knob stayed off |

| | Run 2 first 10% | Run 2 last 10% | Run 2 all (17 windows) | Restart, same traffic as run 2's start |
|---|---|---|---|---|
| decode tok/s (median) | 9.40 | 7.58 | 7.81 | 9.27 (run 2 untuned on that traffic: 8.77) |
| hit rate | 0.36 | 0.43 | 0.43 | 0.43 (untuned: 0.45) |

Peak footprint 14.3 GB, swap flat at 241-249 MB, 0 failed requests. The restart loaded the saved
state (no fingerprint warning; read threads 16).

Gate: **not met as written.** Last-10% tok/s is below first-10%. But each 10% is 1-2 windows, and
the traffic changes as the sessions go on (later turns carry longer context), so start against end
does not isolate tuning. On the same traffic, the restarted, tuned engine ran 9.27 against 8.77 tok/s
untuned (+6%, 2 windows each). That points the right way but is within noise. The threads trial ran
16:16-16:31, overlapping CPU work in this session (16:15-16:25), so its 8-of-8 result may be partly
contention.

What limits it:
- Only about 17 stats windows an hour: prefill dominates replayed traffic, because recorded replies
  break prompt-cache reuse.
- The slot split, expected to matter most, is off on this machine: two runs of the untuned engine
  disagreed by more than half the strict KL thresholds, and the reference keeps only the verdict,
  not the numbers.

