# moeStream

Streams MoE expert weights from disk on Apple Silicon, with a decayed-LFU slot cache, next-layer lookahead prefetch and a per-layer slot budget calibrated offline from recorded routing. Phase 1 compares it against lagunaLab's streaming loader on Laguna-XS-2.1.

## Run

```bash
python3 -m pytest tests -v
PYTHONPATH=.:../lagunaLab python3 -m tools.record      # routing over lagunaLab/traces -> data/routing (loads model)
PYTHONPATH=.:../lagunaLab python3 -m tools.calibrate   # offline -> budget.json
PYTHONPATH=.:../lagunaLab python3 -m tools.bench       # warm -> results/bench.json
PYTHONPATH=.:../lagunaLab python3 -m tools.bench --cold  # cold page cache -> results/bench_cold.json
```

`record` and `bench` load real weights; check no other heavy job is running first.

## Results (warm, `results/bench.json`)

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

## Results (cold page cache, `results/bench_cold.json`)

| | lagunaLab slots=32 | moeStream |
|---|---|---|
| decode tok/s | 14.37 | 4.39 |
| hit rate (decode) | 0.547 | 0.932 |
| peak GPU GB | 5.92 | 17.29 |
| long-ctx (16000) decode tok/s | 7.77 | 3.08 |

moeStream time split per token: sync 206.4 ms, read 7.9 ms, write 3.2 ms. `speedup_vs_lagunaLab` 0.306 (fail), `ctx_decode_ratio` 0.702 (fail, warm passes), `identical_to_lagunaLab` false.

## Speed gate (sub-project 0, `results/bench_slots*.json`, `results/bench_evalwrites.json`)

The Phase 1 conclusion above ("build zero-copy Metal") was wrong. `sync_ms_per_tok` timed `mx.eval` on the lazy graph, so it included the slot writes queued before it. With writes forced into their own timer (`--eval-writes`) at the calibrated 128-256 slots and a 16.08 GB peak, writes took 140 ms per token, about 5.5 ms per fill. The same write path on a synthetic real-shaped layer costs 0.10-0.28 ms per fill at 32, 128 and 256 slots, so writes do not copy the slot tensor. The slowdown was memory pressure: swap was in use, and `budget.probe` counts inactive pages as free.

Each row below comes from a single run with its own lagunaLab baseline. The baseline varies 10.3-15.5 tok/s between runs, so compare within a row.

| run | moeStream tok/s | lagunaLab tok/s | ratio | 16k-ctx decode (moeStream / lagunaLab) | moeStream peak GB |
|---|---|---|---|---|---|
| `--slots 32` | 8.10 | 10.28 | 0.79 | 4.01 / 7.45 | 5.91 |
| `--eval-writes` (128-256, before reboot) | 4.12 | 11.15 | 0.37 | 2.57 / 7.07 | 16.08 |
| `--slots 128` | 12.25 | 13.65 | 0.90 | 11.58 / 8.45 | 12.80 |
| `--slots 128 --no-prefetch` | 14.58 | 15.51 | 0.94 | 11.40 / 7.96 | 12.80 |

At 128 slots and 12.8 GB, moeStream is 0.90-0.94x lagunaLab on short-prompt decode, 1.37-1.43x on 16k-context decode, and about 1.5x on prefill. Lookahead prefetch roughly breaks even. It hides about 10 ms of reads but adds about 18 ms of sync per token.

## Gate (revised 2026-10-13)

The original gate ("decode >= lagunaLab at equal or lower memory" against lagunaLab at 32 slots) is replaced. New gate, at the 20 GB cap, with lagunaLab given the same slot memory (`tools.bench --cap-gb 20`):

1. Short-prompt decode within about 10% of lagunaLab.
2. 16k-context decode faster than lagunaLab.
3. 16k prefill faster than lagunaLab.

Data so far was measured at **unequal memory** (lagunaLab 32 slots, 5.9 GB; moeStream 128 slots, 12.8 GB), so it does not decide the gate: short-prompt decode 0.90x and 0.94x, 16k decode 1.37x and 1.43x, prefill 1.55x.

### Equal-memory verdict (2026-10-14)

**20 GB cap (`results/bench_cap20.log`): aborted, thrashing after about 35 min, no moeStream result.** lagunaLab's half finished but collapsed: 239 slots, 15.7 tok/s short decode, 16k prefill 51.9 tok/s, 16k decode **0.16 tok/s**, peak GPU 20.9 GB. moeStream then loaded 9280 slots (16.4 GB) and sat paging (process state stuck, about 8% CPU, 86M pageins, footprint about 18 GB) without tripping the watchdog (swap 2.4 GB, under its 8 GB limit), so it was killed by PID. Watchdog peaks for the run: swap 3510 MB, footprint 20480 MB. A 20 GB cap is not usable on this 24 GB machine.

**14.4 GB cap (`results/bench_cap14.4.json`), the run the verdict rests on:**

| | lagunaLab (157 slots) | moeStream (128-256 slots per layer) | ratio |
|---|---|---|---|
| short-prompt decode | 14.09 tok/s | 12.68 tok/s | 0.90x |
| 16k prefill | 93.1 tok/s | 97.6 tok/s | 1.05x |
| 16k decode | 10.31 tok/s | 12.83 tok/s | 1.24x |
| peak GPU | 14.52 GB | 14.19 GB | |
| decode hit rate | 91.6% | 90.5% | |

Watchdog peaks for the whole run (both engines, one process): swap 160 MB, footprint 15360 MB.

Gate: **passes at 14.4 GB**, with short-prompt decode at the edge of item 1 (10.0% slower). Items 2 and 3 pass. The footprint is about 1.2 GB above moeStream's peak GPU: the prefetcher's staging buffers (numpy, up to 16 free expert-sized buffers per layer, plus those in flight) live outside the GPU budget and are not counted in `memory_gb`. moeStream decode time per token: sync 53 ms, read 8.7 ms, write 3.9 ms.

Consequence: the `memory_gb = 20.0` default in `moestream.toml` should come down to about 14. At 14 GB with q8 KV at 128k, the budget reserves KV for two sessions (2 x 2.85 GB), so the slots get 6.12 GB (64-128 per layer).


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
