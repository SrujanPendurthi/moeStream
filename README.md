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

## Next

The gate as written ("decode >= lagunaLab at equal or lower memory") fails. moeStream trades memory for long-context speed. Rewording the gate is the user's decision. Sub-project 1 (`docs/specs/2026-10-13-engine-config-serve-design.md`) adds an explicit memory cap in place of the free-page probe.
