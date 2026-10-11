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

Note on 1: on 3 prompts x 128 tokens, outputs first diverge at tokens 19, 104 and 26, where the top-2 logit gap is 0 to 0.375 (about one bf16 step). Consistent with the bf16 near-tie explanation in lagunaLab's `streaming.json`; not verified per-step. Cold-cache results: pending.

## Next

`sync_ms_per_tok` dominates (145 ms vs 10 ms read, 4 ms write), and hit rate is already 0.90, so better prediction (Laya) will not close the gap. Build Phase 2: zero-copy Metal.
