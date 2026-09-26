# moeStream: Pipelined MoE Expert Streaming on Apple Silicon

## Goal

Run Laguna XS 2.1 (4-bit, 256 experts × 39 MoE layers, 16 GB of expert weights) on a
24 GB M4 Mac at close to resident decode speed, using whatever unified memory is free at
load time for an adaptive expert cache, and holding that speed across long agent sessions.

This is a sibling project to `lagunaLab`. It reads `lagunaLab` and never writes to it.

## Why: measured bottlenecks (throwaway spike, 2026-09-26)

Per decode token, `lagunaLab` streaming (`laguna_mlx/streaming.py`), 48 tokens, warm page cache:

| | 16 slots | 128 slots |
|---|---|---|
| tok/s | 11 | 14–22 |
| misses / 312 expert uses per token | ~165 | 27–40 |
| GPU sync wait (39 syncs) | 31–43 ms | 29–40 ms |
| `pread` | 22–27 ms | 5–23 ms |
| bytes → `mx.array` | 11 ms | 2–3 ms |
| slot writes (when forced separately) | ~22 ms | ~5 ms |

Resident model: ~25 ms/token (40 tok/s).

Conclusions:
1. Loading is serial with GPU compute. Each layer syncs, then loads, then computes.
   The load time (10–55 ms) is fully exposed. **Overlap is the main win.**
2. Each missed expert is copied three times (pread → bytes, bytes → `mx.array`, → slot).
   MLX offers no zero-copy wrap from Python (`mx.array`, `mx.from_dlpack` both copy).
3. Slot-write cost does not scale with slot count (0.30 ms/write at N=16 and N=128).
4. More slots help once warm (128 slots: 87–91% hits, 22 tok/s). LFU without decay
   lets old context pin experts, which hurts long sessions (16k-context decode: 4.4 tok/s).

## Dependencies and constraints

- Imports `lagunaLab` through a pytest/python path entry: `laguna_mlx.laguna` (model) and
  `scripts.pack_experts` (`layout.json` format). Reads `lagunaLab/models/Laguna-XS-2.1-packed`
  and `lagunaLab/traces`. Built against `lagunaLab` commit `7a4ae65` (branch `part1-harness`).
  `lagunaLab` is owned by another session and is read-only here, including `models/` and `results/`.
- Shared laptop: before any real-weight model load, message the other Claude sessions and
  wait for their go-ahead. No loads while `lagunaLab/scripts/overnight.sh` runs.
- Output must be bit-identical to the resident model (same bits, same weights).

## Components

```
moeStream/
  moestream/store.py      # SlotStore: per-layer slots, decayed-LFU eviction
  moestream/prefetch.py   # Prefetcher: background preadv into reusable staging buffers
  moestream/layer.py      # StreamingMoE: replaces SparseMoeBlock, one sync per layer
  moestream/budget.py     # free-memory probe, per-layer slot allocation
  moestream/load.py       # load(pack_dir, budget_gb=None) -> model, tokenizer
  scripts/record.py       # record per-layer routing over lagunaLab traces -> data/routing/
  scripts/calibrate.py    # offline miss curves -> budget.json, decay half-life
  scripts/bench.py        # tok/s, hit rate, prediction accuracy, time breakdown
  tests/
```

- `SlotStore` has no threads. `Prefetcher` has no eviction policy. `budget.py` is pure
  functions. Each unit is tested on its own.
- `StreamingMoE` replaces the whole `SparseMoeBlock`, not just `SwitchGLU`. It needs the
  router input to run the next layer's router in the same sync.

## Data flow (decode, MoE layer N)

1. Queue attention N and router N. Also queue **layer N+1's router applied to layer N's
   MoE input** (the prediction). One `mx.eval` returns actual indices for N and predicted
   indices for N+1. The sync count is the same as today.
2. Collect prefetch futures for layer N (issued at N−1). Wait only on unfinished ones.
   Read experts the prediction missed right away, in parallel on the pool.
3. Choose eviction victims (decayed LFU, never an expert in the current set), write slots,
   and queue `gather_qmm` for N.
4. Submit background reads for predicted N+1 experts not already resident.

**Prefetch never evicts.** Predicted experts wait in staging buffers (top_k × stride,
about 14 MB per layer, two layers in flight) until a layer actually uses them. So a wrong
prediction costs disk bandwidth, never a good slot.

**Threading.** Worker threads only run `os.preadv` into preallocated numpy buffers
(which releases the GIL). All MLX graph construction stays on the main thread. Slot writes
are functional MLX updates. A slot tensor is written only at its own layer's step, when no
queued op still reads the old contents.

**Copies per miss:** 3 → 2 (preadv into a staging buffer, then `mx.array` + slot update).
Getting to one copy needs zero-copy (see Phase 2).

**Decayed LFU.** Per layer, `score *= decay` once per token, `score[e] += 1` per use,
and the lowest score is evicted with recency as the tie-break.
`decay = 0.5 ** (1 / half_life)`. The default half-life is 256 tokens; `calibrate.py` tunes it.

**Prefill.** Same multi-pass grouping as `lagunaLab` (chunks that need more experts than
there are slots run several masked passes). No prefetch.

## Memory budget

Probe at load:

```
available   = (free + inactive + speculative + purgeable pages) * page_size   # vm_stat
              - headroom                                                     # default 3 GB
gpu_cap     = max_recommended_working_set_size   # mx.device_info() or mx.metal.device_info(), whichever this MLX has
              - resident weights - KV reserve(target_ctx) - activation reserve (1 GB)
slot_budget = min(available, gpu_cap)          # --budget-gb overrides
```

- KV reserve: full-attention layers cost 40 KB/token. The default target context is 32k (~1.3 GB).
- Log the chosen budget and slot counts at load, so a run is explainable when another job holds memory.
- `--wire`: opt-in `mx.set_wired_limit(slot_bytes)` so macOS never compresses or swaps slots.
  Off by default, because wired memory would starve other sessions' jobs.

**Per-layer split.** `calibrate.py` replays recorded routing through the decayed-LFU
policy for each layer and slot count (16…256) → miss curves. Greedy marginal
allocation gives the next slot to the layer with the largest miss reduction, with a floor of 16 per layer.
Output: `budget.json` (fractions per layer). With no file, or a layer mismatch, fall back to
a uniform split with a warning.

`record.py` needs one streaming-model load (~5 GB, coordinated). Recordings are stored in
`data/routing/`. Calibration and Laya scoring are then offline.

## Error handling

- Missing or truncated expert file, or layout mismatch: raise at load with the path. Never run on partial weights.
- A failed background read falls back to a synchronous read on the main thread. A second failure raises.
- Invalid `budget.json`: uniform split plus a warning.

## Testing

Tiny random model (`lagunaLab/tests/tiny.py`) packed to a temp dir, no real weights:

- **Exactness:** greedy decode and chunked prefill outputs equal the resident model bit for bit,
  at minimum budget, at full budget, with prediction forced always-right, and with
  prediction forced always-wrong.
- **Eviction:** never evicts a current-set expert; old counts decay by the half-life; empty slots are used first.
- **Budget:** greedy allocation respects the floor, sums to the total, and favors steep
  curves; the memory probe parses fixed `vm_stat` output.
- **Prefetch:** predicted experts take no slot until used; staging buffers are reused, not grown.

## Benchmarks (`scripts/bench.py`, real weights, coordinated loads)

Compared: `lagunaLab` streaming at 32 slots, moeStream at the auto budget, and resident if memory allows.
Metrics: decode tok/s (short prompt and after a 16k prefill), hit rate, prediction
accuracy (fraction of layer N+1's actual experts that were predicted), per-token time
breakdown (sync wait / read / copy / compute), and peak GPU memory. Warm vs cold page cache (cold =
evict by reading unrelated files; purging would need sudo). Results go in
`results/*.json`, and every number in any writeup comes from there.

## Success criteria

1. Output identical to resident.
2. Decode ≥ 2× `lagunaLab` streaming at 32 slots (baseline ~11 tok/s).
3. Decode at 16k context within 25% of short-prompt decode.
4. Peak GPU memory within the reported budget.

## Future work (not in scope)

- **Laya predictor.** Score Laya (non-autoregressive universal classifier) offline on
  `data/routing/` as an expert predictor against the router-lookahead baseline. Wire it in
  only if it is more accurate and its prediction is ready before the layer needs it.
- **Phase 2: zero-copy C++/Metal extension.** mmap the expert files with
  `newBufferWithBytesNoCopy` (strides are already 16 KB page-aligned), do the slot lookup
  on the GPU, and use `madvise` to prefetch. Build only if the Phase 1 time breakdown still shows
  load/copy time exposed.
- Runtime slot shrink on macOS memory-pressure notifications, added if collisions with other jobs actually happen.
- Predicting layer 1 of the next token (needs the next token's embedding).
