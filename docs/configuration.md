# Configuration

All engine settings live in one flat file, `moestream.toml` at the repo root. `tools/serve.py` reads
it (or the file given with `--config`), and `load(EngineConfig)` sizes the engine from it. Relative
paths resolve against the file's own directory, so the server starts the same from any working
directory. Unknown keys and wrongly typed values are errors at startup, never silently ignored.

## Keys

| Key | Default | Meaning |
|---|---|---|
| `pack_dir` | required | lagunaLab's packed model directory |
| `memory_gb` | probe free memory | peak GPU memory the engine may use; resident weights, KV and activations come out of it first and the rest becomes expert slots |
| `context_tokens` | 131072 | the longest session the KV reservation is sized for; also pi's `contextWindow` |
| `kv_bits` | 8 | KV cache precision for the 10 full-attention layers: 8 = q8, 4 = q4, 0 = bf16 (lossless) |
| `eviction` | `"lfu"` | slot eviction policy: `lfu`, `lru` or `fifo` |
| `half_life` | from the calibration file, else 256 | LFU decay half-life, in tokens |
| `calibration` | none (uniform split) | `budget.json` from `tools.calibrate`: per-layer miss curves for the slot split, and the half-life |
| `thinking` | true | Laguna's built-in `<think>` reasoning (chat template `enable_thinking`) |
| `port` | 8081 | server port |
| `tune` | false | run the self-tuner; false serves with the saved tuning state and never trials. Off by default: its gate is not met yet ([self-tuning.md](self-tuning.md)) |
| `tuning_dir` | `"tuning"` | tuning state, log and quality reference |
| `trace_tokens` | 65536 | routing rows kept in memory per layer for replay trials |
| `kv_tier` | true | SSD tier for the prompt cache: turn-end KV snapshots on disk, restored instead of re-processing a prompt ([serving.md](serving.md#ssd-tier)) |
| `kv_dir` | `"kvcache"` | the tier's directory (git-ignored); a non-empty directory without the tier's `FORMAT` file is refused, never wiped |
| `kv_disk_gb` | 10.0 | disk cap; above it whole conversations are deleted, least recently used first |
| `lookahead` | false | next-layer lookahead prefetch while decoding. Off since 2026-10-15: at 32k / 64 slots it made decode ~11% slower (186.5 vs 166.0 ms per token) and run-to-run timing noisy, and it was never measured as a win (docs/long-context.md). true turns it on (the tuner can then turn it off per layer) |
| `kv_sessions` | 1 | sessions of `context_tokens` KV the memory budget reserves. 1 (since 2026-10-15): the live session; other conversations are held by the SSD tier (`kv_tier`) and restored in about a second. 2: the live one plus one stored conversation in RAM, for `kv_tier = false` or short contexts. Load warns if it is 1 with the tier off |
| `prompt_cache_entries` | 6 | conversations mlx-lm's RAM prompt cache keeps (its `--prompt-cache-size`; its byte cap is one session of KV either way); 1 leaves the rest to the SSD tier |
| `release_evicted_caches` | false | empty a cache entry's arrays when the RAM prompt cache drops it, so a reference mlx-lm still holds (its batched path's end-of-prompt cache) does not keep a full KV copy; logs a line at startup when on. Opt-in, for bf16 KV only (`kv_bits = 0`): it let an 81k session fit at `memory_gb = 12` (docs/long-context.md) |
| `wired_limit` | false | let mlx-lm wire the whole recommended working set (it does so at startup and around generation). Off since 2026-10-15: the wired memory stalled video playback on the shared laptop. true opts in on a dedicated machine; the server prints the state at startup |
| `mlx_cache_gb` | 1.0 | cap, in GiB (2^30 bytes, as lagunaLab's `cache_limit_gb` and every measured run), on MLX's pool of freed buffers kept for reuse; 0 = MLX's default (no cap). Uncapped, the pool grows past `memory_gb` and pushes expert files out of the page cache; capped, decode is faster too (short decode 12.7 to 17.3 tok/s at 14.4 GB, README) |

Environment variables for the server: `MOESTREAM_SHARED=1` keeps mlx-lm's wired-memory limit off
even if `wired_limit = true` (off is the default since 2026-10-15; the pool cap applies either way); `MOESTREAM_MEMLOG=<file>`
appends MLX memory (active, cache, peak) per request.

The shipped file:

```toml
pack_dir = "../lagunaLab/models/Laguna-XS-2.1-packed"
memory_gb = 14.0
context_tokens = 131072
kv_bits = 8
eviction = "lfu"
calibration = "budget.json"
thinking = true
port = 8081
tune = false
tuning_dir = "tuning"
trace_tokens = 65536
kv_tier = true
kv_dir = "kvcache"
kv_disk_gb = 10.0
mlx_cache_gb = 1.0
lookahead = false
kv_sessions = 1
```

## What fits: worked examples

Fixed costs: resident weights 1.16 GB and the prefill activation reserve, **1.3 GB + 0.021 GB per 1k
tokens of `context_tokens`** (1.99 GB at 32k, 4.05 GB at 128k; measured, docs/prefill-memory.md).
KV is reserved for `kv_sessions` sessions of `context_tokens` (default one: the live session; the SSD
tier holds the others). The load banner prints both. One slot is 1.77 MB; there are 39 MoE layers with a
floor of 16 slots each (624 slots, 1.10 GB).

| `memory_gb` | `context_tokens` | `kv_bits` | `kv_sessions` | KV reserved | Slots | Notes |
|---|---|---|---|---|---|---|
| 14 | 131072 | 8 | 1 | 2.85 GB | 3356 (5.94 GB) | **the default**; other conversations live in the SSD tier |
| 14 | 131072 | 8 | 2 | 5.70 GB | 1744 (3.09 GB) | one more conversation in RAM (tier off) |
| 14 | 131072 | 0 | 2 | 10.74 GB | load fails | bf16 KV at 128k with two sessions does not fit |
| 14 | 131072 | 0 | 1 | 5.37 GB | 1934 (3.42 GB) | bf16: faster decode at long context; see `release_evicted_caches` |
| 14 | 32768 | 8 | 2 | 1.43 GB | 5329 (9.43 GB) | shorter sessions: room for a second one in RAM |
| 20 | 131072 | 8 | 2 | 5.70 GB | | **thrashes on a 24 GB Mac**: do not use |

Before 2026-10-15 the reserve was a flat 1 GB with two KV sessions, which gave the default 6.12 GB of slots; the measured
81k session at that setting used 13.0 GB of footprint (docs/long-context.md).

Rules of thumb:

- **Peak footprint runs 1.5-3.5 GB above MLX's own peak** (MLX's buffer pool and the prefetcher's
  staging buffers are outside the budget). Keep it at least 6-8 GB under physical memory, or macOS starts paging and decode collapses.
  On the 24 GB machine, 14 GB is measured safe with a browser open; 20 GB is not.
- **q8 KV is the right trade at long context.** At 128k, bf16 KV costs 5 GB more than q8, all of
  which comes out of the expert cache.
- **More slots raise the hit rate with diminishing returns.** The calibration curves
  (`budget.json`) show the miss rate per layer at 16 to 256 slots; most layers flatten past 128.
- If `memory_gb` is more than macOS reports free at load, the engine warns and continues.
- If the fixed costs alone exceed `memory_gb`, load fails and says by how much.

## Changing settings

Every key takes effect at server start. Restart the server after editing the file.

With `tune = true` (off by default), the tuner may change `eviction`, `half_life` and the slot split while serving, and
saves what it learned in `tuning/state.json`. **Saved tuning state wins over the TOML** for those three
settings on the next start, as long as the pack, `memory_gb`, `kv_bits` and `context_tokens` are
unchanged. To make a TOML edit of `eviction` or `half_life` stick, delete `tuning/state.json` (or set
`tune = false`, which still loads the saved state). Changing `memory_gb`, `kv_bits` or `context_tokens`
invalidates the saved state automatically.
