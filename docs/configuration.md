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
| `tune` | true | run the self-tuner; false serves with the saved tuning state and never trials |
| `tuning_dir` | `"tuning"` | tuning state, log and quality reference |
| `trace_tokens` | 65536 | routing rows kept in memory per layer for replay trials |

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
tune = true
tuning_dir = "tuning"
trace_tokens = 65536
```

## What fits: worked examples

Fixed costs: resident weights 1.16 GB, activations 1 GB. KV is reserved for two sessions of
`context_tokens` (the live one and one stored in the prompt cache). One slot is 1.77 MB; there are 39
MoE layers with a floor of 16 slots each (624 slots, 1.10 GB).

| `memory_gb` | `context_tokens` | `kv_bits` | KV reserved | Slots | Per layer | Notes |
|---|---|---|---|---|---|---|
| 14 | 131072 | 8 | 5.70 GB | 3456 (6.12 GB) | 64-128 | the default; verified with pi |
| 14 | 131072 | 0 | 10.74 GB | 624 (floor) | 16 | bf16 KV at 128k leaves no room; decode about half as fast |
| 14 | 32768 | 8 | 1.43 GB | about 5900 | up to 256 | shorter sessions, more experts cached |
| 20 | 131072 | 8 | 5.70 GB | about 6860 | | **thrashes on a 24 GB Mac**: do not use |

Rules of thumb:

- **Peak footprint is about `memory_gb + 1.2 GB`** (the prefetcher's staging buffers are outside the
  budget). Keep it at least 6-8 GB under physical memory, or macOS starts paging and decode collapses.
  On the 24 GB machine, 14 GB is measured safe with a browser open; 20 GB is not.
- **q8 KV is the right trade at long context.** At 128k, bf16 KV costs 5 GB more than q8, all of
  which comes out of the expert cache.
- **More slots raise the hit rate with diminishing returns.** The calibration curves
  (`budget.json`) show the miss rate per layer at 16 to 256 slots; most layers flatten past 128.
- If `memory_gb` is more than macOS reports free at load, the engine warns and continues.
- If the fixed costs alone exceed `memory_gb`, load fails and says by how much.

## Changing settings

Every key takes effect at server start. Restart the server after editing the file.

With `tune = true`, the tuner may change `eviction`, `half_life` and the slot split while serving, and
saves what it learned in `tuning/state.json`. **Saved tuning state wins over the TOML** for those three
settings on the next start, as long as the pack, `memory_gb`, `kv_bits` and `context_tokens` are
unchanged. To make a TOML edit of `eviction` or `half_life` stick, delete `tuning/state.json` (or set
`tune = false`, which still loads the saved state). Changing `memory_gb`, `kv_bits` or `context_tokens`
invalidates the saved state automatically.
