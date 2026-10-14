# moeStream engine: config, eviction policies, server, pi (sub-project 1)

Date: 2026-10-13
Status: draft for review

## Goal

Laguna-XS-2.1 on moeStream, usable inside the pi harness, with every engine setting in one
file: how much memory the engine may use, context size, KV cache quantization, the expert
eviction policy and whether the model thinks before answering. Whatever memory is left after the fixed costs is filled with the experts in use.

Done means: a multi-turn pi session with tool calls completes against the moeStream server, and
changing a setting in the file changes what the engine does.

This is sub-project 1 of the roadmap (speed gate, **engine**, parity eval, control UI,
self-tuning, refine loop). It assumes sub-project 0's revised gate passes: at the 20 GB cap,
with lagunaLab given the same slot memory, moeStream is within about 10% on short-prompt decode
and faster on 16k-context decode and prefill (see README). If a fix changes how slots are
stored, only `layer.py` internals move, not anything below.

## Not in scope

The web UI and terminal UI (sub-project 3; they will read and write the same file), self-tuning,
the refine loop, the parity eval, any change to weights, and any write to `lagunaLab/`.

## Config: one file, one object

New `moestream/config.py` with an `EngineConfig` dataclass and `EngineConfig.read(path)`. The
file is `moestream.toml` at the repo root, read with stdlib `tomllib`. Writing the file from code
is added with the UI, which is the first thing that needs it.

```toml
pack_dir = "../lagunaLab/models/Laguna-XS-2.1-packed"
memory_gb = 20.0          # peak GPU memory the engine may use; omit to probe free memory
context_tokens = 32768    # longest session the KV reservation is sized for
kv_bits = 0               # 0 = bf16 (lossless), 8 or 4 = quantized
eviction = "lfu"          # lfu | lru | fifo
half_life = 64            # lfu only; omit to take it from the calibration file
calibration = "budget.json"
thinking = true           # Laguna's built-in <think> reasoning; the model's own default is on
port = 8081
```

`read` rejects unknown keys, `kv_bits` outside {0, 4, 8} and `eviction` outside the three names,
with the bad value in the message. `load()` in `moestream/load.py` gains `load(config)`; its
current keyword form stays for the tools and tests that use it.

## Memory: fill the cap with experts

`slot_budget_bytes` in `moestream/budget.py` already gives the slots whatever is left after the
fixed costs. Two changes:

- **The cap comes from the config.** With `memory_gb` set, slot bytes =
  `memory_gb - resident - kv_bytes(context_tokens, kv_bits) - activations (1 GB)`. With it
  omitted, the existing `vm_stat` probe is the cap. If `memory_gb` is more than the probe says is
  free, load warns and continues. If the fixed costs alone exceed `memory_gb`, load raises with
  each term in the message.
- **KV bytes depend on `kv_bits`.** New `kv_bytes_per_token(bits)`: 40,960 at bf16, times 136/256
  at 8-bit, times 72/256 at 4-bit (stored sizes from lagunaLab's KV compression spec).

**Default cap: 20 GB** (your decision, relayed by the coordinator session; lagunaLab's
watchdog uses the same 20 GB footprint cap). On this 24 GB machine that leaves little for the OS
and other jobs. Evidence from the speed gate (sub-project 0):

| run | slots | peak GPU | decode tok/s | slot write per fill |
|---|---|---|---|---|
| shared machine, swap in use | 128-256 (probe) | 16.1 GB | 4.12 | 5.5 ms |
| fresh machine after reboot | 128 uniform | 12.8 GB | 13.91 | 0.33 ms |

The second row is the known-good operating point. Above it, slot writes slow down about 17x
once macOS starts paging, so the load-time warning (probe says less than `memory_gb` is free)
prints both numbers and this operating point.

The split of those slots across layers is unchanged: `allocate` from the calibration file, or
`uniform` without one.

## KV quantization

`load` replaces the model's cache factory when `kv_bits` is 8 or 4, using lagunaLab's
`laguna_mlx.cache.make_quantized_cache`: K/V stored at `kv_bits` on the 10 full-attention layers
(`DequantKVCache`, attends in bf16 through the fused kernel), `RotatingKVCache` unchanged on the
sliding ones. Not raw mlx-lm `QuantizedKVCache`: lagunaLab found its quantized attention builds
the full score matrix (~16 GB per layer at 82k context) and hung the machine. The cache objects
must never expose a `bits` attribute, since mlx-lm routes those to that attention path.
`kv_bits = 0` leaves `laguna.Model.make_cache` as it is.

Not known yet: decode speed with `QuantizedKVCache` at long context, and its quality cost on
Laguna. lagunaLab's Part 3 plan measures quality; this spec measures speed in its verification
and reports both settings, it does not promise either is free.

## Eviction policies

`SlotStore` (`moestream/store.py`) takes `policy`. `_victims` already sorts candidate slots by a
key; the policy picks the key.

| policy | evicts | key |
|---|---|---|
| `lfu` | lowest decayed use count, oldest use breaks ties (today's behaviour) | `(score, last)` |
| `lru` | least recently used | `last` |
| `fifo` | longest resident | `loaded`, a new per-expert array set to the clock when an expert is filled |

LIFO is left out. It evicts the most recently loaded expert, which is the one routing just asked
for, so every token would reload its own experts.

Offline calibration (`tools/calibrate.py`) simulates LFU only. With `lru` or `fifo` the per-layer
split from `budget.json` is still used but was tuned for LFU; the load banner says so.

## Server

New `tools/serve.py`, run as `PYTHONPATH=.:../lagunaLab python3 -m tools.serve [--config moestream.toml]`.
It is mlx-lm's OpenAI-compatible server with the loader swapped, the same shape as
`lagunaLab/scripts/serve_streaming.py`:

- Importing `scripts.serve_streaming` installs lagunaLab's keepalive filter on the handler; that
  is reused by import.
- The loader calls `moestream.load.load(config)` and adds the extra EOS ids from
  `generation_config.json` (id 24, `</assistant>`), as lagunaLab's loader does.
- Defaults passed to mlx-lm: `--port` from config, `--max-tokens 4096`, `--prompt-cache-size 6`,
  `--prompt-cache-bytes` equal to one session of KV, and `--decode-concurrency 1
  --prompt-concurrency 1`, since the budget reserves KV for one client.
- mlx-lm 0.31.3 builds its prompt cache without a byte limit and applies `--prompt-cache-bytes`
  only on its batched path, which quantized KV never takes. `serve.py` patches the cache class so
  stored conversation caches are capped at one session of KV. The served budget therefore reserves
  KV for two sessions: the live one plus the stored copies.
- The engine loads once. mlx-lm reloads whenever a request names a different model (pi sends
  `Laguna-XS-2.1`); the loader returns the already-loaded engine instead of loading a second one.
- `--chat-template-args '{"enable_thinking": <thinking>}'`. Laguna's chat template already
  implements reasoning (`<think>` blocks, `reasoning_content` carried across turns) and mlx-lm's
  server already separates it from the answer; nothing new is built. A request that sends its own
  `chat_template_kwargs` still overrides the default.
- Sampling from the pack's `generation_config.json` (`--temp 1.0 --top-k 20 --top-p 1.0`), which
  are the settings the model ships with for thinking. Phase 1's benchmarks stay greedy.

`context_tokens` is not enforced by the server in this version. pi is told the same number as its
context window and compacts before reaching it. A client that ignores it can exceed the cap.

## pi

One provider added to `~/.pi/agent/models.json`, next to the existing `llamacpp` entry:

```json
"moestream": {
  "baseUrl": "http://127.0.0.1:8081/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "compat": { "supportsDeveloperRole": false },
  "models": [{ "id": "Laguna-XS-2.1", "reasoning": true, "contextWindow": 32768, "maxTokens": 4096 }]
}
```

`contextWindow` must match `context_tokens`, and `reasoning` must match `thinking` so pi keeps the
model's reasoning between turns. The file is outside the repo: the edit is shown for
confirmation before it is made, and the README gains the snippet.

## Errors

- Bad config value: raise at read, naming the key and value.
- Fixed costs over the cap: raise at load, listing resident, KV, activations and the cap in GB.
- Slot budget under the 16-slot floor: the existing warning stays.
- A failed expert read already retries once and then raises (`moestream/prefetch.py`); unchanged.

## Tests

In the existing style, on the tiny model, no real weights:

- `test_config.py`: a file reads into the expected object; `thinking` defaults to true; unknown key, bad `kv_bits` and bad `eviction` each raise.
- `test_store.py`: for one access sequence, `lfu`, `lru` and `fifo` each evict the expert the
  table above says.
- `test_budget.py`: `kv_bytes_per_token` at 0, 8 and 4 bits; lower `kv_bits` gives more slot
  bytes; fixed costs over the cap raise.
- `test_layer.py` or `test_bench.py`: tiny model loaded from an `EngineConfig` with `kv_bits = 8`
  generates, and its cache list holds `QuantizedKVCache` on full-attention layers only.

## Verification on real weights

One heavy job at a time; check nothing else is running first.

1. Start the server, `curl` one chat completion.
2. Run a multi-turn pi session with tool calls on a small repo. Record from the server log:
   cached tokens growing turn over turn, decode tok/s, hit rate, peak GPU.
3. Peak GPU is at or under `memory_gb`. If it is over, the 1 GB activations term is wrong and is
   corrected from the measurement.
4. Repeat step 2's first turns with `eviction = "lru"`, then `kv_bits = 8`; the load banner and
   the stats change accordingly. Report tok/s for each.
5. Repeat with `thinking = false`: the response carries no reasoning. Report output tokens and
   wall time per turn for both. Whether thinking improves task results is the parity eval's
   question (sub-project 2), not this one's.
