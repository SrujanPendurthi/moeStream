# Architecture

## The problem

Laguna-XS-2.1 is 18.9 GB at 4-bit: 1.16 GB of resident weights plus 17.7 GB of routed experts. A 24 GB
Mac also has to hold macOS, the KV cache, activations and whatever else the user runs. The experts do
not fit, but each token uses only 8 of 256 experts per layer, and routing is skewed: a few experts
per layer take most of the traffic, and the set in use drifts slowly over a session. So moeStream keeps
a fixed number of expert *slots* per layer in GPU memory and treats them as a cache over the experts on
SSD.

## Pieces

| Module | Responsibility |
|---|---|
| `moestream/load.py` | builds the model: resident weights from `resident.safetensors`, one `StreamingMoE` per MoE layer, slot counts from the memory budget, optional quantized KV |
| `moestream/layer.py` | `StreamingMoE`: route, fill missing experts into slots, run the slots, prefetch the next layer's guess |
| `moestream/store.py` | `SlotStore`: which expert sits in which slot, and which slot to evict (LFU with decay, LRU or FIFO) |
| `moestream/prefetch.py` | `Prefetcher`: background `preadv` of expert bytes into reusable staging buffers |
| `moestream/budget.py` | memory arithmetic: slot bytes under a cap, KV bytes per token, the per-layer slot split |
| `moestream/config.py` | `EngineConfig` from `moestream.toml` |
| `moestream/tune.py`, `knobs.py`, `guard.py`, `pressure.py` | the self-tuning runtime ([self-tuning.md](self-tuning.md)) |
| `tools/serve.py` | the OpenAI-compatible server ([serving.md](serving.md)) |

## The pack on disk

moeStream reads lagunaLab's packed model (`scripts.pack_experts`), never the original checkpoint:

```
Laguna-XS-2.1-packed/
  config.json, tokenizer files, generation_config.json
  resident.safetensors         everything except routed experts (1.16 GB)
  layout.json                  num_experts, stride, layers, and where each array sits inside an expert record
  experts/layer_01.bin ...     one file per MoE layer: 256 fixed-size records of 1,769,472 bytes
```

Each record holds one expert's gate, up and down projections (4-bit weights plus scales and biases). A
miss is one `preadv` of one record at `expert * stride`: no seeking inside the file, no decompression.

## One MoE layer, one step

`StreamingMoE.__call__` (`moestream/layer.py`):

1. **Route.** The router picks 8 experts per token. During decode it also runs the *next* layer's
   router on this layer's input as a guess of what the next layer will want (lookahead).
2. **Sync.** One `mx.eval` brings this layer's routing (and the guess) to the CPU. This is the only
   GPU sync per layer.
3. **Assign.** `SlotStore.assign` marks hits, picks victims for misses by the eviction key, and returns
   the fills to do.
4. **Fill.** For each miss, take the expert's bytes from the prefetcher (already read if the guess was
   right, read now if not), write them into the chosen slot of the layer's `SwitchGLU` arrays, and
   return the staging buffer.
5. **Compute.** A gather-matmul over the slots, using a table from expert id to slot.
6. **Prefetch.** Submit the next layer's guessed experts that are not resident to the read pool, so
   their bytes arrive while the GPU computes.

**Prefill** routes many tokens at once. If a chunk needs more distinct experts than the layer has
slots, the layer runs in groups of `slots` experts and sums the masked results. That is exact, but the
grouping depends on the slot count, so a different slot split changes the floating-point summation
order. This is why the slot split is the one tunable setting that can change model output and must
pass the quality guard ([self-tuning.md](self-tuning.md)).

**Slot arrays** start as zeros built from a one-expert template (`_glu`). Building them through
`SwitchGLU`'s own initializer would allocate a float32 random tensor per expert, about 1.6 GB of
transient memory for 128 slots.

## Eviction

`SlotStore.key(e)` ranks resident experts; the lowest is evicted.

| Policy | Key | Behaviour |
|---|---|---|
| `lfu` (default) | (decayed use count, last use) | Each use adds the number of tokens routed to the expert; every step multiplies all counts by `0.5 ** (1 / half_life)`. Experts that were hot long ago stop pinning slots. Ties go to the least recent. |
| `lru` | last use | Evicts the expert unused for longest. |
| `fifo` | load time | Evicts the expert loaded earliest. |

Decay is per token: a prefill chunk of n tokens ages every count by n tokens' worth. Offline
calibration chose a half-life of 64 tokens for the captured agent sessions.

## Memory budget

`budget.cap_budget_bytes`:

```
slot bytes = memory_gb - resident (1.16 GB) - KV - activations (1 GB)
KV         = kv_bytes_per_token(kv_bits) x 2 x context_tokens
```

KV is reserved for **two** sessions: the live one plus one stored conversation in mlx-lm's prompt
cache, which `tools/serve.py` caps at one session of KV. Slot bytes divided by the 1.77 MB stride is
the slot total, split across the 39 layers by `budget.allocate` from the calibration file's miss
curves (or evenly without one), with a floor of 16 slots per layer. Worked examples are in
[configuration.md](configuration.md).

Outside the budget: the prefetcher's staging buffers (numpy, up to 16 free expert-sized buffers per
layer plus those in flight), about 1.2 GB measured. Peak process footprint is therefore about
`memory_gb + 1.2 GB`.

## Why the numbers look the way they do

- **Sync dominates decode.** At 128 slots, about 53 ms per token is the per-layer sync, against
  9 ms of reads and 4 ms of slot writes. Most of the sync is GPU compute the lazy graph runs when
  the routing is read.
- **Memory pressure is the cliff.** Once macOS pages, slot writes slow down about 17x and decode
  collapses (16k-context decode fell to 0.16 tok/s at a 20 GB cap). The engine's memory is fixed at
  load, so the cap must leave room for the OS and the user's apps; see the 14 GB default.
- **Prefetch roughly breaks even.** It hides about 10 ms of reads per token but adds about 18 ms of
  sync for the extra router run. The self-tuner decides per layer whether to keep it.
