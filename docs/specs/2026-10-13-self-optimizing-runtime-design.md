# moeStream: self-optimizing runtime (sub-project 4, system level)

Date: 2026-10-13
Status: draft for review

## Goal

The runtime gets faster the longer it serves, without changing what the model outputs beyond a
strict bound. It measures itself while serving, tries one change at a time, keeps a change only if
it measured better and passed the quality guard, and saves what it learned so the next start
begins where the last one stopped.

You asked for self-optimization at three levels: system, bit and kernel. They share one tuner,
one quality guard and one state file, all built here. This spec builds the **system level** on top
of them. Bit and kernel levels are later specs that plug into the same tuner; the constraints
already known for them are at the end.

Done means: on a long served workload, decode tok/s and hit rate at the end are above the
untuned start, no accepted change failed the quality guard, and a restart resumes from the saved
state instead of from defaults.

## Not in scope

Bit level and kernel level (later specs, constraints below). DFlash speculative decoding: it is
not wired into the runtime yet, so its draft length is not a knob yet. Pinned prefix KV and pinned
experts for the harness's system prompt: they need the harness (sub-project 2). Any write to
`lagunaLab/`.

## Where it runs

Inside the `tools/serve.py` process, which already holds the GPU lock for its lifetime (it runs
under `gpu-run moe`), so tuning adds no new GPU jobs.

mlx-lm 0.31.3 runs all generation on one thread (`ResponseGenerator._generate`), which polls
`_next_request(timeout=0.1)` while idle. Two hooks, both on that thread, so nothing touches the
model concurrently:

- **Per token:** the last `StreamingMoE` layer calls `tuner.on_token()` after each decode step.
  It closes measurement windows and applies a pending change between tokens.
- **Idle:** `serve.py` wraps `_next_request`. When it returns no request, the wrapper runs one
  slice of idle work (replay or quality probe), each slice bounded to about 2 s, so a new request
  waits at most one slice.

CPU-only idle work (trace replay) runs at most 2 worker processes at low priority, so it does not
starve lagunaLab or the user.

## The tuner

New `moestream/tune.py`. A knob is:

```python
class Knob:
    name: str
    tier: str                      # "replay": judged by simulation; "live": judged by A/B timing
    def get(engine): ...
    def propose(engine, telemetry): ...   # a candidate value, or None
    def apply(engine, value): ...         # between tokens; must not exceed memory_gb
```

Loop, one trial at a time, knobs taken round-robin:

1. `propose` gives a candidate from current measurements (rules per knob, below).
2. **Replay knobs** are judged offline: the candidate and the current value are both replayed
   through `SlotStore` over the live routing traces, as `tools/calibrate.py` does. The candidate
   goes on only if its simulated miss rate is lower by at least 1% relative.
3. **Live knobs** are judged by paired A/B timing on real traffic: alternate 64-token decode
   windows, current then candidate, within the same request so context length is matched. After
   8 pairs, keep the candidate only if it was faster in at least 7 of 8 pairs (sign test,
   p ≈ 0.035). Fewer than 8 pairs when traffic stops: no decision, retry later.
4. **Quality guard** (below) runs on every candidate that can change the math before it is kept.
5. Keep or revert, append the trial to the log, save state.

Re-proposal: a knob whose candidate was rejected is not retried until 50k more tokens have been
served, or the hit rate has moved more than 5 points since the rejection (workload changed).

## System-level knobs (this spec)

| knob | tier | proposal rule | changes math? |
|---|---|---|---|
| eviction policy (`lfu`/`lru`/`fifo`) | replay | each other policy in turn | no |
| LFU half-life | replay | neighbours of current on the `HALF_LIVES` grid | no |
| per-layer slot split | replay | `allocate()` on miss curves from live traces | prefill only (see guard) |
| lookahead prefetch per layer | live | turn off layers whose measured prediction hits save less read time than their extra sync costs; turn back on where the reverse holds | no |
| read threads (`_POOL`) | live | ×2 / ÷2 within 2–16 | no |

Measured at 128 slots, prefetch hides about 10 ms of reads per token and adds about 18 ms of
sync, so it roughly breaks even overall. Per layer it will not; this knob settles the open
"drop or fix prefetch" question by measurement.

**Changing the slot split live.** Slot tensors are fixed-shape today. New
`StreamingMoE.resize(n)`: allocate the layer's slot arrays at `n`, copy the experts it keeps (the
`n` best by the current eviction key), rebuild `SlotStore` bookkeeping. Memory spike is one
layer's arrays. A split change shrinks layers before growing others, so the total never exceeds
the cap.

**Memory pressure (a guard, not a trial).** Every 256 tokens, read free memory (`free_bytes()`)
and swap used (`sysctl vm.swapusage`). If swap has grown or free memory is under 1 GB, shrink the
largest layers by 10% of total slots with `resize`, and log it. Grow back toward the tuned split
when free memory has stayed above 3 GB for 5 minutes. Peak never exceeds `memory_gb`. This is the
fix for the 17x slot-write slowdown seen under paging.

## Quality guard: strict KL

Your decision: strict threshold. It compares against a **fixed reference**, never against the
previous tuned state, so accepted changes cannot drift step by step.

- **Probe set:** `scripts.check_parity.PROMPTS` plus fixed slices of lagunaLab's captured traces
  (read-only), teacher-forced, 2048 tokens total.
- **Reference:** the probe run once with the untuned engine (shipped defaults, same pack, same
  `kv_bits`). Stored per position: top-64 token ids and their log-probabilities, plus the
  log-sum-exp. Cached in the tuning directory, keyed by pack and probe set.
- **Measure:** KL(reference ‖ candidate) on the reference's top-64 support, renormalized.
- **Pass:** mean KL ≤ 1e-3 nats/token, 99th percentile ≤ 1e-2, and top-1 token agreement
  ≥ 99.5%.
- **Noise floor:** the reference is run twice at first build. If the two disagree past half the
  threshold, the guard is not trustworthy on this machine: tuning of math-changing knobs is
  disabled and the log says why.

Run in idle slices, about 2 s each. Knobs that cannot change the math (table above) skip it; a
tiny-model test proves each of those is bit-identical. The slot split can change prefill math:
when a chunk routes to more experts than a layer has slots, `StreamingMoE` sums expert groups,
and a different slot count changes the grouping and so the float summation order.

## State and log

Config gains three keys:

```toml
tune = true                 # false = serve with the saved state, never trial
tuning_dir = "tuning"       # state, log, live traces, reference; relative to this file
trace_tokens = 65536        # routing kept in memory for replay (~20 MB)
```

- `tuning/state.json`: current knob values plus a fingerprint (pack layout hash, `memory_gb`,
  `kv_bits`, `context_tokens`). Loaded after `load()` and applied. Fingerprint mismatch: ignored
  with a warning, tuning restarts from defaults. A slot split that no longer fits the cap is
  dropped the same way.
- `tuning/log.jsonl`: one line per trial (time, knob, from, to, A/B or replay numbers, KL, kept or
  reverted) and per pressure action. The control UI (sub-project 3) reads it.
- Live routing traces: a ring of the last `trace_tokens` decode tokens per layer, using the
  existing `io.trace` path. Not written to disk.

## Errors

- A knob's `apply` raises: revert to the previous value, log it, disable that knob until restart.
- State file unreadable: warn, start from defaults, keep serving.
- Quality probe fails mid-run (for example, an out-of-memory error): treat as a failed guard,
  revert.
- Tuning never stops serving: every tuner error path logs and continues untuned.

## Tests

Tiny model, no real weights, in the existing style:

- `SlotStore`: policy and half-life change live, and bookkeeping stays consistent.
- `resize`: shrinking and growing keep the chosen experts resident, and decode output is
  bit-identical before and after.
- Eviction policy, half-life, prefetch and read threads: each change leaves decode output
  bit-identical.
- Tuner: a replay candidate is kept only with ≥1% lower simulated miss rate; a live candidate is
  kept at 7 of 8 faster pairs and reverted at 6 of 8 (fake timer); a rejected knob is not retried
  early.
- KL guard: on synthetic logits, identical gives 0 and passes; a perturbed distribution over the
  threshold fails; noise above half the threshold disables math-changing knobs.
- State: round trip; fingerprint mismatch is ignored; an oversized split is dropped.
- Pressure: a fake `free_bytes` under 1 GB shrinks slots and never raises total above the cap.

## Verification on real weights

Through the GPU lock, under `tools/watch.sh 8 21`, after sub-project 1's Task 6.

1. Build the reference; record the noise floor.
2. Serve a long workload: lagunaLab's captured traces replayed against the server as sessions,
   about 1 hour. Record tok/s and hit rate per 256-token window, every trial, peak memory and swap.
3. Pass: final 10% of windows beats the first 10% on median tok/s, with hit rate not lower; every
   kept change passed its guard; peak ≤ `memory_gb`; no swap growth.
4. Restart the server: it loads the state and the first windows match the end of the previous run
   (within 5%), not the untuned start.

## Later levels (own specs; known constraints)

**Bit level.** Per-expert precision: rarely used experts at lower bits, so more experts fit in the
cap and each miss reads fewer bytes. Known now:
- The packed experts are already 4-bit, and there is no bf16 source on disk (`lagunaLab/models/Laguna-XS-2.1`
  holds only the config). Hot experts **cannot go above 4-bit** without downloading the bf16 model
  (about 66 GB). Disk has 24 GB free. So bit level means demoting cold experts to 3 or 2 bits,
  re-quantized from 4-bit.
- One `SwitchGLU` has one precision. Mixed precision in a layer needs a second, low-bit slot pool
  per layer and a second masked gather, which costs compute.
- Every demotion goes through the strict KL guard above. Under a strict bound, many demotions may
  fail; the tuner will report how many pass.

**Kernel level.** Autotuned Metal kernels (`mx.fast.metal_kernel`) for the real decode shapes:
fused route plus gather, gather-matmul tiling, dequant. Variants benchmarked in idle time and
checked by the KL guard. Starts with a per-op profile of a decode token: the last measurements
point at sync and I/O, not compute, so kernel work goes ahead only where the profile shows compute
dominates.
