# Self-tuning runtime

The server measures itself while it serves, tries one change at a time, keeps a change only if it
measured better and passed the quality guard, and saves what it learned so the next start resumes
from there. This page covers the **system level** (cache and I/O settings). Bit-level (expert
precision, on-disk encoding) and kernel-level (Metal kernels) tuning are planned to plug into the same
tuner; see the spec, `docs/specs/2026-10-13-self-optimizing-runtime-design.md`.

## Where it runs

Everything runs on mlx-lm's single generation thread, so nothing touches the model concurrently:

- **Per decode token:** the last MoE layer's hook calls `Tuner.on_token()` (timing windows, stats)
  and `Pressure.on_token()`. A prefill step calls `Tuner.interrupt()` instead.
- **Idle:** when the generation thread polls an empty request queue, `Tuner.idle()` runs one bounded
  slice of work: start a trial, collect a replay result, step the quality guard. A new request waits
  at most one slice.
- **Replay simulations** run in 2 worker processes at nice 10. Workers exit on their own if the
  server dies.

Consequence: **the tuner only works when the server has idle moments.** Interactive traffic (pi, the
harness) pauses while tools run; back-to-back batch traffic would never be tuned.

Any error inside tuning turns tuning off and serving continues. It never ends the generation thread.

## Knobs

| Knob | Judged by | Candidates | Can change output? |
|---|---|---|---|
| `policy` (eviction) | replay | the other two of `lfu`, `lru`, `fifo` | no |
| `half_life` | replay | every value on the grid 64, 128, 256, 512, 1024, infinite | no |
| `slots` (per-layer split) | replay, then the quality guard | `allocate()` on miss curves from live routing, same total | yes, in prefill (summation order) |
| `prefetch` (lookahead per layer) | live A/B | flip one quarter of the layers | no |
| `threads` (read pool size) | live A/B | double or halve, within 2-16 | no |

Each knob that cannot change output has a tiny-model test proving decode is bit-identical before and
after the change.

## How a trial is decided

One trial at a time. Each round goes through the knobs in turn, **replay knobs first** (they need no
traffic), then the live ones.

**Replay trials.** The last `trace_tokens` routing rows of every layer (decode and prefill) are kept
in a ring buffer in memory. A worker replays them through `SlotStore` with the current setting and
with each candidate, the same simulation `tools/calibrate.py` runs offline. The best candidate is
kept only if its simulated miss rate is at least 1% lower (relative). Replay needs at least 1024
rows per layer.

**Live A/B trials.** Real decode speed, compared on real traffic, inside the same request so the
context length matches:

- After any break in the token stream (a prefill or an idle slice), the first 32 tokens only re-warm
  the cache and are not measured.
- Then 32-token windows alternate between the current value and the candidate, in A B B A order so
  neither arm always goes first. A break drops any half-finished pair.
- After 8 pairs: keep the candidate only if it was faster in at least 7 (a sign test, p about 0.035).
- No decision (and the old value back) if no pair completes for 10 minutes, or if the trial has run
  15 minutes in all.

**Re-proposal.** A rejected knob is not tried again until 50,000 more tokens have been served, or the
hit rate has moved by more than 5 points since the rejection (the workload changed). A trial ended
with no decision can be retried on the knob's next turn.

## Quality guard

Only the slot split can change the model's output, so only it goes through the guard. The guard
compares the engine against a **fixed reference** built once from the untuned engine, so accepted
changes cannot drift step by step.

- **Probe:** the built-in parity prompts (`scripts.check_parity.PROMPTS`) followed by the untuned
  engine's own greedy continuation, 2048 tokens in all, teacher-forced.
- **Reference:** per position, the top-64 token ids and log-probabilities, stored in
  `tuning/reference.npz`, keyed by the pack layout, `kv_bits` and the probe.
- **Measure:** KL(reference ‖ candidate) on the reference's top-64, renormalized; and top-1 agreement.
- **Pass:** mean KL at most 1e-3 nats per token, 99th percentile at most 1e-2, top-1 agreement at
  least 99.5%.
- **Noise floor:** at build time the probe runs a second time on the same engine. If the two runs
  disagree by more than half the thresholds, the guard cannot be trusted on this machine and the slot
  knob stays off (`math_ok` false in the reference file).

A slot candidate goes live before its guard run finishes; if the guard fails, the old split comes
back.

## Memory pressure

Not a trial: a guard that always runs. Every 256 decode tokens it reads free memory and swap use.

- **Shrink** when swap has grown or free memory is under 1 GB: take 10% of all slots off the largest
  layers (never below 16 per layer). An in-flight slot trial is cancelled first.
- **Restore** the pre-shrink split once free memory has stayed above 3 GB for 5 minutes.
- While shrunk, the slot knob is paused, and the tuner reads and writes the restore target, so a
  shrunk split is never saved.

Swap growth from other processes also triggers a shrink. On a busy shared machine that can cost
slots that were not the cause.

## Files

| File | Contents |
|---|---|
| `tuning/state.json` | current knob values and a fingerprint (pack layout, `memory_gb`, `kv_bits`, `context_tokens`); loaded at start; ignored with a warning if the fingerprint differs |
| `tuning/log.jsonl` | one line per stats window, trial and pressure action (below) |
| `tuning/reference.npz` | the quality guard's reference; rebuilt if missing, unreadable or for another key |

`tuning/` is git-ignored. State and reference are written atomically.

## Reading the log

```json
{"time": ..., "tok_s": 9.4, "hit_rate": 0.36, "tokens": 257}
{"time": ..., "knob": "policy", "from": "lfu", "to": "lru", "kept": true, "miss_rate": 0.220, "candidate_miss_rate": 0.205}
{"time": ..., "knob": "prefetch", "from": [...], "to": [...], "kept": false, "pairs": [[3.97, 2.56], ...]}
{"time": ..., "knob": "threads", "cancelled": "ab timeout", "pairs": 7}
{"time": ..., "pressure": "shrink", "free_gb": 0.8, "swap_gb": 1.2, "slots": 3110}
```

- Stats windows: decode tok/s and hit rate over each 256 decode tokens.
- `pairs`: (current seconds, candidate seconds) per window pair; lower is faster.
- `kl_mean`, `kl_p99`, `top1`, `ok`: guard results on slot trials.

A quick summary:

```bash
python3 -c "
import json
for l in open('tuning/log.jsonl'):
    r = json.loads(l)
    if 'knob' in r or 'pressure' in r: print(r)
    else: print('  %.1f tok/s, hit %.2f' % (r['tok_s'], r['hit_rate']))"
```

## Settings

`tune`, `tuning_dir` and `trace_tokens` in `moestream.toml` ([configuration.md](configuration.md)).
`trace_tokens = 65536` keeps 41 MB of routing in all (a 2 x 65,536-row buffer of 8 expert ids per
layer, 39 layers). Saved tuning state overrides the
TOML's `eviction` and `half_life` on the next start.

## Real-weight runs

**2026-10-14, run 1** (60 minutes of recorded agent sessions, then a restart): no change kept. A
read-threads A/B trial gained a pair just often enough to never time out, and the replay knobs
waited behind it for 47 minutes. That led to the replay-first order, the 15-minute cap and the
32-token windows above. Logs: `results/tune_*.log`, `results/tune_log_run1.jsonl`.

**Run 2** with those fixes is in progress; its results will be added here and to the top-level
README.

## Known limits

- No tuning under continuous traffic (idle-only by design).
- Live A/B on short agent replies is slow to decide: a pair needs about 160 tokens of uninterrupted
  decode in one request.
- The guard's reference key omits `memory_gb`, and the guard's own forward passes are counted in the
  hit statistics.
