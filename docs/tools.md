# Tools

All tools run from the repo root with `PYTHONPATH=.:../lagunaLab`. Tools marked **GPU** load real
weights: run them through the GPU lock and the watchdog
(`/Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe tools/watch.sh 4 17 <command>`) and tell
the coordinator first ([development.md](development.md)).

| Tool | GPU | What it does |
|---|---|---|
| `python3 -m tools.serve` | yes | the server ([serving.md](serving.md)) |
| `python3 -m tools.record` | yes | records per-layer routing over lagunaLab's captured sessions |
| `python3 -m tools.calibrate` | no | turns recorded routing into miss curves and a half-life (`budget.json`) |
| `python3 -m tools.bench` | yes | moeStream against lagunaLab's streaming loader |
| `python3 -m tools.drive` | no (client) | replays captured sessions against a running server |
| `tools/verify_pi.sh` | yes | end-to-end server and pi check |
| `tools/watch.sh` | wrapper | memory watchdog around any command |

## record and calibrate: the offline slot split

```bash
python3 -m tools.record [--max-tokens 4096] [--budget-gb 5]      # -> data/routing/
python3 -m tools.calibrate [--routing data/routing] [--out budget.json]
```

`record` teacher-forces every session in `../lagunaLab/traces` through the model (prefill gives the
same routing as decode, token by token) and saves each layer's expert ids per token.

`calibrate` needs no model. For each layer it replays the routing through `SlotStore` at 16, 32, 48,
64, 96, 128, 192 and 256 slots and each half-life (64 to infinite), each session starting with an
empty cache. It writes the half-life with the lowest overall miss rate and each layer's miss curve.
`load()` splits the slot total across layers greedily: each step of slots goes to the layer whose
miss rate drops most per slot. The self-tuner later re-derives this split from live routing.

## bench

```bash
python3 -m tools.bench [--cap-gb X] [--slots N] [--budget-gb X] [--cold] [--no-prefetch] [--eval-writes] [--no-base]
```

Measures greedy decode tok/s (prefill excluded), hit rate, peak GPU memory, decode at a 16k-token
context and 16k prefill, for lagunaLab's streaming loader and moeStream, one model at a time, and
writes `results/bench*.json`.

| Flag | Effect |
|---|---|
| `--cap-gb X` | equal memory: both engines get the slot bytes left under a total cap of X GB (KV sized as bf16 at 32k) |
| `--slots N` | force N slots per layer |
| `--cold` | purge the page cache first |
| `--no-prefetch` | turn lookahead off |
| `--eval-writes` | time slot writes on their own instead of inside the next sync (diagnostic) |
| `--no-base` | skip lagunaLab |

The baseline itself varies 10-15 tok/s between runs, so compare within one run.

## drive

```bash
python3 -m tools.drive [--minutes 60] [--max-chars 16000] [--max-tokens 1024] [--pause 5] [--url http://127.0.0.1:8081/v1]
```

A load generator for the self-tuning verification. It reads lagunaLab's captured agent sessions
(`replay.session.load_session`, read-only) and sends one chat request per recorded assistant turn,
with everything before that turn as the prompt.

- `--max-chars` skips turns whose prompt is longer (16,000 characters is about 4k tokens), so decode
  rather than prefill dominates.
- `--pause` leaves the server idle between turns, as tool execution would; the tuner works only
  when idle.
- Tool-call arguments are converted to JSON strings (the sessions store them as objects).
- A failed turn is printed with its HTTP status and error body; the drive goes on.

Replayed sessions get little prompt-cache reuse, because the recorded assistant turns differ from
what the model generates, so each request re-processes most of its prompt.

## watch.sh

```bash
tools/watch.sh <max_swap_gb> <max_footprint_gb> <command...>
```

Runs the command, polls every 2 s, and kills it if system swap or the command's footprint (from
`top`, which counts Metal buffers) passes its limit. It prints `aborted: paging (...)` and exits 3
when it trips, and `watch: peak swap N MB, peak footprint N MB` when the command ends either way.
Swap is system-wide: other apps' swap counts.
