# Development

## Layout

```
moestream/          the engine (see architecture.md)
tools/              serve, record, calibrate, bench, drive, verify_pi.sh, watch.sh
tests/              tiny-model tests: no real weights
moestream.toml      engine config
budget.json         calibration (slot curves, half-life)
results/            benchmark and verification outputs, committed
tuning/             runtime tuning state (git-ignored)
docs/               these pages, specs, plans
```

moeStream depends on a sibling checkout of lagunaLab (`../lagunaLab`) for the Laguna model code, the
packed weights and the captured sessions. It never writes there.

## Tests

```bash
MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q
```

`MOESTREAM_CPU=1` is part of the habitual command but nothing reads it; the tests run on MLX's
default device. They build a tiny Laguna (a few layers, a handful of
experts), pack it with lagunaLab's packer into a temp directory, and check behaviour against it:
bit-identical decode under every output-preserving knob, eviction victims, budget arithmetic,
config validation, tuner decisions with fake clocks and inline worker pools, guard verdicts on
synthetic logits, pressure shrink and restore, state round trips. About 25 s.

Real-weight checks are not tests: they are the tools in [tools.md](tools.md), and their outputs are
committed under `results/`.

Every change is test-first: write the test, watch it fail for the right reason, make it pass, run
the whole suite.

## The shared machine

The 24 GB Mac is shared with lagunaLab and with the user. The rules
(`/Users/srujanpendurthi/Projects/Current/.gpu/RULES.md`):

- **Every real-weight job** goes through `.gpu/gpu-run moe` (one heavy job at a time; lagunaLab has
  priority and can queue ahead) and `tools/watch.sh` (4 GB swap / 17 GB footprint while the user is
  on the machine; 8 / 21 otherwise).
- **Tell the coordinator session before each real run** and wait for a go.
- **Kill by PID,** never `pkill -f`; the watchdog uses `kill -0`. After a run, check with `ps` that no
  server, drive or worker process is left.
- If the watchdog trips, record it; do not retry with more memory.
- **Never write into `../lagunaLab`.**

## What may be imported from lagunaLab

Only these interfaces are frozen for moeStream's use:

| Import | Used for |
|---|---|
| `laguna_mlx.laguna` | model classes, router |
| `laguna_mlx.streaming` | the baseline in `tools.bench` |
| `laguna_mlx.dflash.generate_greedy` | reference greedy decode |
| `laguna_mlx.cache.make_quantized_cache`, `DequantKVCache` | quantized KV |
| `scripts.pack_experts` | the pack format (`expert_from_bytes`) |
| `scripts.check_parity.PROMPTS` | the guard's probe prompts |
| `scripts.serve_streaming` | the SSE keepalive filter |
| `replay.session.load_session` | captured sessions for `tools.drive` |

KV cache rule: never add a `bits` attribute to a cache object and never use mlx-lm's raw
`QuantizedKVCache`; read a cache's precision through `kv_bits`.

## Branches and records

- Work happens on feature branches (currently `engine`); `main` holds merged sub-projects.
- Specs (`docs/specs/`) are the authority; plans (`docs/superpowers/plans/`) implement them task by
  task.
- Plan execution keeps a ledger under `.superpowers/sdd/<plan>/progress.md` (git-ignored): task
  completions and every decision that departed from the plan, with its cost if wrong.
- Commit messages say what changed and why; results files are committed with the commit that
  reports them.
