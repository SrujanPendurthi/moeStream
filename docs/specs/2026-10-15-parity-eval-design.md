# Parity eval: does the moeStream engine behave like the reference on agent tasks?

Status: approved 2026-10-15; all runs done 2026-10-16, every gate passed (results at the end). Sub-project 2 of the roadmap, first part: before the custom
harness is built, establish that moeStream's engine reproduces the reference model, not only token
by token but in the actions an agent takes. Docs only; the implementation plan follows approval.

## Question

Two separate questions, kept apart because they have different gates:

1. **Engine parity (should be near-lossless).** On the *same weights*, does moeStream (streamed
   experts, slot cache, q8 KV, its prefill path) produce the same next-token distribution and the
   same actions as lagunaLab's resident model? Any gap here is an engine bug.
2. **Model parity (expected lossy, measured).** Against the unpruned reference, how close is what
   moeStream ships? moeStream ships the full pack, so this should be near-lossless too; the pruned-50
   pack is measured alongside because lagunaLab's references and noise floors exist for it.

## What is reused (read-only, from lagunaLab; nothing written there)

Documented in `lagunaLab/docs/parity-harness.md`:

- `replay.session.load_session`, `replay.session.tokenize`: captured `pool` agent sessions in
  `lagunaLab/traces/*.jsonl`, tokenized with the chat template (`enable_thinking=True`).
- `replay.replay.score(model, ref_path, chunk=2048, make_cache=..., forward=None)`: teacher-forced
  top-64 KL and top-1 agreement at every assistant position, weighted by assistant tokens.
- `replay.replay.greedy_actions`, `score_actions`, `scripts/diag_actions.py`
  (`--base/--refs/--out`; `--out` always a path in this repo): greedy action match at sampled
  assistant-turn starts, exact and same-tool.
- References on disk:

| refs | built from | sessions | length | actions |
|---|---|---|---|---|
| `refs/kv/` | pruned-50 resident, plain cache | 5 django | full (42,771-82,671) | 0 |
| `refs/prune/` | full 4-bit base | 5 django (+2 astropy) | 16,384 | 0 |
| `refs/prune_actions/` | full 4-bit base | 5 django | 16,384 | 4 each, 20 total |

**Sessions:** the five django sessions only (held out; the astropy sessions calibrated pruning).
Assistant positions per session 18,397-44,204.

## Comparisons

| id | engine under test | reference | kind | gate |
|---|---|---|---|---|
| E1 | moeStream, **pruned-50 pack**, q8 KV, chunk 2048 | `refs/kv` (pruned-50 resident, plain cache), full length | teacher-forced | KL top-64 mean <= 0.02 **and** top-1 >= 0.98 |
| E2 | as E1 with bf16 KV (`kv_bits = 0`) | `refs/kv` | teacher-forced | top-1 >= 0.995, KL <= 0.005 (streaming is exact in lagunaLab: top-1 1.0, KL 0; only summation order differs) |
| M1 | moeStream, **full pack** (shipped), q8 KV | `refs/prune` (full 4-bit base), 16k | teacher-forced | KL <= 0.02 and top-1 >= 0.98 |
| A1 | moeStream, full pack, q8 KV, lookahead off (shipped) | `refs/prune_actions` (20 actions, 16k prefixes) | greedy actions | see "Action gates" |
| A0 | moeStream, full pack, q8 KV | references moeStream builds itself (`build_ref`, same 20 starts) | greedy actions, noise floor | reported, no gate |
| T1 | moeStream server vs lagunaLab streaming server (`scripts.serve_streaming`), same full pack | pi on 3 small tasks | task outcome | see "Task outcomes" |

Metrics per run: `kl_top64_mean`, `kl_top64_p99`, `top1_agree` (assistant-token weighted over
sessions), `peak_mem_gb`, `tok_per_s`; for actions: exact match, same tool (text between
`<tool_call>` and the next `<`), common-prefix length, and **tool-call validity** (below).

### Tool-call validity (new metric, computed here)

For each generated action that contains a tool call: it parses (mlx-lm's tool parser for Laguna),
the tool name exists in the session's `tools`, and its arguments are valid JSON matching that tool's
parameter schema (required keys present, types right). Reported as valid / total. **Gate: every
generated tool call is valid** wherever the reference action is a valid tool call. A malformed call
from an engine bug is the failure an agent harness cannot recover from.

### Action gates (A1)

Exact match is noisy even for an exact engine: bf16 near-ties (0-4 ulps) flip greedy argmax when
summation order changes, and a flipped token changes the rest of the action. Known points: the base
against its own references 20/20 exact, 20/20 same tool; pruned-50 6/20 exact, 18/20 same tool.

Proposed gates for the shipped engine (full pack):

- **same tool >= 19/20**, and
- **exact >= 14/20**, and
- every mismatch read and classified: *tie divergence* (the first differing token had a top-2 logit
  gap within 4 bf16 ulps in the reference) or *real*. **No real mismatch** in a tool name or argument
  structure.

A0 (moeStream against references it built) gives the engine's own floor; if A0 is itself below
20/20, the exact threshold is lowered to A0's count minus 2 and the change is written down.

### Task outcomes (T1, last, informational)

Three small pi tasks from the repos already captured (one-file fixes), each run once per server
with the shipped sampling (temperature 1.0, top-k 20). Recorded: completed or not, turns, tool-call
errors, wall time. **No pass/fail on completion** (three sampled runs cannot separate engines); the
gate is engine-attributable failure only: **zero malformed tool calls and zero server errors** on
moeStream. A task lagunaLab completes and moeStream does not gets its transcript read.

## Engine contract and settings

- **Loading:** `moestream.load.load(pack, memory_gb=..., kv_bits=..., context_tokens=...)`. moeStream's
  model is Laguna's `Model` with streamed MoE blocks, so `score`'s default path (`model.model` then
  `lm_head` on assistant positions) and `mlx_lm.generate` (actions, through `model.make_cache`) both
  work without a `forward=` callable.
- **Pruned pack:** 128 experts per layer. `budget.json` was calibrated on the full 256; load with
  no calibration (uniform split) to avoid the known grid issue when E < max(grid) (deferred minor).
- **Chunk 2048** for scoring and prefill, as the references were built.
- **Lookahead off, q8 KV** (shipped defaults); `mlx_cache_gb = 1.0`.
- **Tokenizer:** never `fix_mistral_regex=True` (changes ids; every `score` asserts positions).

## Memory and GPU budget (16 GB rules: `gpu-run moe`, `watch.sh 4 17`, swap guard 3 GB)

| run | sessions | tokens prefilled | estimated time | memory |
|---|---|---|---|---|
| smoke: E1 on django-11019 | 1 | 42,771 | ~12 min | pruned pack, `memory_gb = 12`: slots + q8 KV 0.9 GB + activations, footprint ~13 GB (estimate) |
| M1 | 5 | 5 x 16,384 | ~25 min | full pack, `memory_gb = 12` |
| E1 rest | 4 | 283,519 | ~70-90 min | as smoke; 82k q8 measured at 13.0 GB footprint (docs/long-context.md) |
| E2 | 5 | 326,290 | ~90 min | bf16 KV at 82k: 3.4 GB KV; `memory_gb = 12` (fits at 81k, docs/long-context.md) |
| A0 refs | 5 | 20 x <=16k prefill + 20 x 1,024 generated | ~100 min | full pack |
| A1 | 5 | same | ~100 min | full pack |
| T1 | 3 tasks x 2 servers | | ~2 h | one server at a time |

Timing estimates use measured prefill (60-90 tok/s) and decode (~5-15 tok/s by context); `score`
re-runs nothing, `score_actions` re-prefills each action's prefix. Each run is one `gpu-run moe` job,
one at a time, lagunaLab first; results land in `results/parity_<id>.json` (always `--out` in this
repo).

## Order of runs

1. **Smoke E1 on django-11019** (shortest). Stop and debug if it misses the gate: everything after
   assumes the engine is right.
2. **M1** (short, 16k): the shipped pack against the full reference.
3. **E1 rest**, then **E2** (long, overnight).
4. **A0**, then **A1** (actions need the floor first).
5. **T1** when the user is away from the machine (two servers, ~2 h).

## Not in scope

- Building pruned-50 action references (none exist; costs GPU time; E1/E2 cover engine parity).
- Quality of the harness itself (sub-project 2, second part), refine loop, GRPO.
- Changing defaults from these results; findings are reported for the user to decide.

## Implementation sketch (for the plan)

- `tools/parity.py`: one entry point per comparison id; loads moeStream with the settings above,
  calls lagunaLab's `score` / `greedy_actions` (`diag_actions.run` pieces) read-only, adds the
  tool-call validity check, writes `results/parity_<id>.json` (per session plus weighted summary).
- Tests on the tiny model: summary weighting, the validity checker (valid, wrong name, bad JSON,
  missing required key), and a self-score of the tiny model through the moeStream loader giving KL 0,
  top-1 1.0 (as lagunaLab's `test_score_with_forward_callable`).

## Results (2026-10-16)

All gates passed, in run order. Every run: `gpu-run moe`, `watch.sh 4 17`, 3 GB swap guard, never
triggered.

| id | result | gate | file |
|---|---|---|---|
| E1 smoke (django-11019) | KL 0.0081, top-1 0.9922 | pass | `results/parity_E1_smoke.json` |
| M1 (full pack vs 4-bit base, 16k) | KL 0.0032, top-1 0.9891 | pass | `results/parity_M1.json` |
| E1 (pruned-50, q8 KV, full length) | KL 0.0020, top-1 0.9963 | pass | `results/parity_E1.json` |
| E2 (pruned-50, bf16 KV) | KL 0.0, top-1 1.0 (exact) | pass | `results/parity_E2.json` |
| A0 (own references) | 20/20 exact, 20/20 same tool, 20/20 valid calls | floor, so A1's exact threshold stays 14 | `results/parity_A0.json` |
| A1 (vs `refs/prune_actions`) | 18/20 exact, 20/20 same tool, 20/20 valid calls, 0 real mismatches, 1 tie | pass | `results/parity_A1.json` |
| T1 (pi, 3 tasks per server) | moeStream 3/3 completed (91, 99, 177 s), lagunaLab 3/3 (97, 95, 163 s); 0 server errors, no abort on either | pass | `results/pi_tasks/pi_tasks.json` |

- E2 being exact puts all of E1's gap on q8 KV; streaming itself adds nothing.
- A1's two inexact actions:
  - **django-10914 @ 5464**: `find ... | head -30` in the reference,
    `head -20` generated. Same tool, same argument keys. The first differing token's logit gap is
    wider than 4 bf16 ulps, so `classify` calls it *real*; it is not counted in `real_mismatches`,
    which (per the gate) counts only mismatches in tool name or argument structure.
  - **django-11001 @ 6602**: a different `find`/`grep` pipeline, same tool and keys; first
    differing token a tie.
- A1's first attempt crashed in `classify` on numpy int32 ids from the `.npz` references (A0 never
  reached it); fixed in b965854, crash log kept in `results/parity_A1_crash.log`.
- Peaks: A0/A1 footprint 13.3 GB, swap 215/509 MB; T1 moeStream server footprint 11.3 GB
  (lagunaLab's 8.3 GB), swap 509 MB.
