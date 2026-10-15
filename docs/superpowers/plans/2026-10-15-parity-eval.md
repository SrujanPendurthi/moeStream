# Parity eval implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps are TDD:
> each test is written and watched failing before the code.

**Goal:** `tools/parity.py`, which runs the comparisons of the parity-eval spec against lagunaLab's
references and writes `results/parity_<id>.json`; then the smoke run E1 on django__django-11019.

**Architecture:** one module. lagunaLab's harness is imported read-only (`replay.replay.score`,
`assistant_logprobs`, `greedy_actions`; `scripts.diag_actions.tool`); moeStream adds the loader, the
tool-call validity check, the tie-vs-real classifier, assistant-token-weighted summaries and the CLI.

**Spec:** `docs/specs/2026-10-15-parity-eval-design.md` (approved 2026-10-15, with two conditions:
gates in run order, stop and diagnose if E1 or M1 fails; report the smoke E1 before the long runs).

## Global constraints

- Never write into `../lagunaLab`; outputs only under `results/` here (`--out` always passed).
- Real-weight runs: `.gpu/gpu-run moe tools/watch.sh 4 17 ...`, one at a time, 3 GB swap guard,
  ps check after; a per-run copy of any script; never edit a running script.
- Tokenizer: never `fix_mistral_regex=True`. Chunk 2048. Django sessions only.
- Shipped settings unless a comparison says otherwise: q8 KV, lookahead off, `mlx_cache_gb = 1.0`.

## Review focus

1. A reference whose ids do not match the session tokenization: `score` asserts; the CLI must report
   the session as failed, not crash the whole run.
2. An action with several tool calls, or text after the call: every call is checked for validity.
3. A tool call cut off at `max_new` (no `</tool_call>`): invalid when the reference's is complete.
4. Tie classification when the two strings differ only after re-tokenization: the first differing
   *token* is used, not the first differing character.
5. Pruned pack (128 experts) with `budget.json` (256-expert grid): the loader must not use it.

---

### Task 1: tool-call validity

**Files:** create `tools/parity.py`, `tests/test_parity.py`.

- `parse_calls(text) -> list[(name, {key: value}) | None]`: Laguna's format,
  `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`; an unterminated call
  is `None`.
- `valid_call(call, tools) -> (bool, reason)`: name in the session's `tools`; every `required`
  parameter present; no unknown keys; each value converts to its schema type (`integer`, `number`,
  `boolean`, `string`; `array`/`object` must parse as JSON).
- Tests: valid shell call; unknown tool; missing required key; `start_line` = "abc" for an integer;
  unterminated call; two calls in one action (both checked); text with no call (empty list).

### Task 2: tie vs real mismatch

- `is_tie(logits, a, b, ulps=4) -> bool`: token ids `a` (reference) and `b` (generated) are the top
  two and their logit gap is within `ulps` bf16 ulps at that magnitude (ulp = 2^(floor(log2|x|) - 7)).
- `first_diff(ref_ids, got_ids) -> int | None` on token ids (re-tokenized action text).
- `classify(model, prefix_ids, ref_ids, got_ids) -> "exact" | "tie" | "real"`: runs the model on
  prefix + common tokens and applies `is_tie` at the first differing token.
- Tests: constructed logits for a tie, a clear gap, b not in the top 2; `first_diff` on equal, prefix
  and differing lists; `classify` on the tiny model where `got` is the model's own greedy continuation
  with one token swapped for its second choice (tie only if the gap is tiny; assert consistency with
  `is_tie` on the same logits).

### Task 3: scoring adapter (E1, E2, M1)

- `load_engine(pack, kv_bits, memory_gb, context_tokens)`: `moestream.load.load` with no calibration
  for a pack whose `num_experts` differs from the calibration's grid top (pruned-50), else
  `budget.json`; `lookahead` off; `cap_buffer_pool(1.0)`.
- `score_sessions(model, refs_dir, sessions, out, make_cache=None, forward=None)`: lagunaLab `score`
  per session (`chunk=2048`), each result written to `out` as it lands; a failing session is recorded
  with its error; summary weighted by `assistant_tokens`; `gate(summary, kl, top1) -> bool`.
- Tests (tiny model): a reference built from the tiny model's own `assistant_logprobs` (same npz
  fields as `build_ref`: ids, mask, truncated, pos, top_ids, top_lp) scores KL 0, top-1 1.0, through
  the default path and through `forward=lambda ids, cache: model(ids, cache=cache)`; weighting over
  two sessions; a mismatched-ids reference is recorded as failed and the next session still runs.

### Task 4: actions adapter (A0, A1)

- `action_rows(model, tok, refs_dir, sessions, greedy=greedy_actions)`: per reference action,
  exact, same tool (`scripts.diag_actions.tool`), common prefix, validity of every generated call,
  and `classify` for mismatches; summary: exact, same-tool, valid/total, real-mismatch count.
- `build_self_refs` for A0: lagunaLab `build_ref(..., model_id="moestream-full-q8")` on the same
  starts as `refs/prune_actions`, written under `results/parity_refs/`.
- Tests: rows from a stubbed `greedy` (exact, same tool with different args, different tool,
  malformed call); summary counts; the A1 gate (same tool >= 19/20, exact >= 14/20, no real mismatch).

### Task 5: CLI and smoke run

- `python3 -m tools.parity --id E1|E2|M1|A0|A1 [--sessions ...] --out results/parity_<id>.json`;
  settings per id from the spec table; prints the gate verdict.
- Test: argument parsing and the per-id settings table.
- **Smoke E1** (GPU): `gpu-run moe tools/watch.sh 4 17 python3 -m tools.parity --id E1 --sessions
  django__django-11019 --out results/parity_E1_smoke.json`, `memory_gb = 12`; 3 GB swap guard; ps
  check; commit the result; report it to the coordinator before anything longer runs. If it misses
  the gate (KL <= 0.02, top-1 >= 0.98), stop and diagnose.

T1 (pi tasks) and the long runs are scheduled by the coordinator after the smoke result.
