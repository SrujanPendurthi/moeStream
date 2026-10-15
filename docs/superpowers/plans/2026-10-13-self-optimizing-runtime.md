# Self-optimizing runtime (system level) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The served engine measures itself, tries one system-level change at a time, keeps it only if it measured better (and passed the strict KL guard when it can change the math), shrinks under memory pressure, and resumes from saved state after a restart.

**Architecture:** `moestream/tune.py` holds the tuner (trial loop, replay judge, A/B judge, state, log). `moestream/guard.py` is the strict-KL guard, `moestream/knobs.py` the five system knobs, `moestream/pressure.py` the memory-pressure guard. Everything runs on mlx-lm's single generation thread: a per-token hook on the last `StreamingMoE` layer and an idle hook wrapped around `ResponseGenerator._next_request`. Replay simulations run in 2 low-priority worker processes.

**Tech Stack:** Python 3, MLX, mlx-lm 0.31.3 server, numpy, pytest (tiny model from `../lagunaLab/tests/tiny.py`).

**Spec:** `docs/specs/2026-10-13-self-optimizing-runtime-design.md`

## Global Constraints

- Never write into `../lagunaLab`. Import only its frozen interfaces (`laguna_mlx.laguna`, `laguna_mlx.streaming`, `laguna_mlx.dflash.generate_greedy`, `laguna_mlx.cache.make_quantized_cache`/`DequantKVCache`, `scripts.pack_experts`, `scripts.check_parity.PROMPTS`, `scripts.serve_streaming`). Task 10's `replay.session` needs the coordinator's OK first.
- Test command: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`. Tiny-model tests need no GPU lock.
- Real-weight runs only via `PYTHONPATH=.:../lagunaLab /Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe tools/watch.sh 8 21 <cmd>`, one job at a time, after telling the coordinator.
- Peak memory never exceeds `memory_gb`; the slot total never exceeds the total `load()` allocated.
- Strict KL: mean ≤ 1e-3 nats/token, 99th percentile ≤ 1e-2, top-1 agreement ≥ 99.5%; reference is the untuned engine, never the previous tuned state.
- Replay keep rule: candidate miss rate ≤ 0.99 × current. A/B keep rule: candidate faster in ≥ 7 of 8 paired 64-token windows.
- Re-proposal of a rejected knob: after 50,000 more tokens, or when window hit rate moved more than 0.05.
- Tuning never stops serving: every tuner error path logs and continues.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Deviations from the spec (rulings, to copy into the ledger)

- Prefetch proposal flips one quarter of the prefetching layers per trial instead of the spec's per-layer analytic rule: the engine cannot measure a layer's extra sync cost separately from its routing sync. A/B decides. Cost if wrong: 4 trials per sweep instead of 1.
- Half-life candidates are the whole `HALF_LIVES` grid, not only neighbours: cheaper to write, runs in background. Cost: more CPU per trial.
- Probe text is `PROMPTS` plus the untuned engine's own greedy continuation, not slices of lagunaLab traces: reading traces needs `replay.session`, which is not a frozen interface. Cost: less tool-call text in the probe.
- The routing ring records prefill rows too (teacher-forced routing equals decode routing, as `tools/record.py` relies on), and holds up to 2× `trace_tokens` rows between compactions (~40 MB at 65,536). Probe passes also land in it.
- A math-changing candidate is applied before its guard runs, so live requests served during the probe use the unverified candidate. The spec requires the guard before it is *kept*. Cost if wrong: a few requests at the candidate's (strictly bounded) error.

## Review Focus

1. A request arriving mid-trial or mid-probe: the A/B arm must revert to the current value and the half-finished pair must be dropped (Task 6 test `test_interrupt_drops_half_pair_and_reverts_arm`).
2. A restart with a changed `memory_gb`, `kv_bits` or pack: saved state must be ignored, not half-applied (Task 6 `test_state_fingerprint_mismatch_is_ignored`).
3. A replay worker raising or the pool dying: the trial ends as "no decision", serving continues (Task 6 `test_replay_worker_error_ends_trial_without_raising`).
4. Memory pressure during a slot-split trial: the split knob must pause while slots are shrunk (Task 6 `test_knob_not_ready_is_skipped`, Task 9 wiring).
5. JSON round trip of knob values: split keys come back as strings, half-life `inf` as `Infinity` (Task 7 `test_split_accepts_string_keys_from_json`).

---

### Task 1: SlotStore settings change live

**Files:**
- Modify: `moestream/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `SlotStore.set_policy(policy)`, `SlotStore.set_half_life(h)`, `SlotStore.half_life`, `SlotStore.key(e)` (lowest key is evicted first).

- [ ] **Step 1: Write the failing test** (append to `tests/test_store.py`)

```python
def test_policy_and_half_life_change_live():
    s = SlotStore(8, 2, half_life=float("inf"))
    for _ in range(3):
        s.tick(); s.assign([0])
    s.tick(); s.assign([1])          # 0 used most, 1 used last
    s.set_policy("lru")
    s.tick(); _, f = s.assign([2])
    assert f == [(2, 0)]             # lru evicts 0; lfu would have evicted 1
    s.set_half_life(1.0)
    assert s.half_life == 1.0 and s.decay == 0.5
    with pytest.raises(ValueError, match="lifo"):
        s.set_policy("lifo")


def test_key_orders_eviction():
    s = SlotStore(8, 3, half_life=float("inf"))
    for e in (0, 0, 1, 2):
        s.tick(); s.assign([e])
    assert sorted(s.expert_slot, key=s.key) == [1, 2, 0]
```

- [ ] **Step 2: Run it**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_store.py -q`
Expected: 2 failed (`AttributeError: 'SlotStore' object has no attribute 'set_policy'` / `'key'`).

- [ ] **Step 3: Implement**

In `moestream/store.py`, replace the first lines of `__init__` and `_victims`, and add the three methods:

```python
    def __init__(self, num_experts, slots, half_life=256.0, policy="lfu"):
        self.slots = slots
        self.set_policy(policy)
        self.set_half_life(half_life)
        self.slot_expert = [-1] * slots
        self.expert_slot = {}
        self.score = np.zeros(num_experts)
        self.last = np.zeros(num_experts, np.int64)
        self.loaded = np.zeros(num_experts, np.int64)
        self.clock = 0
        self.hits = self.misses = 0

    def set_policy(self, policy):
        if policy not in POLICIES:
            raise ValueError(f"eviction policy {policy!r} not in {POLICIES}")
        self.policy = policy

    def set_half_life(self, half_life):
        self.half_life = half_life
        self.decay = 0.5 ** (1.0 / half_life)

    def key(self, e):
        """Eviction rank of a resident expert: the lowest key is evicted first."""
        if self.policy == "lfu":
            return (self.score[e], self.last[e])
        return (self.last[e] if self.policy == "lru" else self.loaded[e],)
```

```python
    def _victims(self, n, keep):
        free = [s for s, e in enumerate(self.slot_expert) if e == -1]
        used = [s for s, e in enumerate(self.slot_expert) if e != -1 and e not in keep]
        used.sort(key=lambda s: self.key(self.slot_expert[s]))
        return (free + used)[:n]
```

- [ ] **Step 4: Run it**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_store.py -q`
Expected: all pass (existing policy tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add moestream/store.py tests/test_store.py
git commit -m "feat: SlotStore policy and half-life change live; key() exposes eviction rank

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: StreamingMoE resize, per-layer lookahead, token hook; prefetch thread count

**Files:**
- Modify: `moestream/layer.py`, `moestream/prefetch.py`
- Test: `tests/test_layer.py`, `tests/test_prefetch.py`

**Interfaces:**
- Consumes: `SlotStore.key`, `SlotStore.half_life` (Task 1).
- Produces: `StreamingMoE.resize(n)`; `io.lookahead` (bool, default True); `io.hook` (callable `hook(decode: bool)` or None, called at the end of every layer call); `prefetch.threads() -> int`; `prefetch.set_threads(n)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_layer.py`)

```python
def test_resize_between_tokens_keeps_decode_bit_exact(tmp_path):
    d, resident = quantized_model_dir(tmp_path, dtype=mx.bfloat16)
    out = tmp_path / "packed"
    pack(d, out)
    m, _ = load(out, slots=4)
    ids = _ids(16, seed=3)
    cache = m.make_cache()
    got = [m(ids[:, :8], cache=cache)]
    for i in range(8, 16):
        if i in (10, 13):
            for moe in _moe(m):
                moe.resize(2 if i == 10 else 6)
        got.append(m(ids[:, i : i + 1], cache=cache))
    want = _decode(resident, ids, 8)
    assert mx.array_equal(mx.concatenate(got, axis=1), want).item()
    assert {moe.io.store.slots for moe in _moe(m)} == {6}


def test_resize_keeps_the_experts_eviction_ranks_highest(packed):
    out, _ = packed
    m, _ = load(out, slots=4)
    m(_ids(12))
    moe = _moe(m)[0]
    st = moe.io.store
    best = sorted(st.expert_slot, key=st.key, reverse=True)[:2]
    hits, misses = st.hits, st.misses
    moe.resize(2)
    new = moe.io.store
    assert new.slots == 2 and set(new.expert_slot) == set(best)
    assert (new.hits, new.misses, new.policy) == (hits, misses, st.policy)


def test_lookahead_off_skips_prediction_and_stays_exact(packed):
    out, resident = packed
    m, _ = load(out, slots=4)
    for moe in _moe(m):
        moe.io.lookahead = False
    ids = _ids(16, seed=7)
    _close(_decode(m, ids, 10), resident(ids))
    assert stream_stats(m)["pred_acc"] == 0


def test_hook_sees_decode_and_prefill(packed):
    out, _ = packed
    m, _ = load(out, slots=4)
    seen = []
    _moe(m)[-1].io.hook = seen.append
    _decode(m, _ids(6), 4)
    assert seen == [False, True, True]
```

Append to `tests/test_prefetch.py`:

```python
def test_thread_count_changes_and_reads_still_work(tmp_path):
    from moestream import prefetch

    f = tmp_path / "x.bin"
    f.write_bytes(bytes(range(256)) * 2)
    old = prefetch.threads()
    try:
        prefetch.set_threads(3)
        assert prefetch.threads() == 3
        p = prefetch.Prefetcher(f, 256)
        assert p.take([1])[1][5] == 5
    finally:
        prefetch.set_threads(old)
```

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_layer.py tests/test_prefetch.py -q`
Expected: 5 failed (`no attribute 'resize'`, `'lookahead'` has no effect so `pred_acc > 0`, `hook` never called, `no attribute 'threads'`).

- [ ] **Step 3: Implement**

`moestream/layer.py`: add a module-level helper and use it in `__init__`:

```python
def _glu(args, slots, group_size, bits, dtype):
    glu = SwitchGLU(args.hidden_size, args.moe_intermediate_size, slots)
    glu.set_dtype(dtype)  # match packed scales dtype: no cast on write
    nn.quantize(glu, group_size=group_size, bits=bits)
    return glu
```

In `__init__`, replace the three `glu` lines and the `io` namespace head with:

```python
        self.io = SimpleNamespace(
            glu=_glu(args, slots, group_size, bits, dtype),
            make_glu=lambda n: _glu(args, n, group_size, bits, dtype),
            layout=layout, next=None, predictor=None, trace=None, guess=None, lookahead=True, hook=None,
```

(the remaining `store=`, `prefetcher=`, `stats=` lines stay as they are).

In `__call__`:

```python
        lookahead = decode and io.next is not None and io.predictor is None and io.lookahead
```

```python
        if decode and io.next is not None and (lookahead or io.predictor is not None):
```

and just before the final `return`:

```python
        if io.hook is not None:
            io.hook(decode)
```

Add the method:

```python
    def resize(self, n):
        """Change this layer's slot count between tokens, keeping the n experts the eviction key ranks highest."""
        io, old = self.io, self.io.store
        keep = sorted(old.expert_slot, key=old.key, reverse=True)[:n]
        store = SlotStore(len(old.score), n, old.half_life, old.policy)
        store.score, store.last, store.loaded = old.score, old.last, old.loaded
        store.clock, store.hits, store.misses = old.clock, old.hits, old.misses
        glu = io.make_glu(n)
        if keep:
            src = mx.array([old.expert_slot[e] for e in keep])
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    getattr(getattr(glu, proj), part)[: len(keep)] = getattr(getattr(io.glu, proj), part)[src]
        for s, e in enumerate(keep):
            store.slot_expert[s], store.expert_slot[e] = e, s
        mx.eval(glu.parameters())
        io.glu, io.store = glu, store
```

`moestream/prefetch.py`, after `_POOL = ThreadPoolExecutor(8)`:

```python
def threads():
    return _POOL._max_workers


def set_threads(n):
    """Swap the read pool; reads already queued finish on the old one."""
    global _POOL
    old, _POOL = _POOL, ThreadPoolExecutor(n)
    old.shutdown(wait=False)
```

- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass (85).

- [ ] **Step 5: Commit**

```bash
git add moestream/layer.py moestream/prefetch.py tests/test_layer.py tests/test_prefetch.py
git commit -m "feat: resize slots live, per-layer lookahead switch, token hook, settable read threads

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Tuning keys in the config

**Files:**
- Modify: `moestream/config.py`, `moestream.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `EngineConfig.tune: bool = True`, `EngineConfig.tuning_dir: str = "tuning"` (resolved against the TOML's directory), `EngineConfig.trace_tokens: int = 65536`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_config.py`)

```python
def test_tuning_keys_default_and_resolve(tmp_path):
    c = EngineConfig.read(_write(tmp_path, 'pack_dir = "p"\n'))
    assert c.tune is True and c.trace_tokens == 65536
    assert Path(c.tuning_dir) == tmp_path / "tuning"
    shipped = EngineConfig.read(REPO / "moestream.toml")
    assert shipped.tune is True and Path(shipped.tuning_dir) == REPO / "tuning"


@pytest.mark.parametrize("line", ["tune = 1", "trace_tokens = 0", "tuning_dir = 3"])
def test_bad_tuning_values_raise(tmp_path, line):
    with pytest.raises(ValueError, match=line.split(" ")[0]):
        EngineConfig.read(_write(tmp_path, f'pack_dir = "p"\n{line}\n'))
```

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_config.py -q`
Expected: 4 failed (`no attribute 'tune'`; `unknown keys`).

- [ ] **Step 3: Implement**

`moestream/config.py` dataclass, after `port`:

```python
    tune: bool = True
    tuning_dir: str = "tuning"
    trace_tokens: int = 65536
```

In `read`, resolve `tuning_dir` with the other paths:

```python
        for key in ("pack_dir", "calibration", "tuning_dir"):
```

In `_check`, add to `rules`:

```python
            "tune": isinstance(self.tune, bool),
            "tuning_dir": isinstance(self.tuning_dir, str),
            "trace_tokens": _int(self.trace_tokens) and self.trace_tokens > 0,
```

`moestream.toml`, append:

```toml
tune = true               # false = serve with the saved tuning state, never trial
tuning_dir = "tuning"     # tuning state, log and quality reference
trace_tokens = 65536      # routing rows kept per layer for replay
```

Add `tuning/` to `.gitignore`.

- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add moestream/config.py moestream.toml tests/test_config.py .gitignore
git commit -m "feat: tune, tuning_dir and trace_tokens config keys

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Routing ring, replay judge, A/B rule

**Files:**
- Create: `moestream/tune.py`
- Modify: `tools/calibrate.py` (`miss_rate` gains `policy`)
- Test: `tests/test_tune.py`

**Interfaces:**
- Consumes: `SlotStore(num_experts, slots, half_life, policy)`; `budget.allocate`; `tools.calibrate.GRID`, `HALF_LIVES`, `miss_rate`.
- Produces: `Ring(cap)` with `.append(rows)`, `.rows() -> np.ndarray | None`; `replay_rate(rows, slots, policy, half_life, num_experts) -> float`; `replay_trial(knob, snap) -> (value | None, current_rate, candidate_rate)` where `knob in {"policy", "half_life", "slots"}` and `snap = {rows, slots, policy, half_life, num_experts, grid}`; `ab_keep(pairs, n=8, need=7) -> bool | None`.

- [ ] **Step 1: Write the failing tests** (`tests/test_tune.py`)

```python
import numpy as np

from moestream.tune import Ring, ab_keep, replay_rate, replay_trial


def _phase_shift():
    # {0,1} hammered, then {2,3},{4,5} alternating: needs 4 slots; plain LFU keeps 0,1 pinned
    return np.array([[0, 1]] * 300 + [[2, 3], [4, 5]] * 150, np.uint8)


def _snap(rows, policy="lfu", half_life=float("inf"), slots=4, grid=(2, 4, 8)):
    return dict(rows={1: rows, 2: rows}, slots={1: slots, 2: slots}, policy=policy, half_life=half_life,
                num_experts=8, grid=list(grid))


def test_ring_keeps_the_last_cap_rows():
    r = Ring(5)
    assert r.rows() is None
    for i in range(12):
        r.append(np.full((1, 2), i, np.uint8))
    assert r.rows()[:, 0].tolist() == [7, 8, 9, 10, 11]
    r.append(np.zeros((3, 2), np.uint8))
    assert len(r.rows()) == 5


def test_replay_rate_matches_per_layer_miss_rate():
    rows = _phase_shift()
    lfu = replay_rate({1: rows}, {1: 4}, "lfu", float("inf"), 8)
    lru = replay_rate({1: rows}, {1: 4}, "lru", float("inf"), 8)
    assert lru < lfu


def test_policy_trial_finds_a_better_policy():
    value, cur, cand = replay_trial("policy", _snap(_phase_shift()))
    assert value in ("lru", "fifo") and cand < cur * 0.99


def test_half_life_trial_leaves_infinite_half_life():
    value, cur, cand = replay_trial("half_life", _snap(_phase_shift()))
    assert value != float("inf") and cand < cur


def test_trial_without_a_better_value_returns_none():
    rows = np.array([[0, 1]] * 100, np.uint8)  # fits: only cold misses, every policy ties
    value, cur, cand = replay_trial("policy", _snap(rows))
    assert value is None and cand == cur


def test_slot_trial_moves_slots_to_the_layer_that_misses():
    busy, quiet = _phase_shift(), np.array([[0, 1]] * 600, np.uint8)
    snap = _snap(busy, slots=3)
    snap["rows"] = {1: busy, 2: quiet}
    value, cur, cand = replay_trial("slots", snap)
    assert value == {1: 4, 2: 2} and cand < cur


def test_ab_keep_needs_seven_of_eight():
    faster = [(1.0, 0.9)] * 7 + [(1.0, 1.1)]
    assert ab_keep(faster[:7]) is None
    assert ab_keep(faster) is True
    assert ab_keep([(1.0, 0.9)] * 6 + [(1.0, 1.1)] * 2) is False
```

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_tune.py -q`
Expected: collection error `No module named 'moestream.tune'`.

- [ ] **Step 3: Implement**

`tools/calibrate.py`:

```python
def miss_rate(seq, slots, half_life, num_experts=256, policy="lfu"):
    s = SlotStore(num_experts, slots, half_life, policy)
```

`moestream/tune.py`:

```python
"""Self-optimizing runtime: measure, try one change, keep it only if it measured better.

Runs on mlx-lm's generation thread: on_token() after each decode step, idle() when no request
is waiting. Replay trials run in up to 2 low-priority worker processes, so this module stays
free of MLX imports.
"""
import numpy as np

from moestream.budget import allocate
from moestream.store import POLICIES
from tools.calibrate import HALF_LIVES, miss_rate


class Ring:
    """The last `cap` routing rows of one layer; StreamingMoE appends to it through io.trace."""

    def __init__(self, cap):
        self.cap, self.parts, self.n = cap, [], 0

    def append(self, rows):
        self.parts.append(rows)
        self.n += len(rows)
        if self.n > 2 * self.cap:  # ponytail: holds up to 2x cap rows between compactions
            self.parts = [self.rows()]
            self.n = len(self.parts[0])

    def rows(self):
        return np.concatenate(self.parts)[-self.cap :] if self.parts else None


def replay_rate(rows, slots, policy, half_life, num_experts):
    """Token-weighted simulated miss rate over layers; rows {layer: (T, k) uint8}, slots {layer: n}."""
    n = sum(len(r) for r in rows.values())
    return sum(miss_rate(r, slots[l], half_life, num_experts, policy) * len(r) for l, r in rows.items()) / max(1, n)


def replay_trial(knob, snap):
    """Worker: best candidate for one replay knob and both simulated miss rates; candidate None if none is lower."""
    rows, slots, policy, hl, E = (snap[k] for k in ("rows", "slots", "policy", "half_life", "num_experts"))
    if knob == "policy":
        cands = [dict(policy=p) for p in POLICIES if p != policy]
    elif knob == "half_life":
        cands = [dict(half_life=h) for h in HALF_LIVES if h != hl]
    else:
        curves = {l: [miss_rate(r, g, hl, E, policy) for g in snap["grid"]] for l, r in rows.items()}
        cands = [dict(slots=allocate(curves, snap["grid"], sum(slots.values())))]
    cur = replay_rate(rows, slots, policy, hl, E)
    best, best_rate = None, cur
    for c in cands:
        r = replay_rate(rows, c.get("slots", slots), c.get("policy", policy), c.get("half_life", hl), E)
        if r < best_rate:
            best, best_rate = next(iter(c.values())), r
    return best, cur, best_rate


def ab_keep(pairs, n=8, need=7):
    """pairs: (current seconds, candidate seconds) per matched window pair. None until n pairs."""
    if len(pairs) < n:
        return None
    return sum(c < a for a, c in pairs[:n]) >= need
```

- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass. If `test_slot_trial_moves_slots_to_the_layer_that_misses` gives another split, print `curves` and fix the test's expected split from them (the rule is "the busy layer gets more slots, total ≤ 6"); ledger the change.

- [ ] **Step 5: Commit**

```bash
git add moestream/tune.py tools/calibrate.py tests/test_tune.py
git commit -m "feat: routing ring, replay judge and A/B keep rule for the tuner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Strict KL guard

**Files:**
- Create: `moestream/guard.py`
- Test: `tests/test_guard.py`

**Interfaces:**
- Produces: `THRESHOLDS`; `reference(logits, k) -> (ids, logp)`; `kl_topk(ref, logits) -> (kl, top1)`; `verdict(kl, top1) -> {kl_mean, kl_p99, top1, ok}`; `quiet(v) -> bool`; `Guard(model, prompts, path, key, total=2048, chunk=128)` with `.ready`, `.math_ok`, `.start()`, `.step() -> dict | None`.

- [ ] **Step 1: Write the failing tests** (`tests/test_guard.py`)

```python
import numpy as np
import pytest

from moestream.guard import THRESHOLDS, Guard, kl_topk, quiet, reference, verdict
from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir


def test_identical_logits_pass_and_hot_logits_fail():
    lg = np.random.default_rng(0).normal(size=(50, 300)).astype(np.float32)
    ref = reference(lg, 64)
    v = verdict(*kl_topk(ref, lg))
    assert v["ok"] and v["kl_mean"] < 1e-9 and v["top1"] == 1.0 and quiet(v)
    assert not verdict(*kl_topk(ref, 3 * lg))["ok"]


def test_disagreement_past_half_the_threshold_is_not_quiet():
    lg = np.random.default_rng(1).normal(size=(400, 300)).astype(np.float32)
    v = verdict(*kl_topk(reference(lg, 64), lg + np.random.default_rng(2).normal(0, 0.06, lg.shape)))
    assert v["kl_mean"] > THRESHOLDS["mean"] / 2 and not quiet(v)


@pytest.fixture
def model(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    return load(tmp_path / "packed", slots=4)[0]


def _run(g):
    while True:
        v = g.step()
        if v is not None or (g.ready and g._run is None):
            return v


def test_guard_builds_reference_in_slices_then_passes_itself(model, tmp_path):
    prompts = [list(np.random.default_rng(s).integers(0, TINY["vocab_size"], 5)) for s in range(3)]
    g = Guard(model, prompts, tmp_path / "ref.npz", "k1", total=48, chunk=8)
    steps = 0
    while not g.ready:
        g.step(); steps += 1
    assert steps > 6 and g.math_ok and g.seqs.shape == (3, 16)
    g.start()
    assert _run(g)["ok"]
    again = Guard(model, prompts, tmp_path / "ref.npz", "k1", total=48, chunk=8)
    assert again.ready and again.math_ok                      # loaded, not rebuilt
    assert not Guard(model, prompts, tmp_path / "ref.npz", "k2", total=48, chunk=8).ready


def test_guard_fails_a_changed_model_and_survives_errors(model, tmp_path):
    prompts = [[1, 2, 3, 4]]
    g = Guard(model, prompts, tmp_path / "ref.npz", "k", total=16, chunk=8)
    while not g.ready:
        g.step()

    class Hot:
        make_cache = model.make_cache

        def __call__(self, x, cache=None):
            return model(x, cache=cache) * 3.0

    g.model = Hot()
    g.start()
    assert _run(g)["ok"] is False

    class Broken:
        make_cache = model.make_cache

        def __call__(self, x, cache=None):
            raise MemoryError("oom")

    g.model = Broken()
    g.start()
    assert g.step() == {"ok": False, "error": "MemoryError('oom')"}
```

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_guard.py -q`
Expected: collection error `No module named 'moestream.guard'`.

- [ ] **Step 3: Implement** (`moestream/guard.py`)

```python
"""Strict KL guard: the engine against a fixed reference of the untuned engine.

Work runs in bounded slices (one chunk of tokens per step) so it can share mlx-lm's generation
thread with live requests.
"""
from pathlib import Path

import mlx.core as mx
import numpy as np

K = 64
THRESHOLDS = {"mean": 1e-3, "p99": 1e-2, "top1": 0.995}


def _log_softmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def reference(logits, k=K):
    """(T, V) logits -> top-k token ids (best first) and their log-probabilities."""
    lp = _log_softmax(logits)
    ids = np.argsort(-lp, axis=-1)[:, :k]
    return ids, np.take_along_axis(lp, ids, -1)


def kl_topk(ref, logits):
    """KL(reference || candidate) on the reference's top-k support, renormalized; and top-1 agreement."""
    ids, ref_lp = ref
    p, q = _log_softmax(ref_lp), _log_softmax(np.take_along_axis(_log_softmax(logits), ids, -1))
    return (np.exp(p) * (p - q)).sum(-1), ids[:, 0] == logits.argmax(-1)


def verdict(kl, top1, th=THRESHOLDS):
    v = {"kl_mean": float(kl.mean()), "kl_p99": float(np.quantile(kl, 0.99)), "top1": float(top1.mean())}
    v["ok"] = v["kl_mean"] <= th["mean"] and v["kl_p99"] <= th["p99"] and v["top1"] >= th["top1"]
    return v


def quiet(v, th=THRESHOLDS):
    """Two runs of the same engine: do they disagree by less than half the threshold?"""
    return v["kl_mean"] <= th["mean"] / 2 and v["kl_p99"] <= th["p99"] / 2 and 1 - v["top1"] <= (1 - th["top1"]) / 2


class Guard:
    def __init__(self, model, prompts, path, key, total=2048, chunk=128, k=K, th=THRESHOLDS):
        self.model, self.path, self.key, self.chunk, self.k, self.th = model, Path(path), key, chunk, k, th
        self.ready = self.math_ok = False
        self._run = None
        if self.path.exists():
            z = np.load(self.path)
            if str(z["key"]) == key:
                self.seqs, self.ref = z["seqs"], (z["ids"], z["logp"])
                self.math_ok, self.ready = bool(z["math_ok"]), True
        if not self.ready:
            self._run = self._build(prompts, total)

    def start(self):
        self._run = self._compare()

    def step(self):
        """One slice of work. Returns the verdict when a comparison finishes, else None."""
        try:
            next(self._run)
        except StopIteration as s:
            self._run = None
            return s.value
        except Exception as e:  # out of memory and the like: a failed guard, never a crash
            self._run = None
            if not self.ready:
                self.ready = True  # math_ok stays False: math-changing knobs stay off
                return None
            return {"ok": False, "error": repr(e)}
        return None

    def _build(self, prompts, total):
        per = total // len(prompts)
        seqs = []
        for p in prompts:
            ids, cache = [int(t) for t in p[:per]], self.model.make_cache()
            logits = self.model(mx.array([ids]), cache=cache)[0, -1]
            while len(ids) < per:  # the untuned engine's own greedy continuation
                ids.append(int(mx.argmax(logits)))
                logits = self.model(mx.array([ids[-1:]]), cache=cache)[0, -1]
                if len(ids) % 16 == 0:
                    yield
            seqs.append(ids)
            yield
        self.seqs = np.array(seqs, np.int32)
        parts = []
        for lg in self._chunks():
            parts.append(reference(lg, self.k))
            yield
        self.ref = tuple(np.concatenate(a) for a in zip(*parts))
        v = yield from self._compare()
        self.math_ok = quiet(v, self.th)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, key=self.key, seqs=self.seqs, ids=self.ref[0], logp=self.ref[1], math_ok=self.math_ok)
        self.ready = True

    def _chunks(self):
        for ids in self.seqs:
            cache = self.model.make_cache()
            for a in range(0, len(ids), self.chunk):
                yield np.array(self.model(mx.array(ids[None, a : a + self.chunk]), cache=cache)[0].astype(mx.float32))

    def _compare(self):
        kls, tops, i = [], [], 0
        for lg in self._chunks():
            kl, top = kl_topk((self.ref[0][i : i + len(lg)], self.ref[1][i : i + len(lg)]), lg)
            kls.append(kl); tops.append(top); i += len(lg)
            yield
        return verdict(np.concatenate(kls), np.concatenate(tops), self.th)
```

- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add moestream/guard.py tests/test_guard.py
git commit -m "feat: strict KL guard against a fixed untuned reference, in bounded slices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The tuner

**Files:**
- Modify: `moestream/tune.py`
- Test: `tests/test_tuner.py`

**Interfaces:**
- Consumes: `replay_trial`, `ab_keep` (Task 4); a guard with `.ready`, `.math_ok`, `.start()`, `.step()` (Task 5).
- Produces: `Knob(name, tier, get, apply, propose=None, changes_math=False, ready=lambda: True)`; `Tuner(knobs, snapshot, dirpath, fingerprint, *, guard=None, pool=None, clock=time.perf_counter, counts=lambda: (0, 0), trial=True, retry_tokens=50_000)` with `on_token()`, `interrupt()`, `idle()`, `load_state()`, `save()`, `log(record)`, `.trial`, `.disabled`. Files: `<dirpath>/state.json`, `<dirpath>/log.jsonl`.

- [ ] **Step 1: Write the failing tests** (`tests/test_tuner.py`)

```python
import json
from concurrent.futures import Future

import numpy as np
import pytest

from moestream.tune import Knob, Tuner


class Now:
    """Runs replay work inline: futures come back done."""

    def submit(self, fn, *a):
        f = Future()
        try:
            f.set_result(fn(*a))
        except Exception as e:
            f.set_exception(e)
        return f


def _live(cur, propose="new", **kw):
    return Knob("x", "live", lambda: cur["v"], lambda v: cur.update(v=v), lambda: propose, **kw)


def _ab(tmp_path, slow_pairs):
    cur, t = {"v": "old"}, [0.0]
    tuner = Tuner([_live(cur)], lambda: None, tmp_path, "fp", clock=lambda: t[0])
    tuner.idle()
    tuner.on_token()
    while tuner.trial is not None:
        p = len(tuner.trial["pairs"])
        t[0] += 1.0 if cur["v"] == "old" else (1.1 if p < slow_pairs else 0.9)
        tuner.on_token()
    return cur["v"], tuner


def test_live_candidate_kept_at_seven_of_eight(tmp_path):
    assert _ab(tmp_path, 1)[0] == "new"


def test_live_candidate_reverted_at_six_of_eight(tmp_path):
    v, tuner = _ab(tmp_path, 2)
    assert v == "old"
    rec = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert rec[-1]["knob"] == "x" and rec[-1]["kept"] is False


def test_interrupt_drops_half_pair_and_reverts_arm(tmp_path):
    cur, t = {"v": "old"}, [0.0]
    tuner = Tuner([_live(cur)], lambda: None, tmp_path, "fp", clock=lambda: t[0])
    tuner.idle()
    for _ in range(Tuner.WINDOW + 1):
        t[0] += 1; tuner.on_token()
    assert cur["v"] == "new"              # candidate arm running
    tuner.interrupt()
    assert cur["v"] == "old" and tuner.trial["pending"] is None and tuner.trial["pairs"] == []


def test_rejected_knob_waits_before_retry(tmp_path):
    v, tuner = _ab(tmp_path, 8)
    tuner.retry_tokens = 10
    tuner.idle()
    assert tuner.trial is None
    for _ in range(11):
        tuner.on_token()
    tuner.idle()
    assert tuner.trial is not None


def test_knob_not_ready_is_skipped(tmp_path):
    cur = {"v": "old"}
    tuner = Tuner([_live(cur, ready=lambda: False)], lambda: None, tmp_path, "fp")
    tuner.idle()
    assert tuner.trial is None


def test_failing_apply_disables_knob_and_reverts(tmp_path):
    cur = {"v": "old"}

    def apply(v):
        if v == "new":
            raise RuntimeError("no")
        cur.update(v=v)

    t = [0.0]
    tuner = Tuner([Knob("x", "live", lambda: cur["v"], apply, lambda: "new")], lambda: None, tmp_path, "fp",
                  clock=lambda: t[0])
    tuner.idle()
    for _ in range(Tuner.WINDOW + 1):
        t[0] += 1; tuner.on_token()
    assert cur["v"] == "old" and "x" in tuner.disabled and tuner.trial is None


def _replay_snap():
    rows = np.array([[0, 1]] * 300 + [[2, 3], [4, 5]] * 150, np.uint8)
    return dict(rows={1: rows}, slots={1: 4}, policy="lfu", half_life=float("inf"), num_experts=8, grid=[2, 4, 8])


def test_replay_candidate_applied_when_simulation_is_lower(tmp_path):
    cur = {"v": "lfu"}
    knob = Knob("policy", "replay", lambda: cur["v"], lambda v: cur.update(v=v))
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Now())
    tuner.idle(); tuner.idle()
    assert cur["v"] in ("lru", "fifo") and tuner.trial is None


def test_replay_worker_error_ends_trial_without_raising(tmp_path):
    cur = {"v": "lfu"}
    knob = Knob("policy", "replay", lambda: cur["v"], lambda v: cur.update(v=v))
    tuner = Tuner([knob], lambda: {"rows": None}, tmp_path, "fp", pool=Now())
    tuner.idle(); tuner.idle()
    assert cur["v"] == "lfu" and tuner.trial is None


class FakeGuard:
    def __init__(self, ok):
        self.ready = self.math_ok = True
        self.ok, self.started = ok, 0

    def start(self):
        self.started += 1

    def step(self):
        return {"ok": self.ok, "kl_mean": 0.0}


@pytest.mark.parametrize("ok", [True, False])
def test_math_changing_candidate_is_kept_only_if_the_guard_passes(tmp_path, ok):
    cur = {"v": "lfu"}
    knob = Knob("policy", "replay", lambda: cur["v"], lambda v: cur.update(v=v), changes_math=True)
    g = FakeGuard(ok)
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Now(), guard=g)
    tuner.idle(); tuner.idle()
    assert tuner.trial["phase"] == "guard" and cur["v"] != "lfu" and g.started == 1
    tuner.idle()
    assert (cur["v"] != "lfu") is ok and tuner.trial is None


def test_math_changing_knob_off_when_guard_is_not_quiet(tmp_path):
    knob = Knob("policy", "replay", lambda: "lfu", lambda v: None, changes_math=True)
    g = FakeGuard(True)
    g.math_ok = False
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Now(), guard=g)
    tuner.idle()
    assert tuner.trial is None


def test_state_round_trip(tmp_path):
    cur = {"v": "a"}
    Tuner([_live(cur)], lambda: None, tmp_path, "fp").save()
    cur["v"] = "b"
    Tuner([_live(cur)], lambda: None, tmp_path, "fp").load_state()
    assert cur["v"] == "a"


def test_state_fingerprint_mismatch_is_ignored(tmp_path):
    cur = {"v": "a"}
    Tuner([_live(cur)], lambda: None, tmp_path, "fp").save()
    cur["v"] = "b"
    with pytest.warns(UserWarning, match="different pack or config"):
        Tuner([_live(cur)], lambda: None, tmp_path, "other").load_state()
    assert cur["v"] == "b"


def test_saved_value_that_no_longer_applies_is_dropped(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"fingerprint": "fp", "values": {"x": "huge"}}))

    def apply(v):
        raise ValueError("over the cap")

    with pytest.warns(UserWarning, match="saved x dropped"):
        Tuner([Knob("x", "live", lambda: "a", apply, lambda: None)], lambda: None, tmp_path, "fp").load_state()


def test_stats_windows_are_logged(tmp_path):
    t, hm = [0.0], [0, 0]
    tuner = Tuner([], lambda: None, tmp_path, "fp", clock=lambda: t[0], counts=lambda: tuple(hm))
    tuner.on_token()
    for _ in range(Tuner.STATS):
        t[0] += 0.1; hm[0] += 3; hm[1] += 1
        tuner.on_token()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["tok_s"] == pytest.approx(10.0) and rec["hit_rate"] == 0.75
```

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_tuner.py -q`
Expected: collection error `cannot import name 'Knob'`.

- [ ] **Step 3: Implement** (append to `moestream/tune.py`; add `import json, time, warnings`, `from dataclasses import dataclass`, `from pathlib import Path`, `from typing import Callable` at the top)

```python
@dataclass
class Knob:
    name: str
    tier: str                         # "replay": judged by simulation; "live": by A/B timing
    get: Callable
    apply: Callable                   # between tokens; must not exceed memory_gb
    propose: Callable = None          # live knobs: a candidate value, or None
    changes_math: bool = False        # live knobs never do; replay ones may (slot split)
    ready: Callable = lambda: True    # False pauses the knob (e.g. slots shrunk under pressure)


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


class Tuner:
    WINDOW = 64       # tokens per A/B window
    STATS = 256       # tokens per logged stats window
    MIN_GAIN = 0.01   # replay: candidate miss rate must be at least 1% lower

    def __init__(self, knobs, snapshot, dirpath, fingerprint, *, guard=None, pool=None, clock=time.perf_counter,
                 counts=lambda: (0, 0), trial=True, retry_tokens=50_000):
        self.knobs = {k.name: k for k in knobs}
        self.order = list(self.knobs)
        self.snapshot, self.guard, self.pool, self.clock, self.counts = snapshot, guard, pool, clock, counts
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fp, self.trial_on, self.retry_tokens = fingerprint, trial, retry_tokens
        self.trial, self.disabled, self.rejected, self.next_i = None, set(), {}, 0
        self.tokens, self.last_tok, self.quiet = 0, None, False
        self.ab_s = self.ab_n = self.st_s = self.st_n = 0
        self.last_counts, self.window_hit = counts(), 0.0

    # generation thread, per decode token
    def on_token(self):
        if self.quiet:  # the guard's own forward passes are not traffic
            return
        now = self.clock()
        last, self.last_tok = self.last_tok, now
        self.tokens += 1
        if last is None:
            return
        dt = now - last
        self.ab_s += dt; self.ab_n += 1
        if self.ab_n == self.WINDOW:
            self._window(self.ab_s)
            self.ab_s = self.ab_n = 0
        self.st_s += dt; self.st_n += 1
        if self.st_n == self.STATS:
            h, m = self.counts()
            dh, dm = h - self.last_counts[0], m - self.last_counts[1]
            self.last_counts, self.window_hit = (h, m), dh / max(1, dh + dm)
            self.log({"tok_s": self.st_n / max(self.st_s, 1e-9), "hit_rate": self.window_hit, "tokens": self.tokens})
            self.st_s = self.st_n = 0

    def interrupt(self):
        """Prefill or idle: the token stream broke. Drop the partial window and any half pair."""
        self.last_tok = None
        self.ab_s = self.ab_n = 0
        t = self.trial
        if t and t["phase"] == "ab" and t["pending"] is not None:
            t["pending"] = None
            self._switch(t["old"])

    # generation thread, no request waiting; one bounded slice
    def idle(self):
        self.interrupt()
        g = self.guard
        if g is not None and not g.ready:
            self._guard_step()
            if g.ready:
                self.load_state()
            return
        t = self.trial
        if t is None:
            self._start_next()
        elif t["phase"] == "replay" and t["future"].done():
            self._replay_done(t)
        elif t["phase"] == "guard":
            v = self._guard_step()
            if v is not None:
                self._finish(v["ok"], **v)

    def _guard_step(self):
        self.quiet = True
        try:
            return self.guard.step()
        finally:
            self.quiet = False

    def _start_next(self):
        if not self.trial_on:
            return
        for _ in range(len(self.order)):
            name = self.order[self.next_i]
            self.next_i = (self.next_i + 1) % len(self.order)
            k = self.knobs[name]
            if name in self.disabled or not k.ready() or not self._eligible(name):
                continue
            if k.changes_math and not (self.guard and self.guard.math_ok):
                continue
            old = k.get()
            if k.tier == "replay":
                snap = self.snapshot()
                if snap is None:
                    continue
                self.trial = dict(knob=name, old=old, new=None, phase="replay",
                                  future=self.pool.submit(replay_trial, name, snap))
            else:
                new = k.propose()
                if new is None or new == old:
                    continue
                self.trial = dict(knob=name, old=old, new=new, phase="ab", pending=None, pairs=[])
            return

    def _eligible(self, name):
        r = self.rejected.get(name)
        return r is None or self.tokens - r[0] >= self.retry_tokens or abs(self.window_hit - r[1]) > 0.05

    def _window(self, secs):
        t = self.trial
        if t is None or t["phase"] != "ab":
            return
        if t["pending"] is None:   # the current arm just ran; run the candidate next
            t["pending"] = secs
            self._switch(t["new"])
            return
        t["pairs"].append((t["pending"], secs))
        t["pending"] = None
        if self._switch(t["old"]):
            keep = ab_keep(t["pairs"])
            if keep is not None:
                self._finish(keep, pairs=t["pairs"])

    def _replay_done(self, t):
        e = t["future"].exception()
        if e is not None:
            self._finish(False, error=repr(e))
            return
        value, cur, cand = t["future"].result()
        t["new"], nums = value, {"miss_rate": cur, "candidate_miss_rate": cand}
        if value is None or cand > cur * (1 - self.MIN_GAIN):
            self._finish(False, **nums)
        elif self.knobs[t["knob"]].changes_math:
            if self._switch(value):
                t["phase"] = "guard"
                self.guard.start()
        else:
            self._finish(True, **nums)

    def _set(self, name, value, disable=True):
        try:
            self.knobs[name].apply(value)
            return True
        except Exception as e:
            self.log({"knob": name, "value": value, "error": repr(e)})
            if disable:
                self.disabled.add(name)
            return False

    def _switch(self, value):
        """Set the trial's knob; on failure revert it, disable it and end the trial."""
        t = self.trial
        if self._set(t["knob"], value):
            return True
        self._set(t["knob"], t["old"], disable=False)
        self.trial = None
        return False

    def _finish(self, keep, **nums):
        t, self.trial = self.trial, None
        name = t["knob"]
        if keep and not self._set(name, t["new"]):
            keep = False
        if not keep:
            self._set(name, t["old"], disable=False)
            self.rejected[name] = (self.tokens, self.window_hit)
        else:
            self.rejected.pop(name, None)
        self.log({"knob": name, "from": t["old"], "to": t["new"], "kept": keep, **nums})
        self.save()

    def log(self, record):
        with open(self.dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.time(), **record}, default=_plain) + "\n")

    def save(self):
        values = {n: k.get() for n, k in self.knobs.items()}
        (self.dir / "state.json").write_text(json.dumps({"fingerprint": self.fp, "values": values}, default=_plain))

    def load_state(self):
        try:
            s = json.loads((self.dir / "state.json").read_text())
        except FileNotFoundError:
            return
        except Exception as e:
            warnings.warn(f"tuning state unreadable ({e!r}); starting from defaults")
            return
        if s.get("fingerprint") != self.fp:
            warnings.warn("tuning state is for a different pack or config; starting from defaults")
            return
        for name, v in s.get("values", {}).items():
            if name in self.knobs and not self._set(name, v, disable=False):
                warnings.warn(f"saved {name} dropped: it no longer applies")
```


- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add moestream/tune.py tests/test_tuner.py
git commit -m "feat: tuner trial loop with replay and A/B judges, KL-guarded math changes, state and log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The five system knobs

**Files:**
- Create: `moestream/knobs.py`
- Test: `tests/test_knobs.py`

**Interfaces:**
- Consumes: `Knob` (Task 6); `StreamingMoE.resize`, `io.lookahead`, `prefetch.threads/set_threads` (Task 2); `SlotStore.set_policy/set_half_life` (Task 1).
- Produces: `system_knobs(model, max_slots, ready=lambda: True) -> [Knob]` named `policy`, `half_life`, `slots`, `prefetch`, `threads`. Split value: `{layer_index: slots}`; prefetch value: sorted list of layer indices with lookahead on.

- [ ] **Step 1: Write the failing tests** (`tests/test_knobs.py`)

```python
import mlx.core as mx
import numpy as np
import pytest

from moestream import prefetch
from moestream.knobs import system_knobs
from moestream.layer import StreamingMoE
from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir


@pytest.fixture
def bf16(tmp_path):
    d, resident = quantized_model_dir(tmp_path, dtype=mx.bfloat16)
    pack(d, tmp_path / "packed")
    return tmp_path / "packed", resident


def _decode(model, ids, split, between=lambda i: None):
    cache = model.make_cache()
    got = [model(ids[:, :split], cache=cache)]
    for i in range(split, ids.shape[1]):
        between(i)
        got.append(model(ids[:, i : i + 1], cache=cache))
    return mx.concatenate(got, axis=1)


@pytest.mark.parametrize("name, value", [
    ("policy", "lru"), ("half_life", 4.0), ("slots", {1: 2, 2: 6, 3: 4}), ("prefetch", []), ("threads", 2),
])
def test_changing_a_knob_mid_decode_is_bit_exact(bf16, name, value):
    out, resident = bf16
    m, _ = load(out, slots=4)
    k = {k.name: k for k in system_knobs(m, max_slots=12)}[name]
    ids = mx.array(np.random.default_rng(3).integers(0, TINY["vocab_size"], size=(1, 16)))
    old = prefetch.threads()
    try:
        got = _decode(m, ids, 8, lambda i: k.apply(value) if i == 11 else None)
    finally:
        prefetch.set_threads(old)
    assert mx.array_equal(got, _decode(resident, ids, 8)).item()
    assert name == "threads" or k.get() == value


def test_split_over_the_cap_or_wrong_layers_raises(bf16):
    m, _ = load(bf16[0], slots=4)
    k = {k.name: k for k in system_knobs(m, max_slots=12)}["slots"]
    with pytest.raises(ValueError, match="cap"):
        k.apply({1: 6, 2: 6, 3: 6})
    with pytest.raises(ValueError, match="layers"):
        k.apply({1: 4, 2: 4})
    assert k.get() == {1: 4, 2: 4, 3: 4}


def test_split_accepts_string_keys_from_json(bf16):
    m, _ = load(bf16[0], slots=4)
    k = {k.name: k for k in system_knobs(m, max_slots=12)}["slots"]
    k.apply({"1": 3, "2": 5, "3": 4})
    assert k.get() == {1: 3, 2: 5, 3: 4}


def test_prefetch_and_thread_proposals(bf16):
    m, _ = load(bf16[0], slots=4)
    ks = {k.name: k for k in system_knobs(m, max_slots=12)}
    first = ks["prefetch"].propose()
    assert set(first) < set(ks["prefetch"].get())  # one quarter flipped off
    old = prefetch.threads()
    proposals = {ks["threads"].propose(), ks["threads"].propose()}
    assert proposals == {old * 2, old // 2}
```

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_knobs.py -q`
Expected: collection error `No module named 'moestream.knobs'`.

- [ ] **Step 3: Implement** (`moestream/knobs.py`)

```python
"""The system-level knobs the tuner turns: eviction policy, half-life, slot split, lookahead prefetch, read threads."""
import numpy as np

from moestream import prefetch
from moestream.layer import StreamingMoE
from moestream.tune import Knob


def system_knobs(model, max_slots, ready=lambda: True):
    by = {i: l.mlp for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)}
    layers, moe = list(by), list(by.values())
    E = moe[0].io.layout["num_experts"]
    pre = [l for l in layers if by[l].io.next is not None]
    quarters = [[int(l) for l in q] for q in np.array_split(pre, 4) if len(q)]
    turn = {"q": 0, "t": 0}

    def set_policy(p):
        for m in moe:
            m.io.store.set_policy(p)

    def set_half_life(h):
        for m in moe:
            m.io.store.set_half_life(float(h))

    def get_split():
        return {l: m.io.store.slots for l, m in by.items()}

    def set_split(split):
        split = {int(l): int(n) for l, n in split.items()}
        if sorted(split) != layers or not all(1 <= n <= E for n in split.values()):
            raise ValueError(f"slot split {split} does not fit layers {layers} with {E} experts")
        if sum(split.values()) > max_slots:
            raise ValueError(f"slot split needs {sum(split.values())} slots; the cap allows {max_slots}")
        for l in sorted(split, key=lambda l: split[l] - by[l].io.store.slots):  # shrink before growing
            if split[l] != by[l].io.store.slots:
                by[l].resize(split[l])

    def get_prefetch():
        return [l for l in pre if by[l].io.lookahead]

    def set_prefetch(on):
        on = {int(l) for l in on}
        for l in pre:
            by[l].io.lookahead = l in on

    def propose_prefetch():
        q = quarters[turn["q"] % len(quarters)]
        turn["q"] += 1
        return sorted(set(get_prefetch()) ^ set(q))

    def propose_threads():
        turn["t"] ^= 1
        n = prefetch.threads() * 2 if turn["t"] else prefetch.threads() // 2
        return n if 2 <= n <= 16 else None

    store = lambda: moe[0].io.store  # noqa: E731
    return [
        Knob("policy", "replay", lambda: store().policy, set_policy),
        Knob("half_life", "replay", lambda: store().half_life, set_half_life),
        Knob("slots", "replay", get_split, set_split, changes_math=True, ready=ready),
        Knob("prefetch", "live", get_prefetch, set_prefetch, propose_prefetch),
        Knob("threads", "live", prefetch.threads, prefetch.set_threads, propose_threads),
    ]
```

- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass. The tiny model's MoE layers are 1–3 (`mlp_only_layers: [0]`); if `load` names them differently, fix the test's layer keys from `{i for i, l in enumerate(m.layers) if isinstance(l.mlp, StreamingMoE)}`.

- [ ] **Step 5: Commit**

```bash
git add moestream/knobs.py tests/test_knobs.py
git commit -m "feat: system knobs: eviction policy, half-life, slot split, per-layer prefetch, read threads

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Memory-pressure guard

**Files:**
- Create: `moestream/pressure.py`
- Test: `tests/test_pressure.py`

**Interfaces:**
- Consumes: the `slots` knob (Task 7); `budget.free_bytes`, `budget.GB`.
- Produces: `swap_used() -> float` bytes; `shrink(split, floor, frac=0.1) -> dict`; `Pressure(knob, floor, *, every=256, low=1*GB, high=3*GB, calm_s=300, free=free_bytes, swap=swap_used, clock=time.monotonic, log=print)` with `on_token()` and `.shrunk`.

- [ ] **Step 1: Write the failing tests** (`tests/test_pressure.py`)

```python
from moestream.budget import GB
from moestream.pressure import Pressure, shrink, swap_used


def test_shrink_takes_ten_percent_from_the_largest_layers():
    s = shrink({1: 50, 2: 30, 3: 20}, floor=16)
    assert sum(s.values()) == 90 and s[1] < 50 and s[3] == 20


def test_shrink_never_goes_under_the_floor():
    assert shrink({1: 16, 2: 16}, floor=16) == {1: 16, 2: 16}


def test_swap_used_reads_sysctl():
    assert swap_used() >= 0


class Split:
    def __init__(self):
        self.v = {1: 100, 2: 100}

    def get(self):
        return dict(self.v)

    def apply(self, v):
        self.v = dict(v)


def test_low_memory_shrinks_and_calm_restores():
    k, mem, now, logs = Split(), {"free": 0.5 * GB, "swap": 0.0}, [0.0], []
    p = Pressure(k, 16, every=1, free=lambda: mem["free"], swap=lambda: mem["swap"], clock=lambda: now[0],
                 log=logs.append)
    p.on_token()
    assert p.shrunk and sum(k.v.values()) == 180
    mem["free"] = 4 * GB
    p.on_token(); now[0] += 299; p.on_token()
    assert p.shrunk
    now[0] += 2; p.on_token()
    assert not p.shrunk and k.v == {1: 100, 2: 100} and len(logs) == 2


def test_growing_swap_shrinks_even_with_free_memory():
    k, mem = Split(), {"swap": 0.0}
    p = Pressure(k, 16, every=1, free=lambda: 8 * GB, swap=lambda: mem["swap"], clock=lambda: 0.0, log=lambda r: None)
    p.on_token()
    assert not p.shrunk
    mem["swap"] = 1e6
    p.on_token()
    assert p.shrunk and sum(k.v.values()) < 200
```

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_pressure.py -q`
Expected: collection error `No module named 'moestream.pressure'`.

- [ ] **Step 3: Implement** (`moestream/pressure.py`)

```python
"""Shrink expert slots when macOS runs short of memory; grow back once it has been calm. A guard, not a trial."""
import re
import subprocess
import time

from moestream.budget import GB, free_bytes


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1)) * 2**20


def shrink(split, floor, frac=0.1):
    s = dict(split)
    for _ in range(max(1, round(sum(s.values()) * frac))):
        l = max(s, key=s.get)
        if s[l] <= floor:
            break
        s[l] -= 1
    return s


class Pressure:
    def __init__(self, knob, floor, *, every=256, low=1 * GB, high=3 * GB, calm_s=300, free=free_bytes,
                 swap=swap_used, clock=time.monotonic, log=print):
        self.knob, self.floor, self.every, self.low, self.high, self.calm_s = knob, floor, every, low, high, calm_s
        self.free, self.swap, self.clock, self.log = free, swap, clock, log
        self.n, self.target, self.calm_since, self.last_swap = 0, None, None, swap()

    @property
    def shrunk(self):
        return self.target is not None

    def on_token(self):
        self.n += 1
        if self.n % self.every:
            return
        free, swap, now = self.free(), self.swap(), self.clock()
        grew, self.last_swap = swap > self.last_swap, swap
        if grew or free < self.low:
            if self.target is None:
                self.target = self.knob.get()
            self.knob.apply(shrink(self.knob.get(), self.floor))
            self.calm_since = None
            self.log({"pressure": "shrink", "free_gb": free / GB, "swap_gb": swap / GB, "slots": sum(self.knob.get().values())})
        elif self.shrunk:
            if free <= self.high:
                self.calm_since = None
            elif self.calm_since is None:
                self.calm_since = now
            elif now - self.calm_since >= self.calm_s:
                self.knob.apply(self.target)
                self.target = None
                self.log({"pressure": "restore", "free_gb": free / GB, "slots": sum(self.knob.get().values())})
```

- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add moestream/pressure.py tests/test_pressure.py
git commit -m "feat: memory-pressure guard shrinks slots on low memory or swap growth, restores when calm

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Wire tuning into the server

**Files:**
- Modify: `moestream/tune.py` (add `start`, `fingerprint`), `tools/serve.py`
- Test: `tests/test_tuner.py`, `tests/test_serve.py`

**Interfaces:**
- Consumes: everything above; `stream_stats`; `scripts.check_parity.PROMPTS`.
- Produces: `tune.fingerprint(cfg) -> str`; `tune.start(model, tok, cfg) -> Tuner` (hooks installed; state loaded now when `tune = false`, after the reference otherwise); `tools.serve.TUNER` (list holding the tuner); `patch_server` also wraps `ResponseGenerator._next_request` so an empty poll runs `TUNER[0].idle()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tuner.py`:

```python
def test_start_wires_hooks_traces_and_resumes_state(tmp_path):
    import mlx.core as mx

    from moestream.config import EngineConfig
    from moestream.layer import StreamingMoE
    from moestream.load import load
    from moestream.tune import Ring, start
    from scripts.pack_experts import pack
    from tests.tiny import quantized_model_dir

    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    cfg = EngineConfig(pack_dir=str(tmp_path / "packed"), tuning_dir=str(tmp_path / "t"), tune=False, kv_bits=0)
    tuner = start(m, None, cfg)
    moe = [l.mlp for l in m.layers if isinstance(l.mlp, StreamingMoE)]
    assert moe[-1].io.hook is not None and all(x.io.trace is None for x in moe)  # tune=false: no traces
    tuner.knobs["policy"].apply("fifo")
    tuner.save()
    m2, _ = load(tmp_path / "packed", slots=4)
    start(m2, None, cfg)
    assert {l.mlp.io.store.policy for l in m2.layers if isinstance(l.mlp, StreamingMoE)} == {"fifo"}
    cfg.tune = True
    m3, _ = load(tmp_path / "packed", slots=4)
    t3 = start(m3, None, cfg, prompts=[[1, 2, 3, 4]], probe_tokens=8)
    assert isinstance(m3.layers[1].mlp.io.trace, Ring)
    m3(mx.array([[1, 2, 3]]))
    assert t3.tokens == 0  # prefill is not a decode token
    t3.pool.shutdown()
```

Append to `tests/test_serve.py`:

```python
def test_idle_polls_drive_the_tuner(monkeypatch):
    from types import SimpleNamespace

    from mlx_lm.models.cache import LRUPromptCache

    import tools.serve as serve

    class RG:
        def _next_request(self, timeout=None):
            return self.queued

    server = SimpleNamespace(LRUPromptCache=LRUPromptCache, load=None, ResponseGenerator=RG)
    serve.patch_server(server, EngineConfig(pack_dir="/p"))
    idles = []
    monkeypatch.setattr(serve, "TUNER", [SimpleNamespace(idle=lambda: idles.append(1))])
    rg = RG()
    rg.queued = None
    assert rg._next_request(0.1) is None and idles == [1]
    rg.queued = "req"
    assert rg._next_request(0.1) == "req" and idles == [1]
```

Update the two existing tests that build a fake `server` in `tests/test_serve.py` to add `ResponseGenerator=type("RG", (), {"_next_request": lambda self, timeout=None: None})`, and in `test_loader_loads_once_whatever_model_name_a_client_sends` add `monkeypatch.setattr("moestream.tune.start", lambda *a, **k: None)`.

- [ ] **Step 2: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_tuner.py tests/test_serve.py -q`
Expected: failures: `cannot import name 'start'`; `RG` has no wrapper so `idles == []`.

- [ ] **Step 3: Implement**

Append to `moestream/tune.py` (add `import hashlib, os` and `from concurrent.futures import ProcessPoolExecutor` at the top):

```python
def fingerprint(cfg):
    layout = (Path(cfg.pack_dir) / "layout.json").read_bytes()
    return hashlib.sha256(layout + json.dumps([cfg.memory_gb, cfg.kv_bits, cfg.context_tokens]).encode()).hexdigest()[:16]


def start(model, tok, cfg, prompts=None, probe_tokens=2048):
    """Wire tuning into a loaded engine: knobs, routing traces, quality guard, pressure guard, saved state."""
    from moestream.guard import Guard
    from moestream.knobs import system_knobs
    from moestream.layer import StreamingMoE, stream_stats
    from moestream.pressure import Pressure
    from tools.calibrate import GRID

    moe = [(i, l.mlp) for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)]
    E = moe[0][1].io.layout["num_experts"]
    box = {}
    knobs = system_knobs(model, sum(m.io.store.slots for _, m in moe),
                         ready=lambda: not box["pressure"].shrunk)

    def snapshot():
        rows = {i: m.io.trace.rows() for i, m in moe}
        if any(r is None or len(r) < 1024 for r in rows.values()):
            return None
        st = moe[0][1].io.store
        return dict(rows=rows, slots={i: m.io.store.slots for i, m in moe}, policy=st.policy,
                    half_life=st.half_life, num_experts=E, grid=[g for g in GRID if g <= E])

    def counts():
        s = stream_stats(model)
        return s["hits"], s["misses"]

    d, guard, pool = Path(cfg.tuning_dir), None, None
    if cfg.tune:
        for _, m in moe:
            m.io.trace = Ring(cfg.trace_tokens)
        if prompts is None:
            from scripts.check_parity import PROMPTS
            prompts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
                       for p in PROMPTS]
        layout = (Path(cfg.pack_dir) / "layout.json").read_bytes()
        key = hashlib.sha256(layout + json.dumps([cfg.kv_bits, prompts, probe_tokens]).encode()).hexdigest()[:16]
        guard = Guard(model, prompts, d / "reference.npz", key, total=probe_tokens)
        pool = ProcessPoolExecutor(2, initializer=os.nice, initargs=(10,))
    tuner = Tuner(knobs, snapshot, d, fingerprint(cfg), guard=guard, pool=pool, counts=counts, trial=cfg.tune)
    box["pressure"] = Pressure(next(k for k in knobs if k.name == "slots"), min(16, E), log=tuner.log)
    if guard is None or guard.ready:
        tuner.load_state()  # else it waits for the untuned reference to be built (Tuner.idle)

    def hook(decode):
        if decode:
            tuner.on_token()
            box["pressure"].on_token()
        else:
            tuner.interrupt()

    moe[-1][1].io.hook = hook
    return tuner
```

`tools/serve.py`: add `import warnings`, a module-level `TUNER = []`, and in `make_loader` after `loaded.append((model, tok))`:

```python
        try:
            from moestream.tune import start

            TUNER.append(start(model, tok, cfg))
        except Exception as e:  # tuning never stops serving
            warnings.warn(f"tuning off: {e!r}")
```

In `patch_server`, add:

```python
    nxt = server.ResponseGenerator._next_request

    def _next_request(self, timeout=None):
        r = nxt(self, timeout)
        if r is None and TUNER:
            TUNER[0].idle()
        return r

    server.ResponseGenerator._next_request = _next_request
```

Update the `patch_server` docstring: "...and run one slice of tuner work whenever the generation thread polls an empty queue."

- [ ] **Step 4: Run them**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add moestream/tune.py tools/serve.py tests/test_tuner.py tests/test_serve.py
git commit -m "feat: serve runs the tuner on the generation thread: per-token hook and idle slices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Verify on real weights (gated)

Starts only after sub-project 1's Task 6 is done and the coordinator says go. Tell the coordinator before each real run.

**Files:**
- Create: `tools/drive.py`
- Test: `tests/test_drive.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `tools.drive.turns(msgs) -> [messages prefix]`; `python3 -m tools.drive --minutes 60` posts each captured session's assistant turns to the server.

- [ ] **Step 1: Ask the coordinator** to add `replay.session.load_session` to RULES.md rule 7 (frozen interfaces). If refused, drive with `PROMPTS` multi-turn instead and ledger it.

- [ ] **Step 2: Write the failing test** (`tests/test_drive.py`)

```python
from tools.drive import turns


def test_each_assistant_turn_becomes_a_request_of_what_came_before():
    msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}, {"role": "assistant"}]
    assert [len(t) for t in turns(msgs)] == [2, 4]
```

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_drive.py -q`
Expected: `No module named 'tools.drive'`.

- [ ] **Step 3: Implement** (`tools/drive.py`)

```python
"""Drive the server with lagunaLab's captured agent sessions, one request per assistant turn.

PYTHONPATH=.:../lagunaLab python3 -m tools.drive [--minutes 60] [--url http://127.0.0.1:8081/v1]
Light client: the server holds the GPU.
"""
import argparse
import glob
import json
import time
import urllib.request


def turns(msgs):
    return [msgs[:i] for i, m in enumerate(msgs) if m["role"] == "assistant" and i]


def main():
    from replay.session import load_session

    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="../lagunaLab/traces")
    p.add_argument("--url", default="http://127.0.0.1:8081/v1")
    p.add_argument("--minutes", type=float, default=60)
    a = p.parse_args()
    end = time.time() + 60 * a.minutes
    while time.time() < end:
        for f in sorted(glob.glob(f"{a.traces}/*.jsonl")):
            msgs, tools = load_session(f)
            for prefix in turns(msgs):
                if time.time() >= end:
                    return
                body = {"model": "Laguna-XS-2.1", "messages": prefix, "max_tokens": 256, "tools": tools}
                req = urllib.request.Request(f"{a.url}/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=600).read()
            print(f, flush=True)


if __name__ == "__main__":
    main()
```

Run the test again. Expected: pass. Commit `tools/drive.py tests/test_drive.py` ("feat: tools.drive replays captured sessions against the server").

- [ ] **Step 4: Real run** (coordinator go first)

```bash
PYTHONPATH=.:../lagunaLab /Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe tools/watch.sh 8 21 python3 -m tools.serve > results/tune_serve.log 2>&1 &
# wait for "moeStream:" banner and the reference build (tuning/reference.npz), then:
PYTHONPATH=.:../lagunaLab python3 -m tools.drive --minutes 60 > results/tune_drive.log 2>&1
```

Expected: `tuning/reference.npz` written with `math_ok` true (if false, record the noise numbers: math-changing knobs stayed off); `tuning/log.jsonl` has stats windows and trials.

- [ ] **Step 5: Read the gate from the log**

```bash
python3 -c "
import json, numpy as np
r=[json.loads(l) for l in open('tuning/log.jsonl')]
w=[x for x in r if 'tok_s' in x]; n=max(1,len(w)//10)
f=lambda xs,k: float(np.median([x[k] for x in xs]))
print('first', f(w[:n],'tok_s'), f(w[:n],'hit_rate'), 'last', f(w[-n:],'tok_s'), f(w[-n:],'hit_rate'))
print('trials', [(x['knob'], x['kept']) for x in r if 'kept' in x])
print('pressure', [x for x in r if 'pressure' in x])"
```

Pass: last tok/s > first, last hit rate ≥ first, every kept math-changing trial has `ok: true`, `results/tune_serve.log` shows no `aborted: paging`, peak footprint ≤ 20 GB. Kill the server by PID.

- [ ] **Step 6: Restart check.** Start the server again, drive 10 minutes; first windows' median tok/s within 5% of the previous run's last windows. Kill by PID.

- [ ] **Step 7: README** — add a "Self-tuning" section: what it tunes, the KL thresholds, `tune`/`tuning_dir`/`trace_tokens`, the measured first/last numbers and kept trials. Commit `README.md results/tune_*.log` ("docs: self-tuning verification numbers"). Report to the coordinator.

---

## Part 2: SSD tier for the prompt cache (Tasks 11-18)

**Spec:** `docs/specs/2026-10-14-ssd-kv-tier-design.md` (draft, waiting for the user's review). The
spec is the authority for formats, defaults and behaviour; these tasks only cut it into reviewable
pieces and are not copied from it.

**Status:** structure only. Each task names its files, interfaces and tests; the step code is written
once the user has reviewed the spec. No task starts before that review.

**Global constraints (additional):** everything under Global Constraints above; no writes outside
`kv_dir`; no extra KV copy in memory (a restore takes the place of the RAM-hit deep copy); every
disk write is temp file + `fsync` + atomic rename; tiny-model tests use a sliding window smaller than
the test prompts so the rotating buffer wraps.

### Task 11: Config keys for the SSD tier

**Files:** Modify `moestream/config.py`, `moestream.toml`, `.gitignore`; Test `tests/test_config.py`

**Interfaces:**
- Produces: `EngineConfig.kv_tier: bool = True`, `kv_dir: str = "kvcache"` (resolved against the TOML's directory), `kv_disk_gb: float = 10.0` (> 0).

- [ ] Write failing tests: defaults; relative `kv_dir` resolves against the file; `kv_disk_gb = 0` and a non-bool `kv_tier` are rejected naming the key.
- [ ] Run them, watch them fail; implement; run the suite; add `kvcache/` to `.gitignore`; commit.

### Task 12: Entry format, chain manifests and invalidation

**Files:** Create `moestream/kvtier.py`; Test `tests/test_kvtier.py`

**Interfaces:**
- Produces: `fingerprint(pack_dir, kv_bits) -> str`; `Store(dir, fingerprint)` that at construction wipes the directory on a `FORMAT` mismatch and deletes `.tmp` files and files no manifest names; `Store.chains -> {chain_id: Manifest}`; `Store.write_atomic(path, write_fn)`.
- Consumes: the cache list shape of `model.make_cache()` (`state` and `meta_state` per layer).

- [ ] Failing tests (spec tests 6 and 7): leftover `.tmp` ignored and deleted; manifest naming a missing file drops the chain; corrupt manifest drops the chain; a different `kv_bits`, layout hash or format version wipes the directory.
- [ ] Implement the `FORMAT` file, per-chain `manifest.json` and the atomic write helper; run the suite; commit.

### Task 13: Snapshot writer

**Files:** Modify `moestream/kvtier.py`; Test `tests/test_kvtier.py`

**Interfaces:**
- Produces: `Store.save(tokens, cache)` (queues and returns at once); one writer thread; full-attention layers appended as `full_NNN.safetensors` for the new token range only; sliding layers written whole as `sliding_<tokens>.safetensors`; manifest replaced last.
- Consumes: Task 12's `write_atomic`, manifests.

- [ ] Failing tests (spec tests 3 and 10, part of 11): three turns write three full-attention segments whose bytes equal the cache's full-attention bytes at turn 3; a slow fake writer does not delay `save`; a second snapshot for the same chain replaces a queued one; three failing writes turn writes off; a snapshot that does not extend a chain starts a new chain.
- [ ] Implement; run the suite; commit.

### Task 14: Prefix-hash index and lookup

**Files:** Modify `moestream/kvtier.py`; Test `tests/test_kvtier.py`

**Interfaces:**
- Produces: `Store.lookup(tokens, ram_prefix, context_tokens) -> (n_tokens, chain_id) | None`, hashing the prompt once in a single pass at every snapshot length the index holds; `MIN_RESTORE_TOKENS = 2048`.
- Consumes: Task 12/13 manifests (token count and SHA-256 per snapshot).

- [ ] Failing tests (spec test 5): longest matching snapshot wins; a match under `ram_prefix + MIN_RESTORE_TOKENS` returns None and reads no KV; an entry longer than `context_tokens` is skipped; a branched history matches only up to its last common snapshot.
- [ ] Implement; run the suite; commit.

### Task 15: Restore path

**Files:** Modify `moestream/kvtier.py`; Test `tests/test_kvtier.py`

**Interfaces:**
- Produces: `Store.restore(chain_id, n_tokens, cache)`: fills a fresh `model.make_cache()` list in place, reading one array at a time with `F_NOCACHE` into one reusable host buffer, truncating full-attention KV to `n_tokens` and loading the sliding snapshot for that point.
- Consumes: Task 13's files; lagunaLab's `make_quantized_cache` caches (frozen interface), never mlx-lm's `load_prompt_cache`.

- [ ] Failing tests (spec tests 1, 9, part of 11): save then restore gives bit-identical next-token logits at `kv_bits` 0 and 8; restoring turn 2 of a 3-turn chain equals the cache at turn 2; at most one host buffer of the largest array's size is allocated (injected allocator); a failing read drops the chain and returns None.
- [ ] Implement; run the suite; commit.

### Task 16: Disk cap and pruning

**Files:** Modify `moestream/kvtier.py`; Test `tests/test_kvtier.py`

**Interfaces:**
- Produces: `Store.prune(cap_bytes)`, run by the writer after each write: deletes whole chains, oldest last use first, until under the cap; sliding snapshots kept: the latest per chain plus every 8th turn.

- [ ] Failing tests (spec tests 4 and 8): writing past the cap deletes whole chains in last-use order; only the latest and every 8th sliding snapshot survive.
- [ ] Implement; run the suite; commit.

### Task 17: Wire the tier into the server

**Files:** Modify `tools/serve.py`; Test `tests/test_serve.py`; Modify `docs/serving.md`, `docs/configuration.md`

**Interfaces:**
- Produces: the `LRUPromptCache` wrapper in `patch_server` consults `Store.lookup` after mlx-lm's own lookup in `fetch_nearest_cache` and restores into `model.make_cache()` when it wins; `insert_cache` calls `Store.save`; `kv_tier = false` leaves today's behaviour unchanged.
- Consumes: Tasks 11-16.

- [ ] Failing tests (spec test 2): on the tiny model, turn 2 decoded after an SSD restore gives the same tokens and logits as turn 2 after a RAM-cache hit; with `kv_tier = false` nothing is written; a restore error falls back to full prefill with correct output.
- [ ] Implement; run the suite; document the keys and the restore log line; commit.

### Task 18: Verify on real weights (gated)

**Needs the GPU lock (`.gpu/gpu-run moe`), the watchdog (`tools/watch.sh 4 17` while the user is on the machine) and the coordinator's go before each run.** Kill servers by PID; `ps` check afterwards.

**Files:** Modify `tools/drive.py` (live-use mode: keep the model's own replies in the history, sessions interleaved turn by turn); Modify `README.md`; results under `results/kvtier_*`

- [ ] Run the spec's five checks ("Verification on real weights"): 3 interleaved sessions with the tier off and on (prompt tokens within 10% of the simulated floor, restores at least 10x faster than the prefill they replace, peak footprint at most `memory_gb` + 1.2 GB, no swap growth); restart mid-run resumes from SSD; `kill -9` mid-write recovers to the previous turn; greedy continuation after a restore equals the one after a RAM hit for 3 turns; directory stays under `kv_disk_gb`, bytes written per turn reported.
- [ ] Record the numbers in the README; commit `results/kvtier_*` and the README; report to the coordinator.
