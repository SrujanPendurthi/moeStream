# moeStream Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream Laguna XS 2.1's routed experts from SSD into a free-memory-sized, decayed-LFU slot cache, with the next layer's experts prefetched in the background while the GPU computes, so decode runs ≥2× `lagunaLab` streaming with bit-identical output.

**Architecture:** `StreamingMoE` replaces each Laguna MoE block. One GPU sync per layer returns this layer's routing plus a guess at the next layer's routing (the next router run on this layer's input). Worker threads `preadv` guessed experts into reusable staging buffers while the GPU runs. Slots are filled only when a layer actually uses an expert. Bookkeeping (`SlotStore`), disk I/O (`Prefetcher`) and budgeting (`budget.py`) are separate, independently tested units.

**Tech Stack:** Python 3, MLX 0.32 / mlx-lm 0.31.3, numpy, pytest; `lagunaLab` imported read-only via `pythonpath`.

**Spec:** `docs/specs/2026-09-26-moestream-design.md`

## Global Constraints

- `lagunaLab` is read-only: never write under `../lagunaLab` (including `models/`, `results/`, `.git`). It is imported via `pythonpath = [".", "../lagunaLab"]`. Built against `lagunaLab` commit `7a4ae65` (branch `part1-harness`).
- Output must equal the resident model: `mx.array_equal` on bf16; `assert_allclose(atol=1e-5, rtol=1e-5)` on the f32 tiny model (the same bar `lagunaLab/tests/test_streaming.py` uses).
- Before any real-weight model load (Tasks 5, 7): message sessions `current-41` and `current-0f` via SendMessage, and wait for their go-ahead. Never load while `lagunaLab/scripts/overnight.sh` runs.
- All MLX calls happen on the main thread. Worker threads only run `os.preadv`.
- Prefetch never evicts a slot.
- Default decay half-life is 256 tokens. The default floor is 16 slots per layer (capped at `num_experts`). KV reserve is 40 KB/token × target context (default 32768). Default headroom is 3 GB and the activation reserve is 1 GB.
- Scripts live in `tools/`, not `scripts/`: `scripts` and `tests` are `lagunaLab` package names. `moeStream/tests/` has **no** `__init__.py`, so `tests.tiny` resolves to `lagunaLab`'s regular package.
- Run everything from the `moeStream/` root. Scripts run as `PYTHONPATH=.:../lagunaLab python3 -m tools.<name>`.
- Work on branch `phase1` (spec already on `main`).

## Review Focus

1. **Decode steps followed by a multi-token prefill chunk that needs more experts than there are slots** (an agent turn appending a tool result): leftover predictions from decode must not leak, grouping must stay exact. Test: `test_decode_then_prefill_chunk_is_exact` (Task 4).
2. **A wrong prediction every step for many tokens**: staging buffers must not grow. Test: `test_wrong_predictions_do_not_grow_buffers` (Task 4).
3. **Budget smaller than floor × layers** (another job holds memory): the model must still load at the floor with a warning, not crash or allocate zero slots. Tests: `test_allocate_below_floor_gives_floor` (Task 3), `test_tiny_budget_still_loads_and_warns` (Task 4).
4. **Multi-token steps with a cache** (DFlash/spec-decode verify chunks, L>1 after prefill): no lookahead (lookahead is L==1 only), output still exact. Test: `test_multi_token_cached_steps_are_exact` (Task 4).
5. **Corrupt or missing expert file**: fail at load with the path, never at token 5,000. Tests: `test_missing_expert_file_fails_at_load` (Task 4), `test_short_read_raises_after_retry` (Task 2).

---

### Task 1: Scaffold and SlotStore

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `moestream/__init__.py`, `moestream/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `SlotStore(num_experts: int, slots: int, half_life: float = 256.0)` with `.slots`, `.slot_expert: list[int]`, `.expert_slot: dict[int,int]`, `.hits`, `.misses`, `tick(n_tokens: int = 1) -> None`, and `assign(experts: list[int]) -> tuple[dict[int,int], list[tuple[int,int]]]` (expert→slot for all requested experts, plus the `(expert, slot)` pairs that must be filled from disk).

- [ ] **Step 1: Branch and scaffold**

```bash
cd /Users/srujanpendurthi/Projects/Current/moeStream
git checkout -b phase1
mkdir -p moestream tests tools data results
touch moestream/__init__.py tools/__init__.py
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = [".", "../lagunaLab"]
testpaths = ["tests"]
```

`.gitignore`:
```
__pycache__/
.pytest_cache/
data/routing/
```

- [ ] **Step 2: Verify lagunaLab imports resolve**

Run: `PYTHONPATH=.:../lagunaLab python3 -c "import tests.tiny, laguna_mlx.laguna, scripts.pack_experts, replay.session; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Write the failing tests**

`tests/test_store.py`:
```python
import pytest

from moestream.store import SlotStore


def test_empty_slots_fill_before_eviction():
    s = SlotStore(8, 3)
    s.tick(); _, f = s.assign([0, 1])
    assert f == [(0, 0), (1, 1)]
    s.tick(); _, f = s.assign([2])
    assert f == [(2, 2)]
    assert (s.misses, s.hits) == (3, 0)


def test_never_evicts_current_set():
    s = SlotStore(8, 2)
    s.tick(); s.assign([0, 1])
    s.tick(); slot_of, f = s.assign([1, 2])
    assert f == [(2, 0)]
    assert slot_of == {1: 1, 2: 0} and s.expert_slot == {1: 1, 2: 0}
    assert (s.misses, s.hits) == (3, 1)


def test_lowest_score_evicted_recency_breaks_ties():
    s = SlotStore(8, 3, half_life=float("inf"))
    for _ in range(3):
        s.tick(); s.assign([0])
    s.tick(); s.assign([1])
    s.tick(); s.assign([2])  # scores 0:3, 1:1, 2:1; expert 1 is older
    s.tick(); _, f = s.assign([3])
    assert f == [(3, 1)]


def test_old_heavy_use_decays_away():
    def run(half_life):
        s = SlotStore(8, 2, half_life=half_life)
        for _ in range(60):
            s.tick(); s.assign([0])   # hot long ago
        for _ in range(20):
            s.tick(); s.assign([1])   # warm recently
        s.tick(); _, f = s.assign([2])
        return f
    assert run(float("inf")) == [(2, 1)]  # plain LFU keeps stale expert 0
    assert run(4.0) == [(2, 0)]           # decayed LFU evicts it


def test_too_many_experts_raises():
    with pytest.raises(ValueError):
        SlotStore(8, 2).assign([0, 1, 2])
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python3 -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moestream.store'`

- [ ] **Step 5: Implement**

`moestream/store.py`:
```python
"""Per-layer expert slot bookkeeping: which expert sits in which slot, and who gets evicted.

Pure Python/numpy: no MLX, no threads. Eviction is decayed LFU with a recency tie-break,
so experts that were hot long ago stop pinning slots in long sessions.
"""
import numpy as np


class SlotStore:
    def __init__(self, num_experts, slots, half_life=256.0):
        self.slots = slots
        self.decay = 0.5 ** (1.0 / half_life)
        self.slot_expert = [-1] * slots
        self.expert_slot = {}
        self.score = np.zeros(num_experts)
        self.last = np.zeros(num_experts, np.int64)
        self.clock = 0
        self.hits = self.misses = 0

    def tick(self, n_tokens=1):
        """Age every score by n_tokens tokens. Call once per layer call, before assign()."""
        self.clock += 1
        self.score *= self.decay**n_tokens

    def assign(self, experts):
        """Make `experts` (distinct) resident. Returns ({expert: slot}, [(expert, slot) to fill])."""
        if len(experts) > self.slots:
            raise ValueError(f"{len(experts)} experts > {self.slots} slots")
        keep = set(experts)
        missing = [e for e in experts if e not in self.expert_slot]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        self.score[experts] += 1
        self.last[experts] = self.clock
        fills = list(zip(missing, self._victims(len(missing), keep)))
        for e, s in fills:
            old = self.slot_expert[s]
            if old != -1:
                del self.expert_slot[old]
            self.slot_expert[s], self.expert_slot[e] = e, s
        return {e: self.expert_slot[e] for e in experts}, fills

    def _victims(self, n, keep):
        free = [s for s, e in enumerate(self.slot_expert) if e == -1]
        used = [s for s, e in enumerate(self.slot_expert) if e != -1 and e not in keep]
        used.sort(key=lambda s: (self.score[self.slot_expert[s]], self.last[self.slot_expert[s]]))
        return (free + used)[:n]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_store.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore moestream tools tests/test_store.py
git commit -m "feat: SlotStore with decayed-LFU eviction"
```

---

### Task 2: Prefetcher

**Files:**
- Create: `moestream/prefetch.py`
- Test: `tests/test_prefetch.py`

**Interfaces:**
- Produces: `Prefetcher(path: str | Path, stride: int)` with `.pending: dict[int, tuple[np.ndarray, Future]]`, `.allocated: int` (staging buffers ever created), `submit(experts: list[int]) -> None`, `take(experts: list[int]) -> dict[int, np.ndarray]` (uint8 buffers of length `stride`; the caller must `release()` each one), `release(buf) -> None`, `drop_pending() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_prefetch.py`:
```python
import numpy as np
import pytest

from moestream.prefetch import Prefetcher

STRIDE = 16384


def _file(tmp_path, n=6):
    p = tmp_path / "layer.bin"
    p.write_bytes(b"".join(bytes([e]) * STRIDE for e in range(n)))
    return p


def test_take_returns_right_bytes_prefetched_or_not(tmp_path):
    pf = Prefetcher(_file(tmp_path), STRIDE)
    pf.submit([3])
    got = pf.take([3, 5])
    assert (got[3] == 3).all() and (got[5] == 5).all() and len(got[3]) == STRIDE
    assert pf.pending == {}


def test_buffers_are_reused_not_grown(tmp_path):
    pf = Prefetcher(_file(tmp_path), STRIDE)
    for i in range(100):
        pf.submit([i % 6])
        for buf in pf.take([i % 6, (i + 1) % 6]).values():
            pf.release(buf)
    assert pf.allocated <= 2


def test_drop_pending_recycles_buffers(tmp_path):
    pf = Prefetcher(_file(tmp_path), STRIDE)
    pf.submit([0, 1, 2])
    pf.drop_pending()
    assert pf.pending == {} and len(pf.free) == 3
    pf.submit([4]); pf.drop_pending()
    assert pf.allocated == 3


def test_short_read_raises_after_retry(tmp_path):
    p = tmp_path / "layer.bin"
    p.write_bytes(b"\0" * (STRIDE + 10))  # expert 1 is truncated
    pf = Prefetcher(p, STRIDE)
    with pytest.raises(OSError, match="short read"):
        pf.take([1])


def test_failed_background_read_retries_once(tmp_path):
    class Flaky(Prefetcher):
        fails = 1

        def _read(self, e, buf):
            if Flaky.fails:
                Flaky.fails -= 1
                raise OSError("transient")
            return super()._read(e, buf)

    pf = Flaky(_file(tmp_path), STRIDE)
    assert (pf.take([2])[2] == 2).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_prefetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moestream.prefetch'`

- [ ] **Step 3: Implement**

`moestream/prefetch.py`:
```python
"""Background expert reads into reusable staging buffers.

Worker threads only run os.preadv (it releases the GIL). Buffers are handed out and
returned on the main thread, so the free list needs no lock.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_POOL = ThreadPoolExecutor(8)


class Prefetcher:
    def __init__(self, path, stride):
        self.fd = os.open(path, os.O_RDONLY)
        self.stride = stride
        self.pending = {}  # expert -> (buf, future)
        self.free = []
        self.allocated = 0

    def _buf(self):
        if self.free:
            return self.free.pop()
        self.allocated += 1
        return np.empty(self.stride, np.uint8)

    def _read(self, e, buf):
        n = os.preadv(self.fd, [buf], e * self.stride)
        if n != self.stride:
            raise OSError(f"short read: expert {e}, {n}/{self.stride} bytes")
        return buf

    def submit(self, experts):
        for e in experts:
            if e not in self.pending:
                buf = self._buf()
                self.pending[e] = (buf, _POOL.submit(self._read, e, buf))

    def take(self, experts):
        """{expert: staging buffer}, prefetched or read now. Caller must release() each buffer."""
        self.submit(experts)
        out = {}
        for e in experts:
            buf, fut = self.pending.pop(e)
            try:
                out[e] = fut.result()
            except OSError:
                out[e] = self._read(e, buf)  # one synchronous retry; a second failure raises
        return out

    def release(self, buf):
        self.free.append(buf)

    def drop_pending(self):
        """Forget unused predictions and recycle their buffers."""
        for buf, fut in self.pending.values():
            fut.exception()  # ponytail: waits on stragglers (issued a layer ago, usually done); cancel if it shows in profiles
            self.free.append(buf)
        self.pending.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_prefetch.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add moestream/prefetch.py tests/test_prefetch.py
git commit -m "feat: Prefetcher with reusable staging buffers"
```

---

### Task 3: Budget

**Files:**
- Create: `moestream/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Produces:
  - `GB = 1e9`, `KV_BYTES_PER_TOKEN = 40 * 1024`
  - `parse_vm_stat(text: str) -> int` (reclaimable bytes)
  - `slot_budget_bytes(available: int, working_set: int, resident: int, target_ctx: int = 32768, headroom: float = 3*GB, activations: float = 1*GB) -> int`
  - `probe(resident: int, **kw) -> int`
  - `uniform(layers: list[int], total: int, floor: int = 16, num_experts: int = 256) -> dict[int, int]`
  - `allocate(curves: dict[int, list[float]], grid: list[int], total: int) -> dict[int, int]`. `grid[0]` is the floor and `grid[-1]` the max. `curves[l][i]` is the miss rate at `grid[i]` slots.

Note vs spec: the probe runs **before** the model loads, so resident weights, KV and activations are subtracted from `min(available - headroom, working_set)` instead of only from the GPU cap.

- [ ] **Step 1: Write the failing tests**

`tests/test_budget.py`:
```python
from moestream.budget import GB, KV_BYTES_PER_TOKEN, allocate, parse_vm_stat, slot_budget_bytes, uniform

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   100.
Pages active:                                 999.
Pages inactive:                               20.
Pages speculative:                            3.
Pages throttled:                              0.
Pages wired down:                             777.
Pages purgeable:                              7.
"""


def test_parse_vm_stat_counts_reclaimable_pages():
    assert parse_vm_stat(VM_STAT) == (100 + 20 + 3 + 7) * 16384


def test_budget_is_min_of_free_and_gpu_cap_minus_reservations():
    other = 1 * GB + KV_BYTES_PER_TOKEN * 1000 + 1 * GB
    assert slot_budget_bytes(10 * GB, 20 * GB, 1 * GB, target_ctx=1000) == 10 * GB - 3 * GB - other
    assert slot_budget_bytes(30 * GB, 20 * GB, 1 * GB, target_ctx=1000) == 20 * GB - other
    assert slot_budget_bytes(1 * GB, 20 * GB, 1 * GB) == 0


def test_uniform_clamps_to_floor_and_experts():
    assert uniform([1, 2], 100) == {1: 50, 2: 50}
    assert uniform([1, 2], 4) == {1: 16, 2: 16}
    assert uniform([1, 2], 10**6) == {1: 256, 2: 256}


def test_allocate_favors_steep_curves():
    grid = [16, 32, 48]
    curves = {1: [0.9, 0.2, 0.1], 2: [0.5, 0.45, 0.4]}
    assert allocate(curves, grid, 48) == {1: 32, 2: 16}
    assert allocate(curves, grid, 64) == {1: 48, 2: 16}
    assert allocate(curves, grid, 10**6) == {1: 48, 2: 48}


def test_allocate_below_floor_gives_floor():
    assert allocate({1: [0.5, 0.1], 2: [0.5, 0.1]}, [16, 32], 5) == {1: 16, 2: 16}


def test_allocate_never_exceeds_total_above_floor():
    grid = [16, 32, 64, 128, 256]
    curves = {l: [1 / (1 + i * l) for i in range(5)] for l in range(1, 40)}
    for total in (16 * 39, 5000, 7000):
        assert sum(allocate(curves, grid, total).values()) <= total
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moestream.budget'`

- [ ] **Step 3: Implement**

`moestream/budget.py`:
```python
"""How much memory the expert slots get, and how to split it across layers. Pure functions."""
import heapq
import re
import subprocess

GB = 1e9
KV_BYTES_PER_TOKEN = 40 * 1024  # 10 full-attention layers, bf16 K+V (lagunaLab harness spec)


def parse_vm_stat(text):
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"Pages (\w+):\s+(\d+)\.", text)}
    return page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable"))


def slot_budget_bytes(available, working_set, resident, target_ctx=32768, headroom=3 * GB, activations=1 * GB):
    other = resident + KV_BYTES_PER_TOKEN * target_ctx + activations
    return int(max(0, min(available - headroom, working_set) - other))


def probe(resident, **kw):
    import mlx.core as mx

    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    return slot_budget_bytes(parse_vm_stat(vm), mx.device_info()["max_recommended_working_set_size"], resident, **kw)


def uniform(layers, total, floor=16, num_experts=256):
    s = min(num_experts, max(floor, total // len(layers)))
    return {l: s for l in layers}


def allocate(curves, grid, total):
    """Greedy: every layer starts at grid[0]; each grid step goes to the layer whose miss rate drops most per slot."""
    at = {l: 0 for l in curves}
    left = total - grid[0] * len(curves)

    def push(heap, l):
        i = at[l]
        if i + 1 < len(grid):
            cost = grid[i + 1] - grid[i]
            heapq.heappush(heap, (-(curves[l][i] - curves[l][i + 1]) / cost, l, cost))

    heap = []
    for l in curves:
        push(heap, l)
    while heap and left > 0:
        _, l, cost = heapq.heappop(heap)
        if cost > left:
            continue  # ponytail: skips this layer for good; fine at 16-slot grid granularity
        at[l] += 1
        left -= cost
        push(heap, l)
    return {l: grid[i] for l, i in at.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_budget.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add moestream/budget.py tests/test_budget.py
git commit -m "feat: memory budget probe and per-layer slot allocation"
```

---

### Task 4: StreamingMoE and load

**Files:**
- Create: `moestream/layer.py`, `moestream/load.py`
- Test: `tests/test_layer.py`

**Interfaces:**
- Consumes: `SlotStore` (Task 1), `Prefetcher` (Task 2), `probe`, `uniform`, `allocate`, `GB` (Task 3). From lagunaLab: `laguna.Model`, `laguna.ModelArgs`, `laguna.MLP`, `laguna.SparseMoeBlock.route`, `scripts.pack_experts.expert_from_bytes`, `scripts.pack_experts.pack`, `tests.tiny.quantized_model_dir`, `tests.tiny.TINY`.
- Produces:
  - `StreamingMoE(args, layer, pack_dir, layout, slots, half_life, group_size, bits, dtype)`. It has `.io` (a SimpleNamespace, kept out of the parameter tree) with fields `glu`, `store: SlotStore`, `prefetcher: Prefetcher`, `layout`, `next: StreamingMoE | None`, `predictor: Callable[[mx.array], list[int]] | None`, `trace: list[np.ndarray] | None`, `guess: set | None`, `stats: dict`.
  - `stream_stats(model) -> dict` with keys `hits, misses, hit_rate, pred_acc, sync_s, read_s, write_s, bytes_read, slots`.
  - `load(pack_dir, *, slots=None, budget_gb=None, calibration=None, half_life=None, target_ctx=32768, floor=16) -> (model, tokenizer | None)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_layer.py`:
```python
import json

import mlx.core as mx
import numpy as np
import pytest

from moestream.layer import StreamingMoE, stream_stats
from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir


@pytest.fixture
def packed(tmp_path):
    d, resident = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    return out, resident


def _ids(n, seed=6):
    return mx.array(np.random.default_rng(seed).integers(0, TINY["vocab_size"], size=(1, n)))


def _close(a, b):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5, rtol=1e-5)


def _moe(model):
    return [l.mlp for l in model.layers if isinstance(l.mlp, StreamingMoE)]


def _decode(model, ids, split):
    cache = model.make_cache()
    got = [model(ids[:, :split], cache=cache)]
    got += [model(ids[:, i : i + 1], cache=cache) for i in range(split, ids.shape[1])]
    return mx.concatenate(got, axis=1)


def test_prefill_with_fewer_slots_than_experts_is_exact(packed):
    out, resident = packed
    m, _ = load(out, slots=3)
    ids = _ids(20)
    _close(m(ids), resident(ids))
    assert stream_stats(m)["misses"] > 0


@pytest.mark.parametrize("predictor", [None, "all", "fixed"])
def test_decode_is_exact_for_any_prediction(packed, predictor):
    out, resident = packed
    m, _ = load(out, slots=4)
    for moe in _moe(m):
        if predictor == "all":
            moe.io.predictor = lambda x: list(range(TINY["num_experts"]))
        elif predictor == "fixed":
            moe.io.predictor = lambda x: [0, 1]
    ids = _ids(16, seed=7)
    _close(_decode(m, ids, 10), resident(ids))
    s = stream_stats(m)
    assert s["hits"] > 0
    if predictor is None:
        assert 0 <= s["pred_acc"] <= 1 and s["sync_s"] > 0


def test_bfloat16_is_bit_exact(tmp_path):
    d, resident = quantized_model_dir(tmp_path, dtype=mx.bfloat16)
    out = tmp_path / "packed"
    pack(d, out)
    m, _ = load(out, slots=3)
    ids = _ids(16, seed=3)
    got, want = _decode(m, ids, 8), resident(ids)
    assert got.dtype == mx.bfloat16 and mx.array_equal(got, want).item()


def test_decode_then_prefill_chunk_is_exact(packed):
    out, resident = packed
    m, _ = load(out, slots=3)  # a 12-token chunk needs > 3 experts: grouped passes
    ids = _ids(30, seed=9)
    cache = m.make_cache()
    got = [m(ids[:, :8], cache=cache)]
    got += [m(ids[:, i : i + 1], cache=cache) for i in range(8, 14)]  # decode leaves predictions pending
    got += [m(ids[:, 14:26], cache=cache)]                           # tool result appended
    got += [m(ids[:, i : i + 1], cache=cache) for i in range(26, 30)]
    _close(mx.concatenate(got, axis=1), resident(ids))


def test_multi_token_cached_steps_are_exact(packed):
    out, resident = packed
    m, _ = load(out, slots=4)
    ids = _ids(24, seed=10)
    cache = m.make_cache()
    got = [m(ids[:, a : a + 4], cache=cache) for a in range(0, 24, 4)]
    _close(mx.concatenate(got, axis=1), resident(ids))


def test_wrong_predictions_do_not_grow_buffers(packed):
    out, _ = packed
    m, _ = load(out, slots=3)
    for moe in _moe(m):
        moe.io.predictor = lambda x: [5, 6, 7]
    ids = _ids(60, seed=11)
    cache = m.make_cache()
    m(ids[:, :4], cache=cache)
    for i in range(4, 20):
        m(ids[:, i : i + 1], cache=cache)
    before = [moe.io.prefetcher.allocated for moe in _moe(m)]
    for i in range(20, 60):
        m(ids[:, i : i + 1], cache=cache)
    assert [moe.io.prefetcher.allocated for moe in _moe(m)] == before


def test_trace_records_per_token_routing(packed):
    out, _ = packed
    m, _ = load(out, slots=8)
    for moe in _moe(m):
        moe.io.trace = []
    m(_ids(10))
    for moe in _moe(m):
        (t,) = moe.io.trace
        assert t.shape == (10, TINY["num_experts_per_tok"]) and t.dtype == np.uint8


def test_tiny_budget_still_loads_and_warns(packed):
    out, resident = packed
    with pytest.warns(UserWarning, match="floor"):
        m, _ = load(out, budget_gb=1e-9)
    ids = _ids(12)
    _close(m(ids), resident(ids))


def test_calibration_layer_mismatch_falls_back_to_uniform(packed, tmp_path):
    out, _ = packed
    cal = tmp_path / "budget.json"
    cal.write_text(json.dumps({"half_life": 128.0, "grid": [2, 4], "curves": {"7": [0.5, 0.1]}}))
    with pytest.warns(UserWarning, match="calibration"):
        m, _ = load(out, budget_gb=1.0, calibration=cal)
    assert {moe.io.store.slots for moe in _moe(m)} == {TINY["num_experts"]}


def test_missing_expert_file_fails_at_load(packed):
    out, _ = packed
    (out / "experts" / "layer_02.bin").unlink()
    with pytest.raises(ValueError, match="layer_02.bin"):
        load(out, slots=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_layer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'moestream.layer'`

- [ ] **Step 3: Implement `moestream/layer.py`**

```python
"""StreamingMoE: Laguna's MoE block with routed experts streamed from SSD into per-layer slots.

One GPU sync per layer returns this layer's routing and, during decode, a guess at the
next layer's (the next router run on this layer's input). The guess is read from disk in
the background while the GPU works; slots are only filled when a layer uses an expert.
"""
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

from laguna_mlx import laguna
from moestream.prefetch import Prefetcher
from moestream.store import SlotStore
from scripts.pack_experts import expert_from_bytes


class StreamingMoE(nn.Module):
    route = laguna.SparseMoeBlock.route

    def __init__(self, args, layer, pack_dir, layout, slots, half_life, group_size, bits, dtype):
        super().__init__()
        h = args.hidden_size
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = nn.Linear(h, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))
        self.shared_expert = laguna.MLP(h, args.shared_expert_intermediate_size)
        glu = SwitchGLU(h, args.moe_intermediate_size, slots)
        glu.set_dtype(dtype)  # match packed scales dtype: no cast on write
        nn.quantize(glu, group_size=group_size, bits=bits)
        # SimpleNamespace keeps slots and links out of the parameter tree (load_weights strict=True)
        self.io = SimpleNamespace(
            glu=glu, layout=layout, next=None, predictor=None, trace=None, guess=None,
            store=SlotStore(args.num_experts, slots, half_life),
            prefetcher=Prefetcher(Path(pack_dir) / "experts" / f"layer_{layer:02d}.bin", layout["stride"]),
            stats=dict(pred_hit=0, pred_total=0, sync_s=0.0, read_s=0.0, write_s=0.0),
        )

    def __call__(self, x):
        io = self.io
        inds, w = self.route(x)
        decode = x.shape[0] * x.shape[1] == 1
        lookahead = decode and io.next is not None and io.predictor is None
        t0 = time.perf_counter()
        if lookahead:
            pinds = io.next.route(x)[0]
            mx.eval(inds, pinds)
        else:
            mx.eval(inds)
        io.stats["sync_s"] += time.perf_counter() - t0
        idx = np.array(inds)
        if io.trace is not None:
            io.trace.append(idx.reshape(-1, idx.shape[-1]).astype(np.uint8))
        experts = np.unique(idx).tolist()
        if io.guess is not None:
            io.stats["pred_hit"] += len(io.guess & set(experts))
            io.stats["pred_total"] += len(experts)
            io.guess = None
        io.store.tick(idx.size // idx.shape[-1])
        y = None
        for g in range(0, len(experts), io.store.slots):
            yg = self._run(x, idx, experts[g : g + io.store.slots], masked=len(experts) > io.store.slots)
            y = yg if y is None else y + yg
        io.prefetcher.drop_pending()
        if decode and io.next is not None:
            guess = np.unique(np.array(pinds)).tolist() if lookahead else list(io.predictor(x))
            io.next.io.guess = set(guess)
            io.next.io.prefetcher.submit([e for e in guess if e not in io.next.io.store.expert_slot])
        y = (y * w[..., None]).sum(axis=-2)
        return y * self.routed_scaling_factor + self.shared_expert(x)

    def _run(self, x, idx, group, masked):
        io = self.io
        slot_of, fills = io.store.assign(group)
        t0 = time.perf_counter()
        bufs = io.prefetcher.take([e for e, _ in fills])
        t1 = time.perf_counter()
        for e, s in fills:
            for name, arr in expert_from_bytes(bufs[e], io.layout).items():  # mx.array copies here
                proj, part = name.split(".")
                getattr(getattr(io.glu, proj), part)[s] = arr
            io.prefetcher.release(bufs[e])
        io.stats["read_s"] += t1 - t0
        io.stats["write_s"] += time.perf_counter() - t1
        E = io.layout["num_experts"]
        table, member = np.zeros(E, np.int32), np.zeros(E, bool)
        for e, s in slot_of.items():
            table[e], member[e] = s, True
        y = io.glu(x, mx.array(table[idx]))
        if masked:
            y = mx.where(mx.array(member[idx])[..., None], y, 0)
        return y


def stream_stats(model):
    moe = [l.mlp for l in model.layers if isinstance(l.mlp, StreamingMoE)]
    hits = sum(m.io.store.hits for m in moe)
    misses = sum(m.io.store.misses for m in moe)
    tot = {k: sum(m.io.stats[k] for m in moe) for k in moe[0].io.stats}
    return {
        "hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses),
        "pred_acc": tot["pred_hit"] / max(1, tot["pred_total"]),
        "sync_s": tot["sync_s"], "read_s": tot["read_s"], "write_s": tot["write_s"],
        "bytes_read": misses * moe[0].io.layout["stride"],
        "slots": {i: l.mlp.io.store.slots for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)},
    }
```

- [ ] **Step 4: Implement `moestream/load.py`**

```python
"""load(pack_dir) -> (model, tokenizer): Laguna with streamed experts sized to free memory."""
import json
import sys
import warnings
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from laguna_mlx import laguna
from moestream.budget import GB, allocate, probe, uniform
from moestream.layer import StreamingMoE


def _per_layer_slots(layers, E, stride, budget, cal, floor):
    total = int(budget // stride)
    if total < floor * len(layers):
        warnings.warn(f"slot budget {budget / GB:.2f} GB is below the floor ({floor} slots x {len(layers)} layers); "
                      "using the floor anyway")
    if cal is not None:
        grid = [min(E, g) for g in cal["grid"]]
        return allocate({int(l): c for l, c in cal["curves"].items()}, grid, total)
    return uniform(layers, total, min(floor, E), E)


def load(pack_dir, *, slots=None, budget_gb=None, calibration=None, half_life=None, target_ctx=32768, floor=16):
    pack_dir = Path(pack_dir)
    config = json.loads((pack_dir / "config.json").read_text())
    layout = json.loads((pack_dir / "layout.json").read_text())
    q = config["quantization"]
    E, stride, layers = layout["num_experts"], layout["stride"], layout["layers"]
    for l in layers:
        f = pack_dir / "experts" / f"layer_{l:02d}.bin"
        if not f.exists() or f.stat().st_size != E * stride:
            raise ValueError(f"{f}: missing or not {E * stride} bytes")

    cal = None
    if calibration is not None and Path(calibration).exists():
        cal = json.loads(Path(calibration).read_text())
        if sorted(int(l) for l in cal["curves"]) != layers:
            warnings.warn(f"calibration {calibration} does not match layers {layers}; using a uniform split")
            cal = None
    half_life = half_life or (cal["half_life"] if cal else 256.0)

    if slots is not None:
        per = {l: slots for l in layers}
    else:
        budget = budget_gb * GB if budget_gb is not None else probe(
            (pack_dir / "resident.safetensors").stat().st_size, target_ctx=target_ctx)
        per = _per_layer_slots(layers, E, stride, budget, cal, floor)

    args = laguna.ModelArgs.from_dict(config)
    model = laguna.Model(args)  # lazy: the full SwitchGLUs it builds are never evaluated
    dtype = getattr(mx, layout["arrays"]["gate_proj.scales"]["dtype"])
    moe = []
    for l in layers:
        model.layers[l].mlp = StreamingMoE(args, l, pack_dir, layout, per[l], half_life,
                                           q["group_size"], q["bits"], dtype)
        moe.append(model.layers[l].mlp)
    for a, b in zip(moe, moe[1:]):
        a.io.next = b

    def predicate(p, m):
        return q[p] if p in q else hasattr(m, "to_quantized")

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], class_predicate=predicate)
    model.load_weights(str(pack_dir / "resident.safetensors"), strict=True)
    mx.eval(model.parameters())
    n = sum(per.values())
    print(f"moeStream: {n} slots ({n * stride / GB:.2f} GB), per layer {min(per.values())}-{max(per.values())}, "
          f"half-life {half_life:g}", file=sys.stderr)
    tok = None
    if (pack_dir / "tokenizer.json").exists():
        from mlx_lm.utils import load_tokenizer
        tok = load_tokenizer(pack_dir)
    return model, tok
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_layer.py -v`
Expected: 12 passed (the predictor parametrization counts 3)

- [ ] **Step 6: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add moestream/layer.py moestream/load.py tests/test_layer.py
git commit -m "feat: StreamingMoE with next-layer lookahead prefetch, and budgeted load"
```

---

### Task 5: Record routing over captured sessions

**Files:**
- Create: `tools/record.py`
- Test: `tests/test_record.py`

**Interfaces:**
- Consumes: `load`, `StreamingMoE` (Task 4); `replay.session.load_session`, `replay.session.tokenize` (lagunaLab).
- Produces: `record_ids(model, ids: list[int] | np.ndarray, chunk: int = 1024) -> dict[int, np.ndarray]` (layer → uint8 `[T, top_k]`). The CLI writes `data/routing/<session>.npz`, with one array per layer keyed `"layer_<l>"` plus `"assistant_mask"` (bool `[T]`).

- [ ] **Step 1: Write the failing test**

`tests/test_record.py`:
```python
import numpy as np

from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir
from tools.record import record_ids


def test_record_ids_gives_per_token_routing_per_layer(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    ids = np.random.default_rng(0).integers(0, TINY["vocab_size"], size=25)
    rec = record_ids(m, ids, chunk=10)
    assert sorted(rec) == [1, 2, 3]
    for r in rec.values():
        assert r.shape == (25, TINY["num_experts_per_tok"]) and r.dtype == np.uint8
        assert (r < TINY["num_experts"]).all()
    again = record_ids(m, ids, chunk=10)
    assert all((again[l] == rec[l]).all() for l in rec)  # deterministic, trace hooks cleared
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_record.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.record'`

- [ ] **Step 3: Implement**

`tools/record.py`:
```python
"""Record per-layer expert routing over captured agent sessions, for offline calibration.

PYTHONPATH=.:../lagunaLab python3 -m tools.record [--max-tokens 4096] [--budget-gb 5]
Loads real weights (~budget + 1.5 GB): message the other sessions and wait for a go-ahead first.
Teacher-forced prefill gives the same routing as decode, token by token.
"""
import argparse
import glob
from pathlib import Path

import mlx.core as mx
import numpy as np

from moestream.layer import StreamingMoE


def record_ids(model, ids, chunk=1024):
    moe = {i: l.mlp for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)}
    for m in moe.values():
        m.io.trace = []
    try:
        cache = model.make_cache()
        ids = np.asarray(ids)
        for a in range(0, len(ids), chunk):
            mx.eval(model(mx.array(ids[None, a : a + chunk]), cache=cache))
        return {i: np.concatenate(m.io.trace) for i, m in moe.items()}
    finally:
        for m in moe.values():
            m.io.trace = None


def main():
    from transformers import AutoTokenizer

    from moestream.load import load
    from replay.session import load_session, tokenize

    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="../lagunaLab/models/Laguna-XS-2.1-packed")
    p.add_argument("--traces", default="../lagunaLab/traces")
    p.add_argument("--out", default="data/routing")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--budget-gb", type=float, default=5.0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load(a.pack, budget_gb=a.budget_gb)
    tok = AutoTokenizer.from_pretrained(a.pack)
    for f in sorted(glob.glob(f"{a.traces}/*.jsonl")):
        name = Path(f).stem
        if (out / f"{name}.npz").exists():
            continue  # resumable
        msgs, tools = load_session(f)
        ids, mask, _ = tokenize(tok, msgs, tools, max_tokens=a.max_tokens)
        rec = record_ids(model, ids)
        np.savez(out / f"{name}.npz", assistant_mask=mask, **{f"layer_{l}": r for l, r in rec.items()})
        print(name, len(ids), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_record.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tools/record.py tests/test_record.py
git commit -m "feat: routing recorder over captured sessions"
```

- [ ] **Step 6: Coordinate, then record on real weights**

Message `current-41` and `current-0f` (SendMessage): "moeStream: about to load the Laguna streaming model (~6.5 GB) for about 15-30 min to record routing. OK to start?" Wait for both to agree and confirm `overnight.sh` isn't running.

Run: `PYTHONPATH=.:../lagunaLab python3 -m tools.record 2>&1 | tee results/record.log`
Expected: one `<session> <n_tokens>` line per `lagunaLab/traces/*.jsonl` (17 sessions), and `data/routing/*.npz` created. Then message both sessions: "moeStream: record finished, memory released."

---

### Task 6: Calibrate

**Files:**
- Create: `tools/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `SlotStore` (Task 1); `data/routing/*.npz` (Task 5).
- Produces:
  - `miss_rate(seq: np.ndarray, slots: int, half_life: float, num_experts: int = 256) -> float`
  - `calibrate(seqs: dict[int, list[np.ndarray]], grid: list[int], half_lives: list[float], ref_slots: int = 64, num_experts: int = 256) -> dict`
  - `budget.json` with keys `half_life`, `grid`, `curves` (`{str(layer): [miss rate per grid point]}`), `half_life_search`, and `tokens`. `load(..., calibration=...)` reads it (Task 4).

- [ ] **Step 1: Write the failing tests**

`tests/test_calibrate.py`:
```python
import numpy as np

from tools.calibrate import calibrate, miss_rate


def test_miss_rate_falls_with_more_slots_and_hits_zero_when_all_fit():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 16, size=(400, 2)).astype(np.uint8)
    seq[:, 1] = (seq[:, 0] + 1) % 16  # distinct per row
    r = [miss_rate(seq, s, 256.0, 16) for s in (2, 4, 8, 16)]
    assert r == sorted(r, reverse=True)
    assert r[-1] == 16 / 800  # only cold misses


def test_calibrate_picks_short_half_life_for_phase_shift():
    # phase 1 hammers {0,1}; phase 2 alternates {2,3},{4,5} and needs all 4 slots.
    # Plain LFU keeps 0,1 pinned (miss every step); decay frees them within a few steps.
    phase2 = [[2, 3], [4, 5]] * 150
    seq = np.array([[0, 1]] * 300 + phase2, np.uint8)
    seqs = {1: [seq], 2: [seq]}
    out = calibrate(seqs, grid=[2, 4, 8], half_lives=[4.0, float("inf")], ref_slots=4, num_experts=8)
    assert set(out) >= {"half_life", "grid", "curves", "half_life_search", "tokens"}
    assert out["grid"] == [2, 4, 8] and sorted(out["curves"]) == ["1", "2"]
    assert all(len(c) == 3 for c in out["curves"].values())
    assert out["half_life_search"]["4.0"] < out["half_life_search"]["inf"]
    assert out["half_life"] == 4.0 and out["tokens"] == 600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.calibrate'`

- [ ] **Step 3: Implement**

`tools/calibrate.py`:
```python
"""Offline: replay recorded routing through SlotStore, producing per-layer miss curves and a decay half-life.

PYTHONPATH=.:../lagunaLab python3 -m tools.calibrate [--routing data/routing] [--out budget.json]
No model load. Layers run in parallel processes.
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from moestream.store import SlotStore

GRID = [16, 32, 48, 64, 96, 128, 192, 256]
HALF_LIVES = [64.0, 128.0, 256.0, 512.0, 1024.0, float("inf")]


def miss_rate(seq, slots, half_life, num_experts=256):
    s = SlotStore(num_experts, slots, half_life)
    for row in seq:
        s.tick()
        s.assign(row.tolist())
    return s.misses / max(1, s.misses + s.hits)


def _layer_rate(seqs, slots, half_life, num_experts):
    # sessions are independent agent runs: fresh cache each, rate over all tokens
    rates = [miss_rate(q, slots, half_life, num_experts) for q in seqs]
    n = [len(q) for q in seqs]
    return float(np.average(rates, weights=n))


def calibrate(seqs, grid=GRID, half_lives=HALF_LIVES, ref_slots=64, num_experts=256):
    layers = sorted(seqs)
    with ProcessPoolExecutor() as ex:
        search = {}
        for h in half_lives:
            f = partial(_layer_rate, slots=ref_slots, half_life=h, num_experts=num_experts)
            search[str(h)] = float(np.mean(list(ex.map(f, [seqs[l] for l in layers]))))
        best = min(half_lives, key=lambda h: search[str(h)])
        curves = {}
        for l in layers:
            f = partial(_layer_rate, seqs[l], half_life=best, num_experts=num_experts)
            curves[str(l)] = list(ex.map(f, grid))
    return {"half_life": best, "grid": list(grid), "curves": curves, "half_life_search": search,
            "tokens": int(sum(len(q) for q in seqs[layers[0]]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--routing", default="data/routing")
    p.add_argument("--out", default="budget.json")
    p.add_argument("--max-tokens", type=int, default=2048, help="per session; bounds runtime")
    a = p.parse_args()
    seqs = {}
    for f in sorted(glob.glob(f"{a.routing}/*.npz")):
        z = np.load(f)
        for k in z.files:
            if k.startswith("layer_"):
                seqs.setdefault(int(k[6:]), []).append(z[k][: a.max_tokens])
    out = calibrate(seqs)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("half_life", "half_life_search", "tokens")}, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_calibrate.py -v`
Expected: 2 passed

- [ ] **Step 5: Run calibration (offline, no model)**

Run: `PYTHONPATH=.:../lagunaLab python3 -m tools.calibrate | tee results/calibrate.log`
Expected: prints the chosen `half_life` and its search table, and writes `budget.json`. It should take roughly 5-20 min. If it's much slower, rerun with `--max-tokens 1024` and note that in the log.

- [ ] **Step 6: Commit**

```bash
git add tools/calibrate.py tests/test_calibrate.py budget.json results/calibrate.log
git commit -m "feat: offline calibration of per-layer miss curves and decay half-life"
```

---

### Task 7: Benchmark and verdict

**Files:**
- Create: `tools/bench.py`, `README.md`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `load`, `stream_stats` (Task 4); `budget.json` (Task 6). From lagunaLab: `laguna_mlx.streaming.load_streaming`, `stream_stats` (as `lab_stats`), `laguna_mlx.dflash.generate_greedy`, `scripts.check_parity.PROMPTS`.
- Produces: `decode_tok_s(model, prompts: list[list[int]], max_tokens: int, eos: tuple) -> (float, list[list[int]])`, `long_ctx(model, ids: list[int], gen: int = 64, chunk: int = 2048) -> dict`, and `results/bench.json`.

- [ ] **Step 1: Write the failing smoke test**

`tests/test_bench.py`:
```python
import numpy as np

from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir
from tools.bench import decode_tok_s, long_ctx


def test_bench_helpers_run_on_tiny(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, TINY["vocab_size"], size=6).tolist() for _ in range(2)]
    tps, outs = decode_tok_s(m, prompts, max_tokens=5, eos=())
    assert tps > 0 and [len(o) for o in outs] == [5, 5]
    r = long_ctx(m, rng.integers(0, TINY["vocab_size"], size=40).tolist(), gen=4, chunk=16)
    assert r["ctx_tokens"] == 40 and r["decode_ctx_tok_s"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_bench.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.bench'`

- [ ] **Step 3: Implement**

`tools/bench.py`:
```python
"""Decode speed, hit rate, prediction accuracy and time split: lagunaLab streaming vs moeStream.

PYTHONPATH=.:../lagunaLab python3 -m tools.bench [--cold] [--budget-gb X]
Loads real weights one model at a time: message the other sessions and wait for a go-ahead first.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx

from laguna_mlx.dflash import generate_greedy


def decode_tok_s(model, prompts, max_tokens, eos):
    outs, t0 = [], time.perf_counter()
    for pr in prompts:
        outs.append(generate_greedy(model, pr, max_tokens, eos))
    return sum(map(len, outs)) / (time.perf_counter() - t0), outs


def long_ctx(model, ids, gen=64, chunk=2048):
    cache = model.make_cache()
    t0 = time.perf_counter()
    for a in range(0, len(ids), chunk):
        logits = model(mx.array([ids[a : a + chunk]]), cache=cache)[0, -1]
        mx.eval(logits)
    prefill = time.perf_counter() - t0
    t0 = time.perf_counter()
    t = int(mx.argmax(logits))
    for _ in range(gen):
        logits = model(mx.array([[t]]), cache=cache)[0, -1]
        t = int(mx.argmax(logits))
    return {"ctx_tokens": len(ids), "prefill_ctx_tok_s": len(ids) / prefill,
            "decode_ctx_tok_s": gen / (time.perf_counter() - t0)}


def evict_page_cache(paths):
    """Cold-cache approximation without sudo: stream unrelated big files through the page cache."""
    for f in paths:
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass


def main():
    from laguna_mlx.streaming import load_streaming
    from laguna_mlx.streaming import stream_stats as lab_stats
    from moestream.layer import stream_stats
    from moestream.load import load
    from scripts.check_parity import PROMPTS

    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="../lagunaLab/models/Laguna-XS-2.1-packed")
    p.add_argument("--budget-gb", type=float, default=None)
    p.add_argument("--calibration", default="budget.json")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ctx", type=int, default=16000)
    p.add_argument("--cold", action="store_true")
    a = p.parse_args()
    ctx_text = Path("../lagunaLab/tests/reference/modeling_laguna.py").read_text()
    cold_files = sorted(glob.glob("../lagunaLab/models/Laguna-XS-2.1-mlx-4bit/*.safetensors"))

    def run(label, loader, stats):
        if a.cold:
            evict_page_cache(cold_files)
        mx.reset_peak_memory()
        model, tok = loader()
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                           enable_thinking=False) for q in PROMPTS[:3]]
        s0 = stats(model)
        tps, outs = decode_tok_s(model, prompts, a.max_tokens, tuple(tok.eos_token_ids))
        s1 = stats(model)
        n = sum(map(len, outs))
        row = {"label": label, "decode_tok_s": tps, "peak_gpu_gb": mx.get_peak_memory() / 1e9,
               "hit_rate_decode": (s1["hits"] - s0["hits"]) / max(1, s1["hits"] + s1["misses"] - s0["hits"] - s0["misses"])}
        for k in ("sync_s", "read_s", "write_s"):
            if k in s1:
                row[f"{k[:-2]}_ms_per_tok"] = 1000 * (s1[k] - s0[k]) / n
        if "pred_acc" in s1:
            row["pred_acc"] = s1["pred_acc"]
            row["slots_per_layer"] = [min(s1["slots"].values()), max(s1["slots"].values())]
        ids = tok.encode(ctx_text)
        ids = (ids * (a.ctx // len(ids) + 1))[: a.ctx]
        row.update(long_ctx(model, ids))
        row["peak_gpu_gb"] = mx.get_peak_memory() / 1e9
        print(row, flush=True)
        del model
        mx.clear_cache()
        return row, outs

    base, base_outs = run("lagunaLab slots=32", lambda: load_streaming(a.pack, 32), lab_stats)
    mine, my_outs = run("moeStream", lambda: load(a.pack, budget_gb=a.budget_gb, calibration=a.calibration),
                        stream_stats)
    mine["identical_to_lagunaLab"] = my_outs == base_outs
    mine["speedup_vs_lagunaLab"] = mine["decode_tok_s"] / base["decode_tok_s"]
    mine["ctx_decode_ratio"] = mine["decode_ctx_tok_s"] / mine["decode_tok_s"]
    out = Path("results") / ("bench_cold.json" if a.cold else "bench.json")
    out.write_text(json.dumps({"rows": [base, mine], "cold": a.cold}, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_bench.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tools/bench.py tests/test_bench.py
git commit -m "feat: benchmark vs lagunaLab streaming"
```

- [ ] **Step 6: Coordinate, then benchmark warm and cold**

Message `current-41` and `current-0f`: "moeStream: about to benchmark two Laguna streaming models one after the other (peak up to ~15 GB) for about 30-45 min, plus a cold-cache run that reads 18 GB from lagunaLab/models (read-only). OK to start?" Wait for both to agree and confirm `overnight.sh` isn't running.

Run: `PYTHONPATH=.:../lagunaLab python3 -m tools.bench 2>&1 | tee results/bench.log`
Then: `PYTHONPATH=.:../lagunaLab python3 -m tools.bench --cold 2>&1 | tee results/bench_cold.log`
Expected: `results/bench.json` and `results/bench_cold.json` each with two rows. Then message both sessions that the runs are finished.

- [ ] **Step 7: Check success criteria and write README**

Read `results/bench.json` and check the spec's success criteria:
1. `identical_to_lagunaLab` is true. (`lagunaLab` streaming is the reference proven equal to resident. If it's false, diff the outputs and check for bf16 logit near-ties before calling it a failure, as `lagunaLab`'s `streaming.json` notes describe.)
2. `speedup_vs_lagunaLab` ≥ 2.0.
3. `ctx_decode_ratio` ≥ 0.75.
4. `peak_gpu_gb` ≤ the budget logged at load + resident + KV + 1 GB.

Write `README.md`: a one-paragraph purpose, how to run the tests/record/calibrate/bench, and a results table **generated from `results/*.json`** (every number cited from a file) with pass/fail per criterion. If criterion 2 or 3 fails, add a "Next" line: whichever of `sync_ms_per_tok`, `read_ms_per_tok` or `write_ms_per_tok` dominates says whether to build Phase 2 (zero-copy Metal) or improve prediction (Laya).

```bash
git add README.md results/
git commit -m "docs: phase 1 results and README"
```
