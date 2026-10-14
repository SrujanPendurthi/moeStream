# Engine config, eviction, KV quantization, server and pi: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Laguna-XS-2.1 on moeStream, configured from one `moestream.toml` (memory cap, context, KV bits, eviction policy, thinking), served OpenAI-compatible and usable from pi.

**Architecture:** A flat `EngineConfig` read from TOML feeds `moestream.load.load`. `budget.py` turns the memory cap into slot bytes after resident weights, KV (sized by `kv_bits`) and activations. `SlotStore` picks its eviction key by policy. KV quantization reuses lagunaLab's `make_quantized_cache`. `tools/serve.py` is mlx-lm's server with the loader swapped, the same shape as lagunaLab's `serve_streaming.py`.

**Tech Stack:** Python 3.13, MLX 0.31 / mlx-lm 0.31.3, stdlib `tomllib`, pytest.

**Spec:** `docs/specs/2026-10-13-engine-config-serve-design.md`

## Global Constraints

- No new dependencies. TOML is read with stdlib `tomllib`; the config is not written from code in this sub-project.
- Never write into `../lagunaLab`. Imports from it are limited to `.gpu/RULES.md` rule 3, plus `laguna_mlx.cache.make_quantized_cache` and `scripts.serve_streaming` (tell the coordinator session both are now in use).
- Tests: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`. Tiny models only, no real weights.
- Any real-weight run: `PYTHONPATH=.:../lagunaLab /Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe tools/watch.sh 8 21 <cmd>`. lagunaLab has priority.
- `moestream.toml` ships `memory_gb = 20.0`. Omitting the key means "probe free memory".
- Keep `load()`'s keyword form working: `tools/bench.py`, `tools/record.py` and the tests call it.
- Known-good operating point to quote in the memory warning: 12.8 GB peak, 13.9 tok/s at 128 slots.
- Spec deviation (Task 4): KV quantization uses lagunaLab's `make_quantized_cache` (its `DequantKVCache` stores 8/4-bit, attends in bf16) instead of raw mlx-lm `QuantizedKVCache`. lagunaLab found mlx-lm's quantized attention builds the full score matrix (~16 GB per layer at 82k) and hung the machine.

## Review Focus

- Running `tools.serve` from a directory other than the repo root: relative `pack_dir` and `calibration` in the TOML must resolve against the TOML file's directory, not the cwd. Test in Task 1.
- Type mistakes in the TOML (`memory_gb = "20"`, `context_tokens = 0`, `kv_bits = 8.0`, `thinking = "yes"`): reject at read with the key in the message, never fail later inside MLX. Test in Task 1.
- `kv_bits = 8` on a model whose `head_dim` is not a multiple of the quantization group: fail at load with "head_dim" in the message, not mid-generation. Test in Task 4 (tiny model has head_dim 16).
- `eviction = "lru"` or `"fifo"` with a calibration file tuned for LFU: still loads, warns that the split was tuned for LFU. Test in Task 2.
- `memory_gb` above what is free on the machine: loads, warns with the free amount and the known-good operating point. Test in Task 3.

---

## File structure

| file | change | responsibility |
|---|---|---|
| `moestream/config.py` | create | `EngineConfig`: read and validate `moestream.toml` |
| `moestream.toml` | create | the one settings file, defaults from the spec |
| `moestream/store.py` | modify | `policy` argument picks the eviction key |
| `moestream/layer.py` | modify | pass `policy` through to `SlotStore` |
| `moestream/budget.py` | modify | `kv_bytes_per_token`, `cap_budget_bytes`, `free_bytes` |
| `moestream/load.py` | modify | `memory_gb`, `kv_bits`, `eviction`, `load(EngineConfig)` |
| `tools/serve.py` | create | mlx-lm server with moeStream loader and config-derived flags |
| `tests/test_config.py`, `tests/test_serve.py` | create | |
| `tests/test_store.py`, `tests/test_budget.py`, `tests/test_layer.py` | modify | |
| `README.md` | modify | serve + pi instructions, verification results |

---

### Task 1: EngineConfig and moestream.toml

**Files:**
- Create: `moestream/config.py`, `moestream.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `EngineConfig` dataclass with fields `pack_dir: str`, `memory_gb: float | None`, `context_tokens: int`, `kv_bits: int`, `eviction: str`, `half_life: float | None`, `calibration: str | None`, `thinking: bool`, `port: int`; `EngineConfig.read(path) -> EngineConfig` (raises `ValueError`).

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
from pathlib import Path

import pytest

from moestream.config import EngineConfig

REPO = Path(__file__).resolve().parents[1]


def _write(tmp_path, text, name="engine.toml"):
    f = tmp_path / name
    f.write_text(text)
    return f


def test_shipped_file_reads_with_spec_defaults():
    c = EngineConfig.read(REPO / "moestream.toml")
    assert c.memory_gb == 20.0 and c.context_tokens == 32768 and c.kv_bits == 0
    assert c.eviction == "lfu" and c.thinking is True and c.port == 8081
    assert Path(c.pack_dir) == (REPO / "../lagunaLab/models/Laguna-XS-2.1-packed")
    assert Path(c.calibration) == REPO / "budget.json"


def test_omitted_keys_take_defaults(tmp_path):
    c = EngineConfig.read(_write(tmp_path, 'pack_dir = "p"\n'))
    assert c.memory_gb is None and c.half_life is None and c.calibration is None
    assert c.thinking is True and c.eviction == "lfu" and c.kv_bits == 0


def test_relative_paths_resolve_against_the_file_not_the_cwd(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    f = _write(sub, 'pack_dir = "packed"\ncalibration = "cal.json"\n')
    monkeypatch.chdir(tmp_path)
    c = EngineConfig.read(f)
    assert c.pack_dir == str(sub / "packed") and c.calibration == str(sub / "cal.json")


def test_absolute_paths_are_kept(tmp_path):
    c = EngineConfig.read(_write(tmp_path, f'pack_dir = "{tmp_path}/x"\n'))
    assert c.pack_dir == f"{tmp_path}/x"


@pytest.mark.parametrize("line, key", [
    ('colour = "red"', "colour"),
    ("kv_bits = 6", "kv_bits"),
    ("kv_bits = 8.0", "kv_bits"),
    ('eviction = "lifo"', "eviction"),
    ('memory_gb = "20"', "memory_gb"),
    ("memory_gb = 0", "memory_gb"),
    ("context_tokens = 0", "context_tokens"),
    ("context_tokens = 1.5", "context_tokens"),
    ("half_life = -1", "half_life"),
    ('thinking = "yes"', "thinking"),
    ("port = 70000", "port"),
])
def test_bad_values_raise_naming_the_key(tmp_path, line, key):
    with pytest.raises(ValueError, match=key):
        EngineConfig.read(_write(tmp_path, f'pack_dir = "p"\n{line}\n'))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_config.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'moestream.config'`

- [ ] **Step 3: Implement**

`moestream/config.py`:

```python
"""moestream.toml -> EngineConfig: every engine setting in one flat file.

Relative paths resolve against the file's directory, so the server can start from any cwd.
Omitting memory_gb means "probe free memory"; omitting calibration means a uniform slot split.
"""
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

EVICTION = ("lfu", "lru", "fifo")
KV_BITS = (0, 4, 8)


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class EngineConfig:
    pack_dir: str
    memory_gb: float | None = None
    context_tokens: int = 32768
    kv_bits: int = 0
    eviction: str = "lfu"
    half_life: float | None = None
    calibration: str | None = None
    thinking: bool = True
    port: int = 8081

    @classmethod
    def read(cls, path):
        path = Path(path)
        raw = tomllib.loads(path.read_text())
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown}")
        if "pack_dir" not in raw:
            raise ValueError(f"{path}: pack_dir is required")
        c = cls(**raw)
        c._check()
        for key in ("pack_dir", "calibration"):
            v = getattr(c, key)
            if v is not None and not Path(v).is_absolute():
                setattr(c, key, str(path.parent / v))
        return c

    def _check(self):
        rules = {
            "pack_dir": isinstance(self.pack_dir, str),
            "memory_gb": self.memory_gb is None or (_num(self.memory_gb) and self.memory_gb > 0),
            "context_tokens": _int(self.context_tokens) and self.context_tokens > 0,
            "kv_bits": _int(self.kv_bits) and self.kv_bits in KV_BITS,
            "eviction": self.eviction in EVICTION,
            "half_life": self.half_life is None or (_num(self.half_life) and self.half_life > 0),
            "calibration": self.calibration is None or isinstance(self.calibration, str),
            "thinking": isinstance(self.thinking, bool),
            "port": _int(self.port) and 0 < self.port < 65536,
        }
        for key, ok in rules.items():
            if not ok:
                raise ValueError(f"{key} = {getattr(self, key)!r} is not allowed "
                                 f"(kv_bits: {KV_BITS}, eviction: {EVICTION})")
```

`moestream.toml`:

```toml
# moeStream engine settings. Relative paths resolve against this file.
pack_dir = "../lagunaLab/models/Laguna-XS-2.1-packed"
memory_gb = 20.0          # peak GPU memory the engine may use; omit to probe free memory
context_tokens = 32768    # longest session the KV reservation is sized for
kv_bits = 0               # 0 = bf16 (lossless), 8 or 4 = quantized
eviction = "lfu"          # lfu | lru | fifo
# half_life = 64          # lfu only; omitted = take it from the calibration file
calibration = "budget.json"
thinking = true           # Laguna's built-in <think> reasoning; the model's own default is on
port = 8081
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_config.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add moestream/config.py moestream.toml tests/test_config.py
git commit -m "feat: EngineConfig read from moestream.toml"
```

---

### Task 2: Eviction policies

**Files:**
- Modify: `moestream/store.py`, `moestream/layer.py` (constructor), `moestream/load.py` (pass-through and LFU-calibration warning)
- Test: `tests/test_store.py`, `tests/test_layer.py`

**Interfaces:**
- Produces: `SlotStore(num_experts, slots, half_life=256.0, policy="lfu")`, attribute `policy`; `StreamingMoE(..., dtype, policy="lfu")`; `load(..., eviction="lfu")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`:

```python
def _victim(policy, uses):
    """2 slots, no decay; returns the expert evicted to make room for expert 9."""
    s = SlotStore(10, 2, half_life=float("inf"), policy=policy)
    for e in uses:
        s.tick(); s.assign([e])
    before = dict(s.expert_slot)
    s.tick(); s.assign([9])
    return (set(before) - set(s.expert_slot)).pop()


@pytest.mark.parametrize("policy, victim", [("lfu", 1), ("lru", 0), ("fifo", 0)])
def test_frequent_but_stale_expert(policy, victim):
    # 0 used three times then idle; 1 used once, most recently
    assert _victim(policy, [0, 0, 0, 1]) == victim


@pytest.mark.parametrize("policy, victim", [("lfu", 1), ("lru", 1), ("fifo", 0)])
def test_oldest_load_recently_reused(policy, victim):
    # 0 loaded first and used again last; 1 loaded second
    assert _victim(policy, [0, 1, 0]) == victim


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="lifo"):
        SlotStore(8, 2, policy="lifo")
```

Append to `tests/test_layer.py`:

```python
@pytest.mark.parametrize("policy", ["lru", "fifo"])
def test_eviction_policy_reaches_every_layer_and_stays_exact(packed, policy):
    out, resident = packed
    m, _ = load(out, slots=4, eviction=policy)
    assert {moe.io.store.policy for moe in _moe(m)} == {policy}
    ids = _ids(12)
    _close(_decode(m, ids, 6), resident(ids))


def test_non_lfu_policy_with_calibration_warns(packed, tmp_path):
    out, _ = packed
    layers = json.loads((out / "layout.json").read_text())["layers"]
    cal = tmp_path / "budget.json"
    cal.write_text(json.dumps({"half_life": 64.0, "grid": [2, 4],
                               "curves": {str(l): [0.5, 0.1] for l in layers}}))
    with pytest.warns(UserWarning, match="tuned for lfu"):
        m, _ = load(out, budget_gb=1.0, calibration=cal, eviction="lru")
    assert {moe.io.store.policy for moe in _moe(m)} == {"lru"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_store.py tests/test_layer.py -q`
Expected: FAIL, `TypeError: ... unexpected keyword argument 'policy'` / `'eviction'`

- [ ] **Step 3: Implement**

`moestream/store.py`, update the module docstring's second paragraph to "Eviction is decayed LFU with a recency tie-break (default), LRU or FIFO.", then:

```python
POLICIES = ("lfu", "lru", "fifo")


class SlotStore:
    def __init__(self, num_experts, slots, half_life=256.0, policy="lfu"):
        if policy not in POLICIES:
            raise ValueError(f"eviction policy {policy!r} not in {POLICIES}")
        self.slots = slots
        self.policy = policy
        self.decay = 0.5 ** (1.0 / half_life)
        self.slot_expert = [-1] * slots
        self.expert_slot = {}
        self.score = np.zeros(num_experts)
        self.last = np.zeros(num_experts, np.int64)
        self.loaded = np.zeros(num_experts, np.int64)
        self.clock = 0
        self.hits = self.misses = 0
```

In `assign`, inside the `for e, s in fills:` loop, add `self.loaded[e] = self.clock` after `self.slot_expert[s], self.expert_slot[e] = e, s`.

Replace `_victims`:

```python
    def _victims(self, n, keep):
        free = [s for s, e in enumerate(self.slot_expert) if e == -1]
        used = [s for s, e in enumerate(self.slot_expert) if e != -1 and e not in keep]
        key = {
            "lfu": lambda e: (self.score[e], self.last[e]),
            "lru": lambda e: self.last[e],
            "fifo": lambda e: self.loaded[e],
        }[self.policy]
        used.sort(key=lambda s: key(self.slot_expert[s]))
        return (free + used)[:n]
```

`moestream/layer.py`: constructor becomes
`def __init__(self, args, layer, pack_dir, layout, slots, half_life, group_size, bits, dtype, policy="lfu"):`
and the store line becomes `store=SlotStore(args.num_experts, slots, half_life, policy),`.

`moestream/load.py`: add `eviction="lfu"` to `load`'s keyword arguments. After the calibration block (`half_life = half_life or ...`), add:

```python
    if cal is not None and eviction != "lfu":
        warnings.warn(f"calibration {calibration} was tuned for lfu; using its slot split with {eviction}")
```

Pass `policy=eviction` as the last argument of the `StreamingMoE(...)` call. Change the banner's last line to
`f"half-life {half_life:g}, eviction {eviction}", file=sys.stderr)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all PASS (existing LFU tests unchanged)

- [ ] **Step 5: Commit**

```bash
git add moestream/store.py moestream/layer.py moestream/load.py tests/test_store.py tests/test_layer.py
git commit -m "feat: lru and fifo eviction policies"
```

---

### Task 3: Memory cap and KV-sized budget

**Files:**
- Modify: `moestream/budget.py`, `moestream/load.py`
- Test: `tests/test_budget.py`, `tests/test_layer.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `kv_bytes_per_token(bits=0) -> int`; `cap_budget_bytes(cap, resident, target_ctx=32768, kv_bits=0, activations=GB) -> int` (raises `ValueError`); `free_bytes() -> int`; `slot_budget_bytes(..., kv_bits=0)`; `probe(resident, **kw)` forwards `kv_bits`; `load(..., memory_gb=None, kv_bits=0)` (Task 4 adds the cache factory for `kv_bits`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_budget.py`, extend the import to
`from moestream.budget import GB, KV_BYTES_PER_TOKEN, allocate, cap_budget_bytes, kv_bytes_per_token, parse_vm_stat, slot_budget_bytes, uniform`
(add `import pytest` at the top) and append:

```python
def test_kv_bytes_per_token_by_bits():
    assert kv_bytes_per_token(0) == 40960
    assert kv_bytes_per_token(8) == 40960 * 136 // 256
    assert kv_bytes_per_token(4) == 40960 * 72 // 256


def test_cap_budget_subtracts_resident_kv_and_activations():
    assert cap_budget_bytes(20 * GB, 1 * GB, 1000) == 20 * GB - 1 * GB - 40960 * 1000 - 1 * GB


def test_fewer_kv_bits_leave_more_for_slots():
    b = [cap_budget_bytes(20 * GB, 1 * GB, 32768, kv_bits=k) for k in (0, 8, 4)]
    assert b[0] < b[1] < b[2]


def test_cap_below_fixed_costs_raises_listing_each_term():
    with pytest.raises(ValueError, match="resident.*KV.*activations"):
        cap_budget_bytes(2 * GB, 1 * GB, 32768)


def test_probe_budget_counts_kv_bits():
    assert slot_budget_bytes(30 * GB, 20 * GB, 1 * GB, target_ctx=1000, kv_bits=8) == \
        20 * GB - (1 * GB + kv_bytes_per_token(8) * 1000 + 1 * GB)
```

Append to `tests/test_layer.py`:

```python
def test_memory_gb_below_fixed_costs_raises(packed):
    out, _ = packed
    with pytest.raises(ValueError, match="resident"):
        load(out, memory_gb=1e-6)


def test_memory_gb_over_free_memory_warns_with_known_good_point(packed, monkeypatch):
    out, resident = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 0)
    with pytest.warns(UserWarning, match="free.*12.8 GB"):
        m, _ = load(out, memory_gb=50.0)
    ids = _ids(12)
    _close(m(ids), resident(ids))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_budget.py tests/test_layer.py -q`
Expected: FAIL, `ImportError: cannot import name 'cap_budget_bytes'`

- [ ] **Step 3: Implement**

`moestream/budget.py`: keep `KV_BYTES_PER_TOKEN`; add below it:

```python
KV_STORED = {0: 256, 8: 136, 4: 72}  # stored bytes per 128-dim vector (lagunaLab KV compression spec)


def kv_bytes_per_token(bits=0):
    return KV_BYTES_PER_TOKEN * KV_STORED[bits] // 256
```

Change `slot_budget_bytes` to take `kv_bits=0` as its last keyword and use
`other = resident + kv_bytes_per_token(kv_bits) * target_ctx + activations`. Add:

```python
def cap_budget_bytes(cap, resident, target_ctx=32768, kv_bits=0, activations=1 * GB):
    """Slot bytes left under an explicit total cap; raises if the fixed costs alone exceed it."""
    kv = kv_bytes_per_token(kv_bits) * target_ctx
    left = cap - resident - kv - activations
    if left <= 0:
        raise ValueError(f"memory cap {cap / GB:.2f} GB is below the fixed costs: resident {resident / GB:.2f} GB "
                         f"+ KV {kv / GB:.2f} GB ({target_ctx} tokens, kv_bits {kv_bits}) "
                         f"+ activations {activations / GB:.2f} GB")
    return int(left)


def free_bytes():
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    return parse_vm_stat(vm)
```

and make `probe` use it:

```python
def probe(resident, **kw):
    import mlx.core as mx

    return slot_budget_bytes(free_bytes(), mx.device_info()["max_recommended_working_set_size"], resident, **kw)
```

`moestream/load.py`: import `cap_budget_bytes, free_bytes` alongside the existing budget imports. Add `memory_gb=None, kv_bits=0` to `load`'s keyword arguments. Replace the `else:` budget branch with:

```python
    else:
        resident = (pack_dir / "resident.safetensors").stat().st_size
        if budget_gb is not None:
            budget = budget_gb * GB
        elif memory_gb is not None:
            budget = cap_budget_bytes(memory_gb * GB, resident, target_ctx, kv_bits)
            free = free_bytes()
            if free < memory_gb * GB:
                warnings.warn(f"memory_gb {memory_gb:g} GB but only {free / GB:.1f} GB is free; slot writes slow "
                              "about 17x once macOS pages (known good: 12.8 GB peak, 13.9 tok/s at 128 slots)")
        else:
            budget = probe(resident, target_ctx=target_ctx, kv_bits=kv_bits)
        per = _per_layer_slots(layers, E, stride, budget, cal, floor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add moestream/budget.py moestream/load.py tests/test_budget.py tests/test_layer.py
git commit -m "feat: explicit memory cap; KV reservation sized by kv_bits"
```

---

### Task 4: KV quantization and load(EngineConfig)

**Files:**
- Modify: `moestream/load.py`, `docs/specs/2026-10-13-engine-config-serve-design.md` (KV section)
- Test: `tests/test_layer.py`

**Interfaces:**
- Consumes: `EngineConfig` (Task 1); `load(..., memory_gb, kv_bits, eviction)` (Tasks 2-3); `laguna_mlx.cache.make_quantized_cache(model, bits, group_size)` and `DequantKVCache` from lagunaLab.
- Produces: `load(EngineConfig) -> (model, tokenizer)`; with `kv_bits` 8 or 4, `model.make_cache()` returns `DequantKVCache` on full-attention layers and `RotatingKVCache` on sliding ones.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layer.py`:

```python
from laguna_mlx.cache import DequantKVCache
from mlx_lm.models.cache import RotatingKVCache
from moestream.config import EngineConfig


def test_kv_bits_on_unquantizable_head_dim_fails_at_load(packed):
    out, _ = packed  # tiny model: head_dim 16, below the smallest group size (32)
    with pytest.raises(ValueError, match="head_dim"):
        load(out, slots=4, kv_bits=8)


def test_kv_bits_8_quantizes_full_layers_only_and_stays_close(tmp_path, monkeypatch):
    monkeypatch.setitem(TINY, "head_dim", 32)
    d, _ = quantized_model_dir(tmp_path)
    out = tmp_path / "packed"
    pack(d, out)
    plain, _ = load(out, slots=4)
    quant, _ = load(out, slots=4, kv_bits=8)
    kinds = [type(c) for c in quant.make_cache()]
    assert kinds == [DequantKVCache if t == "full_attention" else RotatingKVCache for t in TINY["layer_types"]]
    ids = _ids(12)
    a, b = np.array(_decode(plain, ids, 6)), np.array(_decode(quant, ids, 6))
    assert np.isfinite(b).all()
    assert np.abs(a - b).max() < 0.1 * np.abs(a).max()


def test_load_from_config_file(packed, tmp_path):
    out, resident = packed
    f = tmp_path / "engine.toml"
    f.write_text(f'pack_dir = "{out}"\nmemory_gb = 50.0\neviction = "fifo"\n')
    m, _ = load(EngineConfig.read(f))
    assert {moe.io.store.policy for moe in _moe(m)} == {"fifo"}
    ids = _ids(12)
    _close(_decode(m, ids, 6), resident(ids))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_layer.py -q`
Expected: FAIL. The head_dim test with `DID NOT RAISE` (Task 3 accepts `kv_bits` but builds no quantized cache yet), the cache-kind test on `KVCache`, the config test with `TypeError` from `Path(EngineConfig)`.

- [ ] **Step 3: Implement**

`moestream/load.py`: add `from moestream.config import EngineConfig` to the imports. At the top of `load`'s body:

```python
    if isinstance(pack_dir, EngineConfig):
        c = pack_dir
        return load(c.pack_dir, memory_gb=c.memory_gb, calibration=c.calibration, half_life=c.half_life,
                    target_ctx=c.context_tokens, kv_bits=c.kv_bits, eviction=c.eviction, floor=floor)
```

After `mx.eval(model.parameters())`, add:

```python
    if kv_bits:
        from laguna_mlx.cache import make_quantized_cache

        gs = 64 if args.head_dim % 64 == 0 else 32
        make_quantized_cache(model, bits=kv_bits, group_size=gs)  # bad head_dim fails here, not mid-generation
        model.make_cache = lambda: make_quantized_cache(model, bits=kv_bits, group_size=gs)
```

Extend the banner's last line to `f"half-life {half_life:g}, eviction {eviction}, kv {kv_bits or 16}-bit"`.

Spec: in the "KV quantization" section, replace the first paragraph with:

```markdown
`load` replaces the model's cache factory when `kv_bits` is 8 or 4, using lagunaLab's
`laguna_mlx.cache.make_quantized_cache`: K/V stored at `kv_bits` on the 10 full-attention layers
(`DequantKVCache`, attends in bf16 through the fused kernel), `RotatingKVCache` unchanged on the
sliding ones. Not raw mlx-lm `QuantizedKVCache`: lagunaLab found its quantized attention builds
the full score matrix (~16 GB per layer at 82k context) and hung the machine. `kv_bits = 0`
leaves `laguna.Model.make_cache` as it is.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add moestream/load.py tests/test_layer.py docs/specs/2026-10-13-engine-config-serve-design.md
git commit -m "feat: kv_bits via lagunaLab's quantized cache; load(EngineConfig)"
```

- [ ] **Step 6: Tell the coordinator** (SendMessage to the coordinator session): moeStream now imports `laguna_mlx.cache.make_quantized_cache` and `DequantKVCache`; please add them to RULES.md rule 3's frozen list.

---

### Task 5: Server

**Files:**
- Create: `tools/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `EngineConfig` (Task 1), `kv_bytes_per_token` (Task 3), `load(EngineConfig)` (Task 4), `scripts.serve_streaming` (import side effect: SSE keepalive filter).
- Produces: `server_argv(cfg) -> list[str]`; `make_loader(cfg) -> callable(*a, **k) -> (model, tokenizer)`; `python3 -m tools.serve [--config PATH] [extra mlx_lm.server flags]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve.py`:

```python
import json

from moestream.config import EngineConfig
from tools.serve import server_argv


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_argv_from_defaults():
    a = server_argv(EngineConfig(pack_dir="/p", port=9000))
    assert _flag(a, "--model") == "/p" and _flag(a, "--port") == "9000"
    assert _flag(a, "--prompt-cache-bytes") == str(40960 * 32768)
    assert json.loads(_flag(a, "--chat-template-args")) == {"enable_thinking": True}
    assert (_flag(a, "--temp"), _flag(a, "--top-k"), _flag(a, "--top-p")) == ("1.0", "20", "1.0")
    assert _flag(a, "--max-tokens") == "4096" and _flag(a, "--prompt-cache-size") == "6"


def test_argv_follows_kv_bits_context_and_thinking():
    a = server_argv(EngineConfig(pack_dir="/p", kv_bits=8, context_tokens=1000, thinking=False))
    assert _flag(a, "--prompt-cache-bytes") == str(40960 * 136 // 256 * 1000)
    assert json.loads(_flag(a, "--chat-template-args")) == {"enable_thinking": False}


def test_every_flag_is_one_mlx_lm_server_accepts():
    import inspect

    import mlx_lm.server

    src = inspect.getsource(mlx_lm.server)
    for f in server_argv(EngineConfig(pack_dir="/p"))[::2]:
        assert f'"{f}"' in src, f
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests/test_serve.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.serve'`

- [ ] **Step 3: Implement**

`tools/serve.py`:

```python
"""Serve moeStream through mlx-lm's OpenAI-compatible server, configured by moestream.toml.

PYTHONPATH=.:../lagunaLab python3 -m tools.serve [--config moestream.toml] [mlx_lm.server flags]
Loads real weights: run it as
  .gpu/gpu-run moe tools/watch.sh 8 21 python3 -m tools.serve
Flags given on the command line override the ones derived from the config.
"""
import json
import sys
from pathlib import Path

from moestream.budget import kv_bytes_per_token
from moestream.config import EngineConfig


def server_argv(cfg):
    return [
        "--model", str(cfg.pack_dir), "--port", str(cfg.port), "--max-tokens", "4096",
        "--prompt-cache-size", "6",
        "--prompt-cache-bytes", str(kv_bytes_per_token(cfg.kv_bits) * cfg.context_tokens),
        "--chat-template-args", json.dumps({"enable_thinking": cfg.thinking}),
        "--temp", "1.0", "--top-k", "20", "--top-p", "1.0",  # generation_config.json's sampling
    ]


def make_loader(cfg):
    def _load(*a, **k):
        from moestream.load import load

        model, tok = load(cfg)
        # generation_config.json's eos_token_id is [2, 24]; 24 is "</assistant>". mlx-lm only
        # infers 2, so without this the model never stops at end of turn (as in lagunaLab).
        gen = Path(cfg.pack_dir) / "generation_config.json"
        if gen.exists():
            for eid in json.loads(gen.read_text()).get("eos_token_id", []):
                if eid not in tok.eos_token_ids:
                    tok.add_eos_token(str(eid))
        return model, tok

    return _load


def main():
    import mlx_lm.server as server

    import scripts.serve_streaming  # noqa: F401  installs lagunaLab's SSE keepalive filter on the handler

    path = "moestream.toml"
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        path = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    cfg = EngineConfig.read(path)
    server.load = make_loader(cfg)
    sys.argv = sys.argv[:1] + server_argv(cfg) + sys.argv[1:]
    server.main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MOESTREAM_CPU=1 PYTHONPATH=.:../lagunaLab python3 -m pytest tests -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tools/serve.py tests/test_serve.py
git commit -m "feat: tools.serve, mlx-lm server driven by moestream.toml"
```

---

### Task 6: Real-weight verification, pi hookup, README

Real weights. Every run goes through the GPU lock and the watchdog, behind lagunaLab and behind the two queued equal-memory benches. Do not start until `results/bench_cap14.4.json` and the `--cap-gb 20` run have finished or aborted.

**Files:**
- Modify: `README.md`
- Outside the repo, after the user confirms: `~/.pi/agent/models.json`
- Create: `results/serve_verify.log`

- [ ] **Step 1: Start the server under lock and watchdog** (background)

```bash
cd /Users/srujanpendurthi/Projects/Current/moeStream
PYTHONPATH=.:../lagunaLab /Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe \
  tools/watch.sh 8 21 python3 -m tools.serve > results/serve_verify.log 2>&1
```

Wait until the log shows the moeStream banner and the server's listening line. If the watchdog prints `aborted: paging`, stop and report: 20 GB is not usable on this machine without smaller slots.

- [ ] **Step 2: One completion over HTTP**

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Laguna-XS-2.1","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"max_tokens":512}' \
  | python3 -m json.tool | tee -a results/serve_verify.log
```

Expected: a `choices[0].message.content` answer and a `reasoning` field (thinking on).

- [ ] **Step 3: Ask the user to confirm the pi edit**, then add this provider next to `llamacpp` in `~/.pi/agent/models.json`:

```json
"moestream": {
  "baseUrl": "http://127.0.0.1:8081/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "compat": { "supportsDeveloperRole": false },
  "models": [{ "id": "Laguna-XS-2.1", "reasoning": true, "contextWindow": 32768, "maxTokens": 4096 }]
}
```

- [ ] **Step 4: Multi-turn pi session with tool calls** on a throwaway repo:

```bash
W=$(mktemp -d) && cd "$W" && git init -q
printf 'def mean(xs):\n    return sum(xs) / len(xs) + 1\n' > stats.py
printf 'from stats import mean\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n' > test_stats.py
time pi --provider moestream --model Laguna-XS-2.1 -p "Run the tests in this repo with python3 -m pytest and tell me what fails."
time pi --provider moestream --model Laguna-XS-2.1 -c -p "Fix the bug so the test passes, then run the tests again."
time pi --provider moestream --model Laguna-XS-2.1 -c -p "Add a test for an empty list and make mean raise ValueError for it."
python3 -m pytest -q
```

Pass: all three turns complete with tool calls and the final pytest passes. Record wall time per turn. From the server log, record `cached_tokens` growing turn over turn, if the server reports it.

- [ ] **Step 5: Stop the server and record peaks.** Kill the server's python process by PID (`kill <pid>` from `ps -axo pid,command | grep tools.serve`, never `pkill -f`). Copy the watchdog's `watch: peak swap ... peak footprint ...` line from `results/serve_verify.log`. Pass: footprint at or under 20 GB.

- [ ] **Step 6: Setting changes take effect.** For each of `eviction = "lru"`, `kv_bits = 8`, `thinking = false` (one at a time, in a copy at `results/verify.toml` with the other keys from `moestream.toml` and `pack_dir` made absolute), repeat Step 1 with `--config results/verify.toml` and Step 2. Expect: the banner shows the new eviction / kv bits; with `thinking = false` the response has no `reasoning`. Record decode time and output tokens per run.

- [ ] **Step 7: README.** Under `## Run` add:

````markdown
### Serve (pi, any OpenAI client)

Settings live in `moestream.toml` (memory cap, context, KV bits, eviction, thinking).

```bash
PYTHONPATH=.:../lagunaLab ../.gpu/gpu-run moe tools/watch.sh 8 21 python3 -m tools.serve
```

pi provider (`~/.pi/agent/models.json`), `contextWindow` = `context_tokens`, `reasoning` = `thinking`:

```json
"moestream": {"baseUrl": "http://127.0.0.1:8081/v1", "api": "openai-completions", "apiKey": "none",
  "compat": {"supportsDeveloperRole": false},
  "models": [{"id": "Laguna-XS-2.1", "reasoning": true, "contextWindow": 32768, "maxTokens": 4096}]}
```
````

and a `## Sub-project 1 verification` section with the numbers from Steps 2-6, each citing `results/serve_verify.log`.

- [ ] **Step 8: Commit and report**

```bash
git add README.md results/serve_verify.log
git commit -m "docs: engine served into pi, verification results"
```

Report to the coordinator session and the user: pass/fail per step, peak footprint and swap.
