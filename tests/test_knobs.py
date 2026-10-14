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


def test_new_slot_arrays_start_as_zeros_not_a_random_float32_tensor(bf16, monkeypatch):
    m, _ = load(bf16[0], slots=4)
    moe = [l.mlp for l in m.layers if isinstance(l.mlp, StreamingMoE)][0]
    glu = moe.io.make_glu(6)
    assert glu.gate_proj.weight.shape[0] == 6 and not mx.any(glu.gate_proj.weight).item()
    cleared = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(1))
    k = {k.name: k for k in system_knobs(m, max_slots=12)}["slots"]
    k.apply({1: 2, 2: 6, 3: 4})
    assert cleared  # freed slot buffers go back to the OS, or a shrink frees nothing
