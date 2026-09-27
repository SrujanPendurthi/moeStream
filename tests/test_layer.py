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
    got, want = _decode(m, ids, 8), _decode(resident, ids, 8)  # bf16: same cache path, not a full pass
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


def test_prefill_does_not_leave_a_slot_budget_of_buffers_behind(packed):
    out, _ = packed
    m, _ = load(out, slots=TINY["num_experts"])
    m(_ids(64))
    for moe in _moe(m):
        assert len(moe.io.prefetcher.free) <= 2 * TINY["num_experts_per_tok"]


def test_prefill_scores_count_tokens_not_distinct_experts(packed):
    out, _ = packed
    m, _ = load(out, slots=TINY["num_experts"], half_life=float("inf"))
    m(_ids(32))
    for moe in _moe(m):
        assert moe.io.store.score.sum() == 32 * TINY["num_experts_per_tok"]
