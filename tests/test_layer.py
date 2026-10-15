import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_lm.models.cache import RotatingKVCache

from moestream.config import EngineConfig
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
    for c, t in zip(quant.make_cache(), TINY["layer_types"]):
        if t == "full_attention":
            # mlx-lm routes any cache with `bits` to its score-materializing attention (machine hang)
            assert c.kv_bits == 8 and not hasattr(c, "bits")
        else:
            assert isinstance(c, RotatingKVCache)
    ids = _ids(12)
    a, b = np.array(_decode(plain, ids, 6)), np.array(_decode(quant, ids, 6))
    assert np.isfinite(b).all()
    assert np.abs(a - b).max() < 0.1 * np.abs(a).max()


def test_load_from_config_file(packed, tmp_path, monkeypatch):
    out, resident = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 10**12)  # independent of what else is running
    f = tmp_path / "engine.toml"
    f.write_text(f'pack_dir = "{out}"\nmemory_gb = 50.0\neviction = "fifo"\nkv_bits = 0\n')  # tiny head_dim 16 cannot quantize
    m, _ = load(EngineConfig.read(f))
    assert {moe.io.store.policy for moe in _moe(m)} == {"fifo"}
    ids = _ids(12)
    _close(_decode(m, ids, 6), resident(ids))


def test_kv_sessions_2_reserves_kv_for_the_live_and_one_stored_session(packed, tmp_path, monkeypatch):
    out, _ = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 10**12)
    # 3.5 GB covers resident + activations (1.3 + 0.021 x 32.768 = 1.99 GB) + one 32k-token bf16 KV
    # (1.34 GB), not two
    with pytest.raises(ValueError, match="KV"):
        load(EngineConfig(pack_dir=str(out), memory_gb=3.5, kv_bits=0, context_tokens=32768, kv_sessions=2))


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


def test_eval_groups_evaluates_each_group_and_stays_exact(packed, monkeypatch):
    import moestream.layer as layer

    out, resident = packed
    m, _ = load(out, slots=3)
    ids = _ids(20)
    calls = []
    real = mx.eval
    monkeypatch.setattr(layer.mx, "eval", lambda *a: calls.append(len(a)) or real(*a))
    m(ids)
    plain = len(calls)
    calls.clear()
    monkeypatch.setattr(layer, "EVAL_GROUPS", True)
    _close(m(ids), resident(ids))
    assert len(calls) > plain        # 3 slots < experts routed by 20 tokens: several groups per layer


def _slots(m):
    return [list(moe.io.store.slot_expert) for moe in _moe(m)]


def test_prefill_set_is_exact_and_leaves_the_slots_as_they_were(packed, monkeypatch):
    import moestream.layer as layer

    out, resident = packed
    m, _ = load(out, slots=3)
    ids = _ids(20)
    m(ids[:, :1])                                    # decode-sized call: fills a few slots normally
    before, misses = _slots(m), stream_stats(m)["misses"]
    monkeypatch.setattr(layer, "PREFILL_SET", True)
    _close(m(ids), resident(ids))
    assert _slots(m) == before                       # prefill did not evict what decode loaded
    assert stream_stats(m)["misses"] > misses        # experts not in slots were read and counted


def test_prefill_set_only_applies_when_a_layer_needs_more_than_one_group(packed, monkeypatch):
    import moestream.layer as layer

    out, resident = packed
    m, _ = load(out, slots=TINY["num_experts"])
    monkeypatch.setattr(layer, "PREFILL_SET", True)
    ids = _ids(20)
    _close(m(ids), resident(ids))
    assert any(e != -1 for s in _slots(m) for e in s)  # fits in one group: slots used as before


def test_prefill_chunks_evaluated_as_mlx_lm_does_do_not_pile_up_slot_writes(packed):
    """mlx-lm's prefill evaluates only the caches per chunk; nothing reads the last layer's expert
    output, so its slot writes must not stay pending across chunks (results/prefill_mem_main.json)."""
    out, _ = packed
    m, _ = load(out, slots=3)
    cache = m.make_cache()
    rng = np.random.default_rng(0)
    mx.eval(m.parameters())
    active = []
    for _ in range(12):
        m(mx.array(rng.integers(0, TINY["vocab_size"], size=(1, 16))), cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        active.append(mx.get_active_memory())
    assert active[-1] - active[2] < 9 * 2000      # was ~25 KB per chunk


def test_kv_sessions_1_reserves_kv_for_the_live_session_only(packed, monkeypatch):
    out, _ = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 10**12)
    load(EngineConfig(pack_dir=str(out), memory_gb=3.5, kv_bits=0, context_tokens=32768, kv_sessions=1))


def test_config_load_reserves_activations_by_context_and_says_so(packed, monkeypatch, capsys):
    out, _ = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 10**12)
    # 3.3 GB covered the flat 1 GB reserve plus one 32k bf16 KV; the context-dependent 1.99 GB does not
    with pytest.raises(ValueError, match="activations 1.99 GB"):
        load(EngineConfig(pack_dir=str(out), memory_gb=3.3, kv_bits=0, context_tokens=32768, kv_sessions=1))
    load(EngineConfig(pack_dir=str(out), memory_gb=8.0, kv_bits=0, context_tokens=32768))
    assert "activations 1.99 GB" in capsys.readouterr().err


def test_lookahead_false_turns_lookahead_off_on_every_layer(packed, monkeypatch):
    out, _ = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 10**12)
    m, _ = load(EngineConfig(pack_dir=str(out), memory_gb=8.0, kv_bits=0, context_tokens=1024))   # default: off
    assert all(not moe.io.lookahead for moe in _moe(m))
    assert all(moe.io.lookahead for moe in _moe(load(EngineConfig(pack_dir=str(out), memory_gb=8.0, kv_bits=0,
                                                                   context_tokens=1024, lookahead=True))[0]))


def test_the_banner_says_how_many_sessions_of_kv_are_reserved(packed, monkeypatch, capsys):
    out, _ = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 10**12)
    load(EngineConfig(pack_dir=str(out), memory_gb=8.0, kv_bits=0, context_tokens=1024))
    assert "KV for 1 session of 1024 tokens" in capsys.readouterr().err


def test_one_kv_session_without_the_ssd_tier_warns(packed, monkeypatch):
    out, _ = packed
    monkeypatch.setattr("moestream.load.free_bytes", lambda: 10**12)
    with pytest.warns(UserWarning, match="kv_sessions = 1.*kv_tier"):
        load(EngineConfig(pack_dir=str(out), memory_gb=8.0, kv_bits=0, context_tokens=1024, kv_tier=False))
