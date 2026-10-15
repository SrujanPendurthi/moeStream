import pytest

from tools.parity import parse_calls, valid_call

TOOLS = [
    {"type": "function", "function": {"name": "shell", "parameters": {
        "type": "object", "required": ["cmd"], "additionalProperties": False,
        "properties": {"cmd": {"type": "string"}, "description": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "required": ["path"], "additionalProperties": False,
        "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"},
                       "flags": {"type": "array"}}}}},
]


def _call(name, **kv):
    return f"<tool_call>{name}" + "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"
                                          for k, v in kv.items()) + "</tool_call>"


def test_a_valid_call_parses_and_passes():
    calls = parse_calls(_call("shell", cmd="ls -la", description="list"))
    assert calls == [("shell", {"cmd": "ls -la", "description": "list"})]
    assert valid_call(calls[0], TOOLS) == (True, "")


@pytest.mark.parametrize("text, reason", [
    (_call("rm_rf", cmd="x"), "unknown tool"),
    (_call("read", start_line="3"), "missing required"),
    (_call("read", path="/a", start_line="abc"), "start_line"),
    (_call("read", path="/a", flags="not json"), "flags"),
    (_call("shell", cmd="ls", colour="red"), "unknown key"),
])
def test_invalid_calls_say_why(text, reason):
    ok, why = valid_call(parse_calls(text)[0], TOOLS)
    assert not ok and reason in why


def test_an_unterminated_call_is_none_and_invalid():
    calls = parse_calls("<tool_call>shell<arg_key>cmd</arg_key><arg_value>ls")
    assert calls == [None]
    assert valid_call(None, TOOLS)[0] is False


def test_every_call_in_an_action_is_parsed_and_plain_text_has_none():
    text = "thinking...\n" + _call("shell", cmd="ls") + "\n" + _call("read", path="/a", start_line="2")
    assert [c[0] for c in parse_calls(text)] == ["shell", "read"]
    assert parse_calls("I am done.") == []


def test_is_tie_needs_both_tokens_on_top_and_a_gap_of_a_few_bf16_ulps():
    import mlx.core as mx

    from tools.parity import is_tie

    # around 16, a bf16 ulp is 2^(4-7) = 0.125
    assert is_tie(mx.array([16.0, 15.875, 3.0]), 0, 1)              # 1 ulp apart
    assert is_tie(mx.array([16.0, 15.5, 3.0]), 1, 0)                # 4 ulps, order does not matter
    assert not is_tie(mx.array([16.0, 15.0, 3.0]), 0, 1)            # 8 ulps
    assert not is_tie(mx.array([16.0, 15.9, 15.95]), 0, 1)          # b is third


def test_first_diff_on_token_ids():
    from tools.parity import first_diff

    assert first_diff([1, 2, 3], [1, 2, 3]) is None
    assert first_diff([1, 2, 3], [1, 2, 4]) == 2
    assert first_diff([1, 2], [1, 2, 9]) == 2


def test_classify_agrees_with_is_tie_on_the_models_own_logits(tmp_path):
    import mlx.core as mx

    from moestream.load import load
    from scripts.pack_experts import pack
    from tests.tiny import quantized_model_dir
    from tools.parity import classify, is_tie

    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    model, _ = load(tmp_path / "packed", slots=4)
    prefix = [5, 9, 13, 2, 7]
    logits = model(mx.array([prefix]))[0, -1]
    top2 = mx.argsort(-logits)[:2].tolist()
    ref, got = [top2[0], 3], [top2[1], 3]
    assert classify(model, prefix, ref, ref) == "exact"
    want = "tie" if is_tie(logits, top2[0], top2[1]) else "real"
    assert classify(model, prefix, ref, got) == want


def _tiny(tmp_path):
    from moestream.load import load
    from scripts.pack_experts import pack
    from tests.tiny import quantized_model_dir

    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    return load(tmp_path / "packed", slots=4)[0]


def _self_ref(model, path, n=40, seed=0, k=16):
    """A reference in build_ref's format, built from the model's own log-probs (no tokenizer)."""
    import mlx.core as mx
    import numpy as np

    from replay.replay import assistant_logprobs
    from tests.tiny import TINY

    rng = np.random.default_rng(seed)
    ids = rng.integers(0, TINY["vocab_size"], size=n).astype(np.int32)
    mask = np.zeros(n, bool)
    mask[n // 2:] = True
    pos, top_ids, top_lp = [], [], []
    for p, lp in assistant_logprobs(model, ids, mask, chunk=16):
        order = mx.argsort(-lp, axis=-1)[:, :k]
        pos.append(p)
        top_ids.append(np.array(order))
        top_lp.append(np.array(mx.take_along_axis(lp, order, axis=-1)))
    np.savez(path, ids=ids, mask=mask, truncated=False, pos=np.concatenate(pos),
             top_ids=np.concatenate(top_ids), top_lp=np.concatenate(top_lp))


def test_a_self_reference_scores_exact_through_both_engine_paths(tmp_path):
    from tools.parity import score_sessions

    model = _tiny(tmp_path)
    (tmp_path / "refs").mkdir()
    _self_ref(model, tmp_path / "refs" / "s1.npz")
    for fwd in (None, lambda ids, cache: model(ids, cache=cache)):
        res = score_sessions(model, tmp_path / "refs", ["s1"], tmp_path / "out.json", chunk=16, forward=fwd)
        assert res["summary"]["kl_top64_mean"] < 1e-6 and res["summary"]["top1_agree"] == 1.0


def test_summary_weights_by_assistant_tokens_and_a_bad_reference_does_not_stop_the_run(tmp_path):
    import json

    import numpy as np

    from tools.parity import gate, score_sessions

    model = _tiny(tmp_path)
    (tmp_path / "refs").mkdir()
    _self_ref(model, tmp_path / "refs" / "bad.npz")
    bad = dict(np.load(tmp_path / "refs" / "bad.npz"))
    bad["pos"] = bad["pos"] + 1                                  # positions no longer match the ids
    np.savez(tmp_path / "refs" / "bad.npz", **bad)
    _self_ref(model, tmp_path / "refs" / "a.npz", n=40, seed=1)
    _self_ref(model, tmp_path / "refs" / "b.npz", n=80, seed=2)
    res = score_sessions(model, tmp_path / "refs", ["bad", "a", "b"], tmp_path / "out.json", chunk=16)
    assert "error" in res["sessions"]["bad"] and res["summary"]["sessions"] == 2
    assert res["summary"]["assistant_tokens"] == 20 + 40
    assert json.loads((tmp_path / "out.json").read_text())["summary"] == res["summary"]
    assert gate(res["summary"], kl=0.02, top1=0.98)
    assert not gate({"kl_top64_mean": 0.03, "top1_agree": 0.99}, kl=0.02, top1=0.98)


@pytest.mark.parametrize("experts, want_cal", [(128, False), (256, True)])
def test_load_engine_skips_a_calibration_made_for_another_expert_count(tmp_path, monkeypatch, experts, want_cal):
    import json

    import moestream.load
    from tools.parity import load_engine

    (tmp_path / "pack").mkdir()
    (tmp_path / "pack" / "layout.json").write_text(json.dumps({"num_experts": experts}))
    (tmp_path / "budget.json").write_text(json.dumps({"grid": [16, 64, 256]}))
    seen = {}
    monkeypatch.setattr(moestream.load, "load", lambda cfg: seen.setdefault("cfg", cfg) and ("m", "t"))
    monkeypatch.setattr("tools.parity.cap_buffer_pool", lambda gb: seen.setdefault("pool", gb))
    load_engine(tmp_path / "pack", kv_bits=8, memory_gb=12.0, context_tokens=90000, calibration=tmp_path / "budget.json")
    c = seen["cfg"]
    assert (c.calibration is not None) == want_cal and c.kv_bits == 8 and c.lookahead is False
    assert c.memory_gb == 12.0 and c.context_tokens == 90000 and seen["pool"] == 1.0
