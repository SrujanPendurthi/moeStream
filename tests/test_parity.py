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
