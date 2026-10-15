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
