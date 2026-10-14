from tools.drive import turns


def test_each_assistant_turn_becomes_a_request_of_what_came_before():
    msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}, {"role": "assistant"}]
    assert [len(t) for t in turns(msgs)] == [2, 4]


def test_tool_call_arguments_are_sent_as_json_strings():
    call = {"function": {"name": "shell", "arguments": {"cmd": "ls"}}}
    msgs = [{"role": "user"}, {"role": "assistant", "tool_calls": [call]}, {"role": "tool"}, {"role": "assistant"}]
    sent = turns(msgs)[1][1]["tool_calls"][0]["function"]["arguments"]
    assert sent == '{"cmd": "ls"}'
    assert call["function"]["arguments"] == {"cmd": "ls"}  # the loaded session is not changed


def test_turns_longer_than_max_chars_are_skipped():
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant"}, {"role": "user", "content": "y" * 1000}, {"role": "assistant"}]
    assert [len(t) for t in turns(msgs, max_chars=500)] == [1]
