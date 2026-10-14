from tools.drive import turns


def test_each_assistant_turn_becomes_a_request_of_what_came_before():
    msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "tool"}, {"role": "assistant"}]
    assert [len(t) for t in turns(msgs)] == [2, 4]
