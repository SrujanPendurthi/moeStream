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



SESSION = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
           {"role": "assistant", "content": "recorded 1"}, {"role": "tool", "content": "t1"},
           {"role": "assistant", "content": "recorded 2"}, {"role": "tool", "content": "t2"},
           {"role": "assistant", "content": "recorded 3"}]


def _fake(monkeypatch, sent):
    import tools.drive as drive

    def post(url, body):
        sent.append(body)
        n = sum(len(m.get("content") or "") for m in body["messages"])
        return {"choices": [{"message": {"role": "assistant", "content": f"live {len(sent)}", "reasoning": None},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 1}}}
    monkeypatch.setattr(drive, "_post", post)
    monkeypatch.setattr(drive, "_session", lambda f: ([dict(m) for m in SESSION], [{"tool": f}]))


def test_live_use_keeps_the_models_replies_and_the_recorded_tool_output(tmp_path, monkeypatch):
    import json

    from tools.drive import live

    sent = []
    _fake(monkeypatch, sent)
    state, log = tmp_path / "state.json", tmp_path / "log.jsonl"
    live(state, ["a", "b"], rounds=1, url="u", log=log, tag="off", pause=0)
    live(state, ["a", "b"], rounds=5, url="u", log=log, tag="off", pause=0)   # resumes; stops when sessions end
    assert [b["tools"] for b in sent] == [[{"tool": "a"}], [{"tool": "b"}]] * 3
    assert [m["content"] for m in sent[4]["messages"]] == ["s", "u", "live 1", "t1", "live 3", "t2"]
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert [(r["tag"], r["session"], r["turn"]) for r in rows[:2]] == [("off", "a", 0), ("off", "b", 0)]
    assert rows[0]["cached_tokens"] == 1 and rows[0]["finish"] == "stop" and len(rows) == 6


def test_probes_are_greedy_and_do_not_advance_the_conversation(tmp_path, monkeypatch):
    import json

    from tools.drive import live

    sent = []
    _fake(monkeypatch, sent)
    state, out = tmp_path / "state.json", tmp_path / "probe.json"
    live(state, ["a"], rounds=1, url="u", log=tmp_path / "l", tag="on", pause=0, probe_after=out)
    assert [b["temperature"] for b in sent[1:]] == [0.0, 0.0] and len(sent[1]["messages"]) == len(sent[2]["messages"]) == 4
    assert json.loads(out.read_text())["a"] == ["live 2", "live 3"]
    live(state, ["a"], rounds=0, url="u", log=tmp_path / "l", tag="restart", pause=0, probe=tmp_path / "p2.json")
    assert json.loads((tmp_path / "p2.json").read_text()) == {"a": ["live 4"]}
    assert json.loads(state.read_text())["a"]["history"][-1]["content"] == "t1"
