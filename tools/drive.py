"""Drive the server with lagunaLab's captured agent sessions, one request per assistant turn.

PYTHONPATH=.:../lagunaLab python3 -m tools.drive [--minutes 60] [--url http://127.0.0.1:8081/v1]
Light client: the server holds the GPU.

--live STATE: live use, as an agent harness would send it: sessions interleaved turn by turn, each
conversation keeping the model's own replies (then the recorded tool output), so every prompt
extends the previous turn's cache. STATE (JSON) carries the conversations across runs and restarts.
"""
import argparse
import glob
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def _wire(m):
    """OpenAI wire format: tool-call arguments are a JSON string; captured sessions hold dicts."""
    if not m.get("tool_calls"):
        return m
    calls = [{**c, "function": {**c["function"], "arguments": json.dumps(c["function"]["arguments"])}}
             if not isinstance(c["function"]["arguments"], str) else c for c in m["tool_calls"]]
    return {**m, "tool_calls": calls}


def turns(msgs, max_chars=None):
    """Each assistant turn's prefix; prefixes over max_chars (JSON) are skipped, so decode, not prefill, dominates."""
    msgs = [_wire(m) for m in msgs]
    out = [msgs[:i] for i, m in enumerate(msgs) if m["role"] == "assistant" and i]
    return [t for t in out if max_chars is None or len(json.dumps(t)) <= max_chars]


def _post(url, body):
    req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def _session(f):
    from replay.session import load_session

    msgs, tools = load_session(f)
    return [_wire(m) for m in msgs], tools


def _text(m):
    return (m.get("reasoning") or "") + (m.get("content") or "") + (json.dumps(m["tool_calls"]) if m.get("tool_calls") else "")


def live(state, files, rounds, url, log, tag, pause=5, max_tokens=1024, probe_after=None, probe=None):
    """`rounds` turns of each unfinished session, interleaved. probe_after: after each turn, two greedy
    48-token continuations of the next prompt (RAM hits), kept in that JSON file per session; probe:
    one greedy continuation per session, without turns. Probes never advance a conversation."""
    state, log = Path(state), Path(log)
    st = json.loads(state.read_text()) if state.exists() else {}
    rec = {f: _session(f) for f in files}
    for f in files:
        if f not in st:
            msgs = rec[f][0]
            i = next(i for i, m in enumerate(msgs) if m["role"] == "assistant" and i)
            st[f] = {"history": msgs[:i], "next": i, "turn": 0}

    def ask(f, greedy):
        body = {"model": "Laguna-XS-2.1", "messages": list(st[f]["history"]), "tools": rec[f][1],
                "max_tokens": 48 if greedy else max_tokens}
        if greedy:
            body["temperature"] = 0.0
        t = time.time()
        r = _post(url, body)
        return r, time.time() - t

    def probes(out, f, n):
        out = Path(out)
        got = json.loads(out.read_text()) if out.exists() else {}
        got[f] = [_text(ask(f, True)[0]["choices"][0]["message"]) for _ in range(n)]
        out.write_text(json.dumps(got, indent=1))

    for _ in range(rounds):
        for f in files:
            s, msgs = st[f], rec[f][0]
            if s["next"] >= len(msgs):
                continue
            try:
                r, sec = ask(f, False)
            except urllib.error.HTTPError as e:  # logged; this session stops, the others go on
                print(f"{f} turn {s['turn']}: HTTP {e.code} {e.read()[:500]!r}", flush=True)
                s["next"] = len(msgs)
                continue
            c, u = r["choices"][0], r.get("usage", {})
            with log.open("a") as fh:
                fh.write(json.dumps({"tag": tag, "session": f, "turn": s["turn"], "t": round(time.time(), 1),
                                     "s": round(sec, 2), "prompt_tokens": u.get("prompt_tokens"),
                                     "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens"),
                                     "completion_tokens": u.get("completion_tokens"),
                                     "finish": c.get("finish_reason")}) + "\n")
            m = c["message"]
            s["history"].append({k: m[k] for k in ("role", "content", "reasoning", "tool_calls") if m.get(k) is not None
                                 or k == "content"})
            j = s["next"] + 1
            while j < len(msgs) and msgs[j]["role"] != "assistant":
                s["history"].append(msgs[j])
                j += 1
            s["next"], s["turn"] = j, s["turn"] + 1
            state.write_text(json.dumps(st))
            time.sleep(pause)  # tool time; also lets the snapshot writer finish before the probes
            if probe_after and j < len(msgs):
                probes(probe_after, f, 2)
    if probe:
        for f in files:
            probes(probe, f, 1)
    state.write_text(json.dumps(st))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="../lagunaLab/traces")
    p.add_argument("--url", default="http://127.0.0.1:8081/v1")
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--max-chars", type=int, default=16000, help="skip turns whose prefix is longer (about 4k tokens)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--pause", type=float, default=5, help="idle seconds between turns (tool time); the tuner works only when idle")
    p.add_argument("--live", help="state file: live use (see the module docstring)")
    p.add_argument("--sessions", help="comma-separated trace file names, for --live")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--log", default="results/drive_live.jsonl")
    p.add_argument("--tag", default="live")
    p.add_argument("--probe-after")
    p.add_argument("--probe")
    a = p.parse_args()
    if a.live:
        files = [f"{a.traces}/{n}" for n in a.sessions.split(",")]
        return live(a.live, files, a.rounds, a.url, a.log, a.tag, a.pause, a.max_tokens, a.probe_after, a.probe)
    end = time.time() + 60 * a.minutes
    while time.time() < end:
        for f in sorted(glob.glob(f"{a.traces}/*.jsonl")):
            msgs, tools = _session(f)
            for prefix in turns(msgs, a.max_chars):
                if time.time() >= end:
                    return
                body = {"model": "Laguna-XS-2.1", "messages": prefix, "max_tokens": a.max_tokens, "tools": tools}
                req = urllib.request.Request(f"{a.url}/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(req, timeout=600).read()
                except urllib.error.HTTPError as e:  # one failed turn is logged, the drive goes on
                    print(f"{f} turn {len(prefix)}: HTTP {e.code} {e.read()[:500]!r}", flush=True)
                time.sleep(a.pause)
            print(f, flush=True)


if __name__ == "__main__":
    main()
