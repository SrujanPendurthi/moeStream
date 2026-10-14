"""Drive the server with lagunaLab's captured agent sessions, one request per assistant turn.

PYTHONPATH=.:../lagunaLab python3 -m tools.drive [--minutes 60] [--url http://127.0.0.1:8081/v1]
Light client: the server holds the GPU.
"""
import argparse
import glob
import json
import time
import urllib.error
import urllib.request


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


def main():
    from replay.session import load_session

    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="../lagunaLab/traces")
    p.add_argument("--url", default="http://127.0.0.1:8081/v1")
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--max-chars", type=int, default=16000, help="skip turns whose prefix is longer (about 4k tokens)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--pause", type=float, default=5, help="idle seconds between turns (tool time); the tuner works only when idle")
    a = p.parse_args()
    end = time.time() + 60 * a.minutes
    while time.time() < end:
        for f in sorted(glob.glob(f"{a.traces}/*.jsonl")):
            msgs, tools = load_session(f)
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
