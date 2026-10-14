"""Drive the server with lagunaLab's captured agent sessions, one request per assistant turn.

PYTHONPATH=.:../lagunaLab python3 -m tools.drive [--minutes 60] [--url http://127.0.0.1:8081/v1]
Light client: the server holds the GPU.
"""
import argparse
import glob
import json
import time
import urllib.request


def turns(msgs):
    return [msgs[:i] for i, m in enumerate(msgs) if m["role"] == "assistant" and i]


def main():
    from replay.session import load_session

    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="../lagunaLab/traces")
    p.add_argument("--url", default="http://127.0.0.1:8081/v1")
    p.add_argument("--minutes", type=float, default=60)
    a = p.parse_args()
    end = time.time() + 60 * a.minutes
    while time.time() < end:
        for f in sorted(glob.glob(f"{a.traces}/*.jsonl")):
            msgs, tools = load_session(f)
            for prefix in turns(msgs):
                if time.time() >= end:
                    return
                body = {"model": "Laguna-XS-2.1", "messages": prefix, "max_tokens": 256, "tools": tools}
                req = urllib.request.Request(f"{a.url}/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=600).read()
            print(f, flush=True)


if __name__ == "__main__":
    main()
