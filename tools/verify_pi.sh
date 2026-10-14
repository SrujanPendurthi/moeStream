#!/bin/bash
# Engine plan Task 6: serve on real weights, one curl, a 3-turn pi session, then one run per setting change.
# usage: tools/verify_pi.sh            (from the repo root; tell the coordinator first)
# Every server runs through the GPU lock and the swap watchdog. Tuning is off here (tune = false):
# this checks the engine, the self-tuning run is its own plan's Task 10.
# The pi turns run only if ~/.pi/agent/models.json already has the "moestream" provider (the user adds it).
set -u
cd "$(dirname "$0")/.."
GPU=/Users/srujanpendurthi/Projects/Current/.gpu/gpu-run
LOG=results/serve_verify.log
URL=http://127.0.0.1:8081/v1

config() {  # $1 = extra TOML lines; writes results/verify.toml from moestream.toml, absolute paths, tuning off
  python3 - "$1" <<'EOF'
import sys, tomllib
from pathlib import Path
c = tomllib.loads(Path("moestream.toml").read_text())
for k in ("pack_dir", "calibration", "tuning_dir"):
    c[k] = str((Path("moestream.toml").parent / c[k]).resolve())
c["tune"] = False
c.update(tomllib.loads(sys.argv[1]))
out = "\n".join(f"{k} = {str(v).lower() if isinstance(v, bool) else repr(v) if isinstance(v, str) else v}" for k, v in c.items())
Path("results/verify.toml").write_text(out.replace("'", '"') + "\n")
EOF
}

serve() {  # $1 = tag; starts the server in the background, waits until it listens
  echo "=== $1 $(date +%T)" >> "$LOG"
  PYTHONPATH=.:../lagunaLab "$GPU" moe tools/watch.sh ${WATCH:-8 21} python3 -m tools.serve --config results/verify.toml >> "$LOG" 2>&1 &
  for _ in $(seq 900); do
    grep -q "aborted: paging" "$LOG" && { echo "$1: aborted: paging"; exit 3; }
    tail -5 "$LOG" | grep -q "Starting httpd" && return 0
    sleep 1
  done
  echo "$1: server did not start in 15 min"; exit 1
}

stop() {  # kill the server by PID (never pkill -f), then let the watchdog print its peaks
  # the Python process itself, not watch.sh or gpu-run (their command lines contain the same text)
  pid=$(ps -axo pid,command | grep -E "[Pp]ython3? -m tools.serve --config results/verify.toml" | grep -v -e watch.sh -e gpu-run | awk '{print $1}')
  [ -n "$pid" ] && kill $pid
  for _ in $(seq 60); do lsof -nP -iTCP:8081 -sTCP:LISTEN > /dev/null || break; sleep 1; done
  lsof -nP -iTCP:8081 -sTCP:LISTEN > /dev/null && { echo "server still listening on 8081 after 60 s"; exit 1; }
  sleep 3
  grep "watch: peak" "$LOG" | tail -1
}

ask() {  # $1 = tag
  local t0=$(date +%s)
  curl -s $URL/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"Laguna-XS-2.1","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"max_tokens":512}' \
    > results/ask.json
  python3 -c "import json,sys; r=json.load(open('results/ask.json')); m=r['choices'][0]['message']; print('$1', 'reasoning:', bool(m.get('reasoning')), 'usage:', r.get('usage'), 'secs:', $(date +%s)-$t0)" | tee -a "$LOG"
}

pi_session() {
  grep -q '"moestream"' ~/.pi/agent/models.json 2>/dev/null || { echo "pi: no moestream provider yet, skipped" | tee -a "$LOG"; return; }
  W=$(mktemp -d)
  ( cd "$W" && git init -q
    printf 'def mean(xs):\n    return sum(xs) / len(xs) + 1\n' > stats.py
    printf 'from stats import mean\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n' > test_stats.py
    for p in "Run the tests in this repo with python3 -m pytest and tell me what fails." \
             "Fix the bug so the test passes, then run the tests again." \
             "Add a test for an empty list and make mean raise ValueError for it."; do
      c=$([ "$p" = "Run the tests in this repo with python3 -m pytest and tell me what fails." ] || echo -c)
      s=$(date +%s); pi --provider moestream --model Laguna-XS-2.1 $c -p "$p" > /dev/null 2>&1
      echo "pi turn: $(( $(date +%s) - s )) s" | tee -a "$OLDPWD/$LOG"
    done
    python3 -m pytest -q 2>&1 | tail -1 | sed 's/^/pi final pytest: /' | tee -a "$OLDPWD/$LOG" )
}

config ""; serve base; ask base; pi_session; stop
for v in 'eviction = "lru"' 'kv_bits = 0' 'thinking = false'; do
  config "$v"; serve "$v"; ask "$v"; stop
done
grep -E "^moeStream:|=== |reasoning:|pi |watch: peak|aborted" "$LOG"
