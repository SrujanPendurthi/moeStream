#!/bin/bash
# T1 of the parity eval (docs/specs/2026-10-15-parity-eval-design.md): three small pi tasks against
# the moeStream server, then against lagunaLab's streaming server, same full pack, one server at a
# time (never two model loads at once). Writes $OUT/pi_tasks.json.
# usage: tools/pi_tasks.sh [results dir]          (default results/pi_tasks)
# Refuses to start while .gpu/lock is held. Each server runs through gpu-run and watch.sh 4 17; every
# process it starts is stopped by PID on exit. Run a copy of this script, never edit it mid-run.
# Overrides for a dry run: SERVE_MOE, SERVE_LAB (server commands), PI (the pi command), TASK_SECS,
# GPUDIR (the .gpu directory whose lock is checked).
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
OUT=${1:-results/pi_tasks}
GPUDIR=${GPUDIR:-"$REPO/../.gpu"}
GPU="$GPUDIR/gpu-run"
FULL="$REPO/../lagunaLab/models/Laguna-XS-2.1-packed"
TASK_SECS=${TASK_SECS:-1200}
PI=${PI:-pi}
if [ -d "$GPUDIR/lock" ]; then
  echo "GPU lock held ($(cat "$GPUDIR/lock/owner" 2>/dev/null)); not starting" >&2
  exit 4
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH=.:../lagunaLab PYTHONDONTWRITEBYTECODE=1
cat > "$OUT/moe.toml" <<EOF
pack_dir = "$FULL"
memory_gb = 14.0
calibration = "$REPO/budget.json"
tune = false
kv_dir = "$OUT/kvcache"
EOF
SERVE_MOE=${SERVE_MOE:-"$GPU moe tools/watch.sh 4 17 python3 -m tools.serve --config $OUT/moe.toml"}
SERVE_LAB=${SERVE_LAB:-"$GPU moe tools/watch.sh 4 17 python3 -m tools.lab_serve --pack $FULL --slots 64"}

server_pid() { lsof -tiTCP:8081 -sTCP:LISTEN 2>/dev/null | head -1; }
JOB="" PIPID=""
cleanup() {  # nothing this script started may outlive it
  [ -n "$PIPID" ] && kill "$PIPID" 2>/dev/null
  local p; p=$(server_pid); [ -n "$p" ] && kill "$p"
  for _ in $(seq 60); do [ -z "$(server_pid)" ] && break; sleep 1; done
  [ -n "$JOB" ] && kill "$JOB" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

task() {  # $1 = dir, $2 = task name: writes a tiny repo with one bug and a failing test
  rm -rf "$1"; mkdir -p "$1"; cd "$1" || exit 1; git init -q
  case $2 in
    mean)
      printf 'def mean(xs):\n    return sum(xs) / len(xs) + 1\n' > stats.py
      printf 'from stats import mean\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n' > test_stats.py ;;
    words)
      printf 'def reverse_words(s):\n    raise NotImplementedError\n' > words.py
      printf 'from words import reverse_words\n\ndef test_reverse():\n    assert reverse_words("a b c") == "c b a"\n' > test_words.py ;;
    fizz)
      printf 'def fizzbuzz(n):\n    out = []\n    for i in range(1, n):\n        out.append("Fizz" if i %% 3 == 0 else str(i))\n    return out\n' > fizz.py
      printf 'from fizz import fizzbuzz\n\ndef test_fizzbuzz():\n    assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]\n' > test_fizz.py ;;
  esac
  git add -A && git -c user.email=t@t -c user.name=t commit -qm init
  cd "$REPO" || exit 1
}

run_engine() {  # $1 = moe|lab, $2 = server command
  local log="$OUT/serve_$1.log"
  $2 > "$log" 2>&1 &
  JOB=$!
  while kill -0 "$JOB" 2>/dev/null; do  # no timeout: gpu-run may queue behind lagunaLab
    grep -q "Starting httpd" "$log" && [ -n "$(server_pid)" ] && break
    sleep 2
  done
  [ -z "$(server_pid)" ] && { echo "$1 server did not start" >&2; tail -3 "$log" >&2; return 1; }
  for t in mean words fizz; do
    local d="$OUT/work/$1/$t" s; task "$d" "$t"
    s=$(date +%s)
    (cd "$d" && $PI --provider moestream --model Laguna-XS-2.1 \
        -p "Fix the bug so the tests pass. Run python3 -m pytest to check." > "$OUT/pi_$1_$t.out" 2>&1) &
    PIPID=$!
    while kill -0 "$PIPID" 2>/dev/null && [ $(( $(date +%s) - s )) -lt "$TASK_SECS" ]; do sleep 2; done
    kill -0 "$PIPID" 2>/dev/null && { kill "$PIPID"; echo "timeout" >> "$OUT/pi_$1_$t.out"; }
    wait "$PIPID"; local rc=$?; PIPID=""
    local pass=false; (cd "$d" && python3 -m pytest -q > /dev/null 2>&1) && pass=true
    echo "{\"engine\": \"$1\", \"task\": \"$t\", \"passed\": $pass, \"pi_exit\": $rc, \"secs\": $(( $(date +%s) - s ))}" >> "$OUT/rows.jsonl"
    [ -n "$(server_pid)" ] || { echo "$1 server died during $t" >&2; break; }
  done
  local p; p=$(server_pid); [ -n "$p" ] && kill "$p"
  for _ in $(seq 60); do [ -z "$(server_pid)" ] && break; sleep 1; done
  wait "$JOB" 2>/dev/null; JOB=""
  case $2 in *gpu-run*)  # the lock is free (this load is gone) before the next server starts
    while ps -axo command | grep -q "[g]pu-run moe.*tools\.\(serve\|lab_serve\)"; do sleep 1; done ;; esac
}

rm -f "$OUT/rows.jsonl"
run_engine moe "$SERVE_MOE"
run_engine lab "$SERVE_LAB"
python3 - "$OUT" <<'EOF'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines()] if (out / "rows.jsonl").exists() else []
eng = {}
for e in ("moe", "lab"):
    log = out / f"serve_{e}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    peak = re.findall(r"watch: peak swap (\d+) MB, peak footprint (\d+) MB", text)
    eng[e] = {"passed": sum(r["passed"] for r in rows if r["engine"] == e),
              "tasks": sum(r["engine"] == e for r in rows),
              "server_errors": text.count("Traceback") + text.count(" 500 "),
              "aborted": "aborted" in text, "watch_peak_mb": [int(x) for x in peak[-1]] if peak else None}
res = {"rows": rows, "engines": eng,
       "gate": {"moe_engine_errors": eng["moe"]["server_errors"], "pass": eng["moe"]["server_errors"] == 0 and not eng["moe"]["aborted"]}}
(out / "pi_tasks.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["engines"]), json.dumps(res["gate"]))
EOF
