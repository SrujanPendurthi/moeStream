#!/bin/bash
# Task 18: the SSD prompt-cache tier on real weights (spec docs/specs/2026-10-14-ssd-kv-tier-design.md).
# usage: tools/kvtier_check.sh [results dir]          (default results/kvtier)
# Phases, server stopped between each; `touch $OUT/STOP` ends the run at the next phase boundary:
#   off      3 captured sessions interleaved, 5 rounds, kv_tier = false
#   on       the same with the tier on; the last round probes each session greedily twice (RAM hits)
#   restart  fresh server: one greedy probe per session (SSD restores), compared with the RAM probes
#   kill9    a turn whose snapshot write is killed with kill -9; restart; the next turn must restore
#   disk     directory size and bytes per turn
# context_tokens 24576 so the RAM prompt cache holds about one of these sessions, as 128k holds one
# 75k session in real use. Thinking off so replies finish and the history re-renders exactly.
# SERVE overrides the server command (dry run against a stub); WATCH the watchdog caps (default "4 16").
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
OUT=${1:-results/kvtier}
WATCH=${WATCH:-"4 16"}
SERVE=${SERVE:-"/Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe tools/watch.sh $WATCH python3 -m tools.serve"}
PAUSE=${PAUSE:-5}
SESSIONS=astropy__astropy-12907.jsonl,django__django-10924.jsonl,astropy__astropy-6938.jsonl
KV=$REPO/kvcache
mkdir -p "$OUT"
export PYTHONPATH=.:../lagunaLab MOESTREAM_SHARED=1

for tier in false true; do
  cat > "$OUT/kvtier_$tier.toml" <<EOF
pack_dir = "$REPO/../lagunaLab/models/Laguna-XS-2.1-packed"
memory_gb = 14.0
context_tokens = 24576
kv_bits = 8
calibration = "$REPO/budget.json"
thinking = false
tune = false
kv_tier = $tier
kv_dir = "$KV"
kv_disk_gb = 10.0
EOF
done

say() { echo "[$(date +%T)] $*" | tee -a "$OUT/phases.log"; }
server_pid() { lsof -tiTCP:8081 -sTCP:LISTEN 2>/dev/null | head -1; }
boundary() { [ -e "$OUT/STOP" ] && { say "stopped on request before $1"; exit 0; }; }

start() {  # $1 = tier true|false, $2 = log
  $SERVE --config "$OUT/kvtier_$1.toml" > "$2" 2>&1 &
  local job=$!
  while kill -0 $job 2>/dev/null; do  # no timeout: gpu-run may queue behind lagunaLab
    grep -q "Starting httpd" "$2" && grep -q "^moeStream:" "$2" && [ -n "$(server_pid)" ] && return 0
    sleep 2
  done
  grep -q "aborted: paging" "$2" && say "aborted: paging ($2)" || say "server exited before serving ($2)"
  tail -3 "$2" | tee -a "$OUT/phases.log"; exit 4
}

stop() {  # $1 = log
  local pid; pid=$(server_pid)
  [ -n "$pid" ] && kill "$pid"
  for _ in $(seq 60); do [ -z "$(server_pid)" ] && break; sleep 1; done
  wait 2>/dev/null
  grep "watch: peak\|aborted" "$1" | tail -2 | tee -a "$OUT/phases.log"
}

orphans() { ps -axo pid,command | grep -E "[Pp]ython3? -m tools\.(serve|drive)|[w]atch\.sh .*tools\.serve|[g]pu-run moe" || true; }

drive() { python3 -m tools.drive --sessions "$SESSIONS" --pause "$PAUSE" --log "$OUT/requests.jsonl" "$@"; }

boundary off
say "phase off: start"
start false "$OUT/serve_off.log"
drive --live "$OUT/state_off.json" --rounds 5 --tag off
say "phase off: done"; stop "$OUT/serve_off.log"

boundary on
rm -rf "$KV"
say "phase on: start"
start true "$OUT/serve_on.log"
drive --live "$OUT/state_on.json" --rounds 4 --tag on
drive --live "$OUT/state_on.json" --rounds 1 --tag on --probe-after "$OUT/probes_ram.json"
say "phase on: done"; stop "$OUT/serve_on.log"

boundary restart
say "phase restart: start"
start true "$OUT/serve_restart.log"
drive --live "$OUT/state_on.json" --rounds 0 --tag restart --probe "$OUT/probes_ssd.json"
grep "kv tier" "$OUT/serve_restart.log" | tee -a "$OUT/phases.log"
say "phase restart: done"; stop "$OUT/serve_restart.log"

boundary kill9
say "phase kill9: start"
start true "$OUT/serve_kill9.log"
first=${SESSIONS%%,*}
python3 -m tools.drive --sessions "$first" --pause 0 --log "$OUT/requests.jsonl" --live "$OUT/state_on.json" --rounds 1 --tag kill9 &
dpid=$!
caught=""
for _ in $(seq 60000); do  # 10 ms polls; the write of one turn's snapshot takes about a tenth of a second
  if ls "$KV"/*/*.tmp > /dev/null 2>&1; then caught=$(ls "$KV"/*/*.tmp | head -1); break; fi
  if ! kill -0 $dpid 2>/dev/null; then  # reply sent: the snapshot write starts right after
    for _ in $(seq 3000); do ls "$KV"/*/*.tmp > /dev/null 2>&1 && { caught=$(ls "$KV"/*/*.tmp | head -1); break 2; }; sleep 0.01; done
    break
  fi
  sleep 0.01
done
pid=$(server_pid)
[ -n "$pid" ] && kill -9 "$pid"
say "kill -9 server $pid; temp file caught mid-write: ${caught:-none (write finished first)}"
wait $dpid 2>/dev/null
for _ in $(seq 30); do [ -z "$(server_pid)" ] && break; sleep 1; done
sleep 3; wait 2>/dev/null
say "left after kill -9 (want none):"; orphans | tee -a "$OUT/phases.log"
start true "$OUT/serve_kill9_after.log"
grep "kv tier" "$OUT/serve_kill9_after.log" | tee -a "$OUT/phases.log"
python3 -m tools.drive --sessions "$first" --pause 0 --log "$OUT/requests.jsonl" --live "$OUT/state_on.json" --rounds 1 --tag kill9_after
grep "kv tier" "$OUT/serve_kill9_after.log" | tee -a "$OUT/phases.log"
say "phase kill9: done"; stop "$OUT/serve_kill9_after.log"

say "phase disk: $(du -sh "$KV" | cut -f1) in $KV"
ls "$KV" | grep -v FORMAT | while read -r c; do
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[2], 'turns', m['turns'], 'bytes', sum(e['bytes'] for e in m['segments']+m['snapshots']))" "$KV/$c/manifest.json" "$c"
done | tee -a "$OUT/phases.log"
say "left at the end (want none):"; orphans | tee -a "$OUT/phases.log"
say "all phases done"
