#!/bin/bash
# Task 18: the SSD prompt-cache tier on real weights (spec docs/specs/2026-10-14-ssd-kv-tier-design.md).
# usage: tools/kvtier_check.sh [results dir]          (default results/kvtier)
# Phases, server stopped between each; `touch $OUT/STOP` ends the run at the next phase boundary, and
# any failure (watchdog abort, drive error, server exit) stops the whole run:
#   off       3 captured sessions interleaved, 5 rounds, kv_tier = false
#   on        the same with the tier on
#   restart   fresh server, one more round: every session must restore from SSD
#   kill9     a turn whose snapshot write is killed with kill -9; restart; the next turn must restore
#   equality  one more round, each turn followed by one greedy probe (RAM hit); restart; the same probe
#             again (SSD restore); the two must match
#   disk      directory size and bytes per turn
# context_tokens 24576 so the RAM prompt cache holds about one of these sessions, as 128k holds one
# 75k session in real use. Thinking off so replies finish and the history re-renders exactly.
# $OUT/mem.log: the server's MLX memory per request (MOESTREAM_MEMLOG), top's MEM every 2 s, phase marks.
# SERVE overrides the server command (dry run against a stub); WATCH the watchdog caps (default "4 16");
# MEM the memory_gb (default 14.0).
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
OUT=${1:-results/kvtier}
WATCH=${WATCH:-"4 16"}
SERVE=${SERVE:-"/Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe tools/watch.sh $WATCH python3 -m tools.serve"}
PAUSE=${PAUSE:-5}
MEM=${MEM:-14.0}
SESSIONS=astropy__astropy-12907.jsonl,django__django-10924.jsonl,astropy__astropy-6938.jsonl
KV=$REPO/kvcache
mkdir -p "$OUT"
export PYTHONPATH=.:../lagunaLab MOESTREAM_SHARED=1 MOESTREAM_MEMLOG=$OUT/mem.log

for tier in false true; do
  cat > "$OUT/kvtier_$tier.toml" <<EOF
pack_dir = "$REPO/../lagunaLab/models/Laguna-XS-2.1-packed"
memory_gb = $MEM
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
echo "{\"event\": \"run\", \"t\": $(date +%s), \"memory_gb\": $MEM, \"prefill_step_size\": 2048, \"watch\": \"$WATCH\"}" >> "$OUT/mem.log"

say() { echo "[$(date +%T)] $*" | tee -a "$OUT/phases.log"; }
mark() { echo "{\"event\": \"phase\", \"t\": $(date +%s), \"name\": \"$1\"}" >> "$OUT/mem.log"; }
server_pid() { lsof -tiTCP:8081 -sTCP:LISTEN 2>/dev/null | head -1; }
orphans() { ps -axo pid,command | grep -E "[Pp]ython3? -m tools\.(serve|drive)|[w]atch\.sh .*tools\.serve|[g]pu-run moe|[r]esource_tracker" | grep -v -e "zsh -c" -e kvtier_check || true; }
mb() { awk '{v=$1; u=substr(v,length(v)); n=v+0; if(u=="G")n*=1024; else if(u=="K")n/=1024; else if(u=="B")n/=1048576; print int(n)}'; }

SAMPLER=""
sample() {  # top's MEM of the serving process every 2 s, into mem.log
  while true; do
    p=$(server_pid)
    [ -n "$p" ] && echo "{\"event\": \"top\", \"t\": $(date +%s), \"mem_mb\": $(top -l 1 -pid "$p" -stats mem | tail -1 | tr -d '+-' | mb)}" >> "$OUT/mem.log"
    sleep 2
  done
}
sample & SAMPLER=$!

finish() {  # $1 = exit code; nothing may be left running
  local pid; pid=$(server_pid)
  [ -n "$pid" ] && kill "$pid"
  for _ in $(seq 60); do [ -z "$(server_pid)" ] && break; sleep 1; done
  while ps -axo command | grep -q "[g]pu-run moe"; do sleep 1; done  # a queued or loading start ends too
  kill $SAMPLER 2>/dev/null; wait 2>/dev/null
  say "left running (want none):"; orphans | tee -a "$OUT/phases.log"
  exit "$1"
}
fail() { say "FAILED: $*; stopping the run"; finish 5; }
boundary() { [ -e "$OUT/STOP" ] && { say "stopped on request before $1"; finish 0; }; mark "$1"; }

start() {  # $1 = tier true|false, $2 = log
  $SERVE --config "$OUT/kvtier_$1.toml" > "$2" 2>&1 &
  local job=$!
  while kill -0 $job 2>/dev/null; do  # no timeout: gpu-run may queue behind lagunaLab
    grep -q "Starting httpd" "$2" && grep -q "^moeStream:" "$2" && [ -n "$(server_pid)" ] && return 0
    sleep 2
  done
  tail -3 "$2" | tee -a "$OUT/phases.log"
  fail "server exited before serving ($2)"
}

stop() {  # $1 = log
  local pid; pid=$(server_pid)
  [ -n "$pid" ] && kill "$pid"
  for _ in $(seq 60); do [ -z "$(server_pid)" ] && break; sleep 1; done
  while ps -axo command | grep -q "[g]pu-run moe"; do sleep 1; done
  grep "watch: peak\|aborted" "$1" | tail -2 | tee -a "$OUT/phases.log"
  grep -q "aborted" "$1" && fail "watchdog ($1)"
  return 0
}

alive() {  # $1 = log: the server must still be up after a drive
  [ -n "$(server_pid)" ] || { stop "$1"; fail "server gone ($1)"; }
}

drive() {  # $1 = serve log, rest = drive arguments
  local log=$1; shift
  python3 -m tools.drive --pause "$PAUSE" --log "$OUT/requests.jsonl" "$@" || { local rc=$?; stop "$log"; fail "drive exit $rc ($log)"; }
  alive "$log"
}
ALL=(--sessions "$SESSIONS")

boundary off
say "phase off: start"
start false "$OUT/serve_off.log"
drive "$OUT/serve_off.log" "${ALL[@]}" --live "$OUT/state_off.json" --rounds 5 --tag off
say "phase off: done"; stop "$OUT/serve_off.log"

boundary on
rm -rf "$KV"
say "phase on: start"
start true "$OUT/serve_on.log"
drive "$OUT/serve_on.log" "${ALL[@]}" --live "$OUT/state_on.json" --rounds 5 --tag on
say "phase on: done ($(grep -c 'kv tier: restored' "$OUT/serve_on.log") restores)"; stop "$OUT/serve_on.log"

boundary restart
say "phase restart: start"
start true "$OUT/serve_restart.log"
drive "$OUT/serve_restart.log" "${ALL[@]}" --live "$OUT/state_on.json" --rounds 1 --tag restart
grep "kv tier" "$OUT/serve_restart.log" | tee -a "$OUT/phases.log"
say "phase restart: done ($(grep -c 'kv tier: restored' "$OUT/serve_restart.log") of 3 sessions restored)"; stop "$OUT/serve_restart.log"

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
while ps -axo command | grep -q "[g]pu-run moe"; do sleep 1; done
say "left after kill -9 (want none):"; orphans | tee -a "$OUT/phases.log"
start true "$OUT/serve_kill9_after.log"
grep "kv tier" "$OUT/serve_kill9_after.log" | tee -a "$OUT/phases.log"
drive "$OUT/serve_kill9_after.log" --sessions "$first" --live "$OUT/state_on.json" --rounds 1 --tag kill9_after
grep "kv tier" "$OUT/serve_kill9_after.log" | tee -a "$OUT/phases.log"
say "phase kill9: done"; stop "$OUT/serve_kill9_after.log"

boundary equality
say "phase equality: start"
start true "$OUT/serve_equality.log"
drive "$OUT/serve_equality.log" "${ALL[@]}" --live "$OUT/state_on.json" --rounds 1 --tag equality --probe-after "$OUT/probes_ram.json"
stop "$OUT/serve_equality.log"
start true "$OUT/serve_equality_ssd.log"
drive "$OUT/serve_equality_ssd.log" "${ALL[@]}" --live "$OUT/state_on.json" --rounds 0 --tag equality_ssd --probe "$OUT/probes_ssd.json"
grep "kv tier" "$OUT/serve_equality_ssd.log" | tee -a "$OUT/phases.log"
python3 - "$OUT/probes_ram.json" "$OUT/probes_ssd.json" <<'EOF' | tee -a "$OUT/phases.log"
import json, sys
ram, ssd = (json.load(open(f)) for f in sys.argv[1:])
for s in ram:
    print(f"{s.split('/')[-1]}: {'MATCH' if ram[s][0] == ssd.get(s, [None])[0] else 'DIFFER'}")
EOF
say "phase equality: done"; stop "$OUT/serve_equality_ssd.log"

boundary disk
say "phase disk: $(du -sh "$KV" | cut -f1) in $KV"
ls "$KV" | grep -v FORMAT | while read -r c; do
  python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(sys.argv[2], 'turns', m['turns'], 'bytes', sum(e['bytes'] for e in m['segments']+m['snapshots']))" "$KV/$c/manifest.json" "$c"
done | tee -a "$OUT/phases.log"
say "all phases done"
finish 0
