#!/bin/bash
# Long context on real weights: lagunaLab's longest captured session (django__django-11039) started
# four turns from its end, so the first request is one ~81k-token prompt (thinking on renders the
# recorded reasoning), then two more turns; then a restart and the last turn, which must restore
# the conversation from the SSD tier.
# usage: tools/longctx_check.sh [results dir]           (default results/longctx)
# $OUT/mem.log: MLX memory and cumulative expert hits/misses per request; serve logs carry
# mlx-lm's timestamped prompt-progress lines (prefill time). Any failure stops the whole run.
# SERVE overrides the server command (dry run against a stub); WATCH the watchdog caps (default "4 17").
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
OUT=${1:-results/longctx}
WATCH=${WATCH:-"4 17"}
SERVE=${SERVE:-"$REPO/../.gpu/gpu-run moe tools/watch.sh $WATCH python3 -m tools.serve"}
SESSION=django__django-11039.jsonl
mkdir -p "$OUT"
KV=$(cd "$OUT" && pwd)/kvcache
export PYTHONPATH=.:../lagunaLab MOESTREAM_SHARED=1 MOESTREAM_MEMLOG=$OUT/mem.log
cat > "$OUT/longctx.toml" <<EOF
pack_dir = "$REPO/../lagunaLab/models/Laguna-XS-2.1-packed"
memory_gb = 14.0
context_tokens = 131072
kv_bits = 8
calibration = "$REPO/budget.json"
thinking = true
tune = false
kv_tier = true
kv_dir = "$KV"
kv_disk_gb = 10.0
mlx_cache_gb = 1.0
EOF

say() { echo "[$(date +%T)] $*" | tee -a "$OUT/phases.log"; }
server_pid() { lsof -tiTCP:8081 -sTCP:LISTEN 2>/dev/null | head -1; }
orphans() { ps -axo pid,command | grep -E "[Pp]ython3? -m tools\.(serve|drive)|[w]atch\.sh .*tools\.serve|[g]pu-run moe" | grep -v -e "zsh -c" -e longctx_check || true; }

stop() {  # $1 = log
  local pid; pid=$(server_pid)
  [ -n "$pid" ] && kill "$pid"
  for _ in $(seq 60); do [ -z "$(server_pid)" ] && break; sleep 1; done
  case $SERVE in *gpu-run*) while ps -axo command | grep -q "[g]pu-run moe"; do sleep 1; done ;; esac
  grep "watch: peak\|aborted" "$1" | tail -2 | tee -a "$OUT/phases.log"
  grep -q "aborted" "$1" && fail "watchdog ($1)"
  return 0
}
fail() { say "FAILED: $*; stopping the run"; local pid; pid=$(server_pid); [ -n "$pid" ] && kill "$pid"
         sleep 5; say "left running (want none):"; orphans | tee -a "$OUT/phases.log"; exit 5; }

start() {  # $1 = log
  $SERVE --config "$OUT/longctx.toml" > "$1" 2>&1 &
  local job=$!
  while kill -0 $job 2>/dev/null; do  # no timeout: gpu-run may queue behind lagunaLab
    grep -q "Starting httpd" "$1" && grep -q "^moeStream:" "$1" && [ -n "$(server_pid)" ] && return 0
    sleep 2
  done
  tail -3 "$1" | tee -a "$OUT/phases.log"
  fail "server exited before serving ($1)"
}

drive() {  # $1 = serve log, rest = drive arguments
  local log=$1; shift
  python3 -m tools.drive --sessions "$SESSION" --live "$OUT/state.json" --start-back 4 --max-tokens 4096 \
    --pause 5 --log "$OUT/requests.jsonl" "$@" || { local rc=$?; stop "$log"; fail "drive exit $rc ($log)"; }
  [ -n "$(server_pid)" ] || { stop "$log"; fail "server gone ($log)"; }
}

rm -rf "$KV"
say "phase long: start"
start "$OUT/serve_long.log"
drive "$OUT/serve_long.log" --rounds 3 --tag long
say "phase long: done"; stop "$OUT/serve_long.log"

say "phase restart: start"
start "$OUT/serve_restart.log"
drive "$OUT/serve_restart.log" --rounds 1 --tag restart
grep "kv tier" "$OUT/serve_restart.log" | tee -a "$OUT/phases.log"
say "phase restart: done ($(grep -c 'kv tier: restored' "$OUT/serve_restart.log") restores)"; stop "$OUT/serve_restart.log"
say "disk: $(du -sh "$KV" | cut -f1)"
say "left running (want none):"; orphans | tee -a "$OUT/phases.log"
