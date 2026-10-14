#!/bin/bash
# usage: tools/watch.sh <max_swap_gb> <max_footprint_gb> <cmd...>
# Runs cmd, polls every 2 s, kills it if swap or its footprint (top MEM, counts Metal buffers) exceeds the cap.
# Prints peak swap and footprint at exit; exit 3 and "aborted: paging" when it trips.
max_swap=$1 max_fp=$2; shift 2
mb() { awk '{v=$1; u=substr(v,length(v)); n=v+0; if(u=="G")n*=1024; else if(u=="K")n/=1024; else if(u=="B")n/=1048576; print int(n)}'; }
"$@" & pid=$!
peak_swap=0 peak_fp=0
while kill -0 "$pid" 2>/dev/null; do
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1)
  fp=$(top -l 1 -pid "$pid" -stats mem 2>/dev/null | tail -1 | tr -d '+-' | mb)
  [ "${fp:-0}" -gt "$peak_fp" ] && peak_fp=$fp
  [ "$swap" -gt "$peak_swap" ] && peak_swap=$swap
  if [ "$swap" -gt $((max_swap * 1024)) ] || [ "${fp:-0}" -gt $((max_fp * 1024)) ]; then
    kill "$pid"; sleep 2; kill -9 "$pid" 2>/dev/null
    echo "aborted: paging (swap ${swap} MB, footprint ${fp} MB; caps ${max_swap} GB / ${max_fp} GB)"
    echo "watch: peak swap ${peak_swap} MB, peak footprint ${peak_fp} MB"
    exit 3
  fi
  sleep 2
done
wait "$pid"; rc=$?
echo "watch: peak swap ${peak_swap} MB, peak footprint ${peak_fp} MB"
exit $rc
