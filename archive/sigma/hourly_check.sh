#!/bin/bash
# Hourly health check on the outdoor campaign. Reports the NUMBERS, not just
# that processes are alive -- a run where every arm sits at 98% looks perfectly
# healthy by process count and is already worthless.
cd /home/sai/RL_projects/rl-landing/archive/sigma   # these drivers run from their own folder
while true; do
  A=$(ls out_sweep/*/ablation.json 2>/dev/null | wc -l)
  B=$(ls out_final/*/ablation.json 2>/dev/null | wc -l)
  J=$(ps -eo comm= | grep -c '^python$')
  R=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)
  D=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  C=$(grep -lE 'Traceback|Killed' out_sweep/*.log out_final/*.log 2>/dev/null | wc -l)
  STAT=$(/home/sai/miniconda3/envs/mujoco_drones/bin/python - <<'PY' 2>/dev/null
import json,glob,numpy as np
msgs=[]
for tag,pat in (("sweep","out_sweep/{a}_c*/ablation.json"),("final","out_final/{a}_s*/ablation.json")):
    vals={}
    for a in ("none","staleness","sigma","random"):
        v=[json.load(open(p))["results"][a]["success_%"] for p in sorted(glob.glob(pat.format(a=a)))]
        if v: vals[a]=v
    if not vals: continue
    allv=[x for v in vals.values() for x in v]
    spread=max(allv)-min(allv); best=max(allv)
    flags=[]
    if spread<=1.0: flags.append("DEAD-POLICY(spread %.2f)"%spread)
    if best>95:     flags.append("SATURATED(%.1f%%)"%best)
    msgs.append("%s n=%d best=%.1f%% spread=%.1f%s"%(tag,len(allv),best,spread," !! "+",".join(flags) if flags else ""))
print(" | ".join(msgs) if msgs else "no results yet")
PY
)
  FLAG=""
  [ "$C" -gt 0 ] && FLAG="$FLAG CRASHES=$C"
  [ "$R" -lt 3500 ] && FLAG="$FLAG LOW-RAM"
  echo "[$(date '+%H:%M')] sweep $A/24 final $B/48 | jobs $J | RAM ${R}MB disk ${D}G | $STAT$FLAG"
  if [ "$B" -ge 48 ]; then echo "[$(date '+%H:%M')] CAMPAIGN COMPLETE"; break; fi
  if ! pgrep -f out_run.sh >/dev/null && [ "$B" -lt 48 ]; then
    echo "[$(date '+%H:%M')] DRIVER GONE with only $B/48 done -- investigate"; break; fi
  sleep 3600
done
