#!/usr/bin/env bash
# wind_gz_30sep.sh -- the Gazebo ladder for the reversal-priced decider (Sai 30 Sept: "try the fixes and re-fly it in Gazebo").
# Waits for the training queue (PID below) to exit, picks the stage2_c100h100f02g seed with the best validation mean (the
# checkpoint's own meta), then waits for the ACRA session's GNSS-free flights (queue/geo_calm10.done) and an idle GPU, and flies
# the same 20 yards x 4 winds as the rule, stage2_c100_s0 and stage2_c100h100_s1 (gazebo_wind/ladder_rule), one SITL.
R=/home/sai/RL_projects/rl-landing; cd $R; LOG=$R/queue/wind_flip.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
WAIT_PID=${WAIT_PID:-2048533}
while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
BEST=$($PY - <<'PYEOF'
import json, numpy as np, glob
best = None
for f in sorted(glob.glob("wind/runs/stage2_c100h100f02g_s[0-9]_best.npz")):
    v = float(np.mean(json.loads(str(np.load(f)["meta"]))["val"]))
    print(f"{f} val mean {v:.4f}", file=open("queue/wind_flip.log", "a"))
    if best is None or v > best[1]: best = (f, v)
print(best[0])
PYEOF
)
echo "$(date +%H:%M) gz ladder: chose $BEST; waiting for queue/geo_calm10.done" >> $LOG
until [ -f queue/geo_calm10.done ]; do sleep 60; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "$(date +%H:%M) gz ladder start: $BEST" >> $LOG
EPISODES="$(seq 0 19)" WINDS="9 12 15 18" DECIDERS="$BEST" N=1 OUT=ladder_rule ./wind/gz_ladder.sh
echo "$(date +%H:%M) gz ladder done" >> $LOG
touch queue/wind_gz_30sep.done
