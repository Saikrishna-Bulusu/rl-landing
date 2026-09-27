#!/usr/bin/env bash
# yolo_flights.sh -- once YOLO is scored (after_yolo.sh): pick its operating threshold -- at least the small CNN's
# 86.5% found, fewest false alarms per picture -- re-score at it, then fly the three live magpie flights again
# (planner, rig, seeds 0-2) with YOLO in the loop, and replay both detectors' logs through the tracker (phantoms).
R=/home/sai/RL_projects/rl-landing; Q=$R/queue; P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
until grep -q "yolo scored\|FAILED yolo score" $Q/rig.log; do sleep 30; done
grep -q "FAILED yolo score" $Q/rig.log && exit 1
TH=$(python3 - <<'PY'
import json
s = json.load(open("/home/sai/RL_projects/rl-landing/gz_logan/rig/data2/score_yolo.json"))["by_threshold"]
ok = [(v["false_alarms_per_image"], float(t)) for t, v in s.items() if v["found"] >= 0.865]
print(min(ok)[1] if ok else max((v["found"], float(t)) for t, v in s.items())[1])
PY
)
log "yolo operating threshold $TH"
cd $R && PYTHONPATH=.yolo_pkgs YOLO_AUTOINSTALL=false $P rig_yolo.py --score gz_logan/rig/data2 --op $TH > gz_logan/rig/yolo/score_op.log 2>&1
cd $R && RIG_YOLO=$R/gz_logan/rig/yolo/run/weights/best.pt RIG_THRESH=$TH OUT=planner_rig_yolo SEEDS="0 1 2" DECIDERS=planner SENSE=rig ./rigfly.sh
cd $R/urban && $P rig_replay.py ../gazebo_rig/planner_rig/*_rig.jsonl > ../gazebo_rig/phantoms_cnn.txt 2>&1
$P rig_replay.py ../gazebo_rig/planner_rig_yolo/*_rig.jsonl > ../gazebo_rig/phantoms_yolo.txt 2>&1
log "yolo flights + phantom replay done"
