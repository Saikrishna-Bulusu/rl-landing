#!/usr/bin/env bash
# conditions/magpie/rig_detector_run.sh -- after the bench has made the pictures: stop the bench (GPU memory), train
# the detector, score it on held-out drone places, write the calibration the fast simulator flies
# with (shared/fast_sim/rig_calib.json) and its pessimistic twin (rig_calib_half.json: a detector that needs
# twice the pixels for the same chance, i.e. half the detection range).
#   ./conditions/magpie/rig_detector_run.sh DATA_PID BENCH_PID
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
cd $R
while kill -0 $1 2>/dev/null; do sleep 20; done
log "pictures done: $(grep -c . shared/px4_gazebo/gz_logan/rig/data/labels.jsonl) images"
kill $2 2>/dev/null && log "bench stopped (pid $2)"
sleep 10
log "start detector training"
$P conditions/magpie/rig_detect.py --train shared/px4_gazebo/gz_logan/rig/data > shared/px4_gazebo/gz_logan/rig/data/train.log 2>&1 || { log "FAILED detector training"; exit 1; }
log "start detector score"
$P conditions/magpie/rig_detect.py --score shared/px4_gazebo/gz_logan/rig/data > shared/px4_gazebo/gz_logan/rig/data/score.log 2>&1 || { log "FAILED detector score"; exit 1; }
/usr/bin/python3 shared/px4_gazebo/gz_rig.py --calib shared/px4_gazebo/gz_logan/rig/cover1 shared/px4_gazebo/gz_logan/rig/data > shared/px4_gazebo/gz_logan/rig/calib.log 2>&1 || { log "FAILED calib"; exit 1; }
/usr/bin/python3 - <<'EOF' || { log "FAILED half calib"; exit 1; }
import json
c = json.load(open("shared/fast_sim/rig_calib.json"))
c["p_detect"]["size_px"] = [2 * s for s in c["p_detect"]["size_px"]]
c["source"]["variant"] = "half detection range: each detection chance needs twice the pixels"
json.dump(c, open("shared/fast_sim/rig_calib_half.json", "w"), indent=1)
EOF
log "detector + calibration done"
