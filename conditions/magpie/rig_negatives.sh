#!/usr/bin/env bash
# conditions/magpie/rig_negatives.sh -- hard-negative mining: 200 places over the landing pads, no magpie, into the
# mast rig's training set; retrain, re-score (0.7), recalibrate, install, restart the live comparison.
#   ./conditions/magpie/rig_negatives.sh BENCH_PID
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue; cd $R
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
source /opt/ros/jazzy/setup.bash 2>/dev/null
until [ "$(GZ_PARTITION=rig timeout 10 gz topic -l 2>/dev/null | grep -c '^/rig/')" -ge 6 ]; do sleep 5; done
log "negatives: rendering"
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --negatives shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/negatives.log 2>&1 || { log "FAILED negatives"; exit 1; }
kill $1 && log "bench stopped"
sleep 10
$P conditions/magpie/rig_detect.py --train shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/data2_train.log 2>&1 || { log "FAILED detector (negatives)"; exit 1; }
$P conditions/magpie/rig_detect.py --score shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/data2_score.log 2>&1 || { log "FAILED score (negatives)"; exit 1; }
/usr/bin/python3 shared/px4_gazebo/gz_rig.py --calib shared/px4_gazebo/gz_logan/rig/cover2 shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/calib2.log 2>&1 || { log "FAILED calib (negatives)"; exit 1; }
/usr/bin/python3 - <<'PY'
import json
c = json.load(open("shared/fast_sim/rig_calib.json"))
c["p_detect"]["size_px"] = [2 * s for s in c["p_detect"]["size_px"]]
c["source"]["variant"] = "half detection range: each detection chance needs twice the pixels"
json.dump(c, open("shared/fast_sim/rig_calib_half.json", "w"), indent=1)
PY
cp shared/px4_gazebo/gz_logan/rig/data2/det.pt shared/px4_gazebo/gz_logan/rig/det.pt
log "negatives: detector + calibration done ($(python3 -c "import json; c=json.load(open('shared/fast_sim/rig_calib.json')); print('false alarms per picture', round(c['source']['false_alarms_per_image'], 4))"))"
rm -rf outputs/gazebo_rig/planner_rig outputs/gazebo_rig/planner_omni
setsid nohup ./conditions/magpie/gz_compare.sh > outputs/queue/gz_compare.out 2>&1 < /dev/null &
log "gz_compare restarted"
