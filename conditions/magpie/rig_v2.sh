#!/usr/bin/env bash
# conditions/magpie/rig_v2.sh -- the bench steps again for the mast rig (cameras above the propellers, below the legs):
# coverage + views, training pictures, detector, score, calibration (+ half-range twin). Then the new
# detector is installed at shared/px4_gazebo/gz_logan/rig/det.pt (what conditions/magpie/rig_perceive.py loads) and
# outputs/queue/rig_calib2.done lets conditions/magpie/rig_queue.sh start its ladders.   ./conditions/magpie/rig_v2.sh BENCH_PID
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue; cd $R
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
BENCH=$1
source /opt/ros/jazzy/setup.bash 2>/dev/null
until [ "$(GZ_PARTITION=rig timeout 10 gz topic -l 2>/dev/null | grep -c '^/rig/')" -ge 6 ]; do sleep 5; done
log "mast rig: coverage"
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --cover shared/px4_gazebo/gz_logan/rig/cover2 > shared/px4_gazebo/gz_logan/rig/cover2.log 2>&1 || { log "FAILED mast coverage"; exit 1; }
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --report shared/px4_gazebo/gz_logan/rig/cover2 >> shared/px4_gazebo/gz_logan/rig/cover2.log 2>&1
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --views shared/px4_gazebo/gz_logan/rig/cover2 >> shared/px4_gazebo/gz_logan/rig/cover2.log 2>&1
log "mast rig: training pictures"
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --data shared/px4_gazebo/gz_logan/rig/data2 --poses 600 > shared/px4_gazebo/gz_logan/rig/data2.log 2>&1 || { log "FAILED mast data"; exit 1; }
kill $BENCH && log "bench stopped"
sleep 10
$P conditions/magpie/rig_detect.py --train shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/data2_train.log 2>&1 || { log "FAILED mast detector"; exit 1; }
$P conditions/magpie/rig_detect.py --score shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/data2_score.log 2>&1 || { log "FAILED mast score"; exit 1; }
/usr/bin/python3 shared/px4_gazebo/gz_rig.py --calib shared/px4_gazebo/gz_logan/rig/cover2 shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/calib2.log 2>&1 || { log "FAILED mast calib"; exit 1; }
/usr/bin/python3 - <<'PY' || { log "FAILED half calib"; exit 1; }
import json
c = json.load(open("shared/fast_sim/rig_calib.json"))
c["p_detect"]["size_px"] = [2 * s for s in c["p_detect"]["size_px"]]
c["source"]["variant"] = "half detection range: each detection chance needs twice the pixels"
json.dump(c, open("shared/fast_sim/rig_calib_half.json", "w"), indent=1)
PY
cp shared/px4_gazebo/gz_logan/rig/data2/det.pt shared/px4_gazebo/gz_logan/rig/det.pt
touch $Q/rig_calib2.done
log "mast rig: detector + calibration done"
