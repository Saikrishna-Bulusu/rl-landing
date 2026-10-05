#!/usr/bin/env bash
# rig_pipeline.sh VARIANT -- a rig variant (gz_unified.VARIANTS) through the whole magpie pipeline (25 Sept):
#   B  labelled pictures on its bench (600 places + 200 empty), YOLO fine-tuned from rig v2's, held-out score,
#      calibration (shared/fast_sim/rig_calib_VARIANT.json)
#   C  the fast simulator's magpie ladder, then the rig learner, 3 seeds (..._rig_VARIANT_s*)
#   D  live Gazebo flights against the hunter magpie, known only through this rig: planner and learner, seeds 0-2
V=${1:?variant}; R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue; P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export RIG=$V YOLO_AUTOINSTALL=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) pipeline $V: $*" >> $Q/rig.log; }
L=shared/px4_gazebo/gz_logan/rig_$V; D=$L/data; cd $R; mkdir -p $D
G=$(/usr/bin/python3 shared/px4_gazebo/gz_rig.py --bench)
trap 'kill -- -$G 2>/dev/null; sleep 3; kill -9 -- -$G 2>/dev/null' EXIT
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --data $D --poses 600 > $L/data.log 2>&1 || { log "data failed"; exit 1; }
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --negatives $D > $L/negatives.log 2>&1 || { log "negatives failed"; exit 1; }
kill -- -$G 2>/dev/null; sleep 3; kill -9 -- -$G 2>/dev/null
log "B pictures: $(wc -l < $D/labels.jsonl) labelled"
PYTHONPATH=.yolo_pkgs $P conditions/magpie/rig_yolo.py --data $D > $L/yolo_build.log 2>&1 || { log "build failed"; exit 1; }
PYTHONPATH=.yolo_pkgs $P conditions/magpie/rig_yolo.py --train --epochs 30 --init $R/shared/px4_gazebo/gz_logan/rig_v2/yolo/run/weights/best.pt > $L/yolo_train.log 2>&1 \
    || { log "train failed"; exit 1; }
PYTHONPATH=.yolo_pkgs $P conditions/magpie/rig_yolo.py --score $D > $L/yolo_score.log 2>&1 || { log "score failed"; exit 1; }
/usr/bin/python3 shared/px4_gazebo/gz_rig.py --calib $L/cover $D --score score_yolo.json --out shared/fast_sim/rig_calib_$V.json > $L/calib.log 2>&1 \
    || { log "calib failed"; exit 1; }
log "B done: $(grep -o '"found": [0-9.]*' $D/score_yolo.json | head -1)"
cd shared/fast_sim; export RIG_CALIB=$R/shared/fast_sim/rig_calib_$V.json
$P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter --rig --tag _$V > $R/outputs/fast_sim/headroom_lidar_gps_birds_hunter_rig_$V.log 2>&1 \
    || { log "ladder failed"; exit 1; }
log "C ladder: $(grep -m2 -A2 'corridor world' $R/outputs/fast_sim/headroom_lidar_gps_birds_hunter_rig_$V.log | grep -m1 rule | tr -s ' ' | cut -c1-110)"
RL="$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --birds --hunter --rig --tag _$V"
$RL --seeds 0 1 > $R/outputs/fast_sim/rl_cond_birds_hunter_rig_${V}_s01.log 2>&1 & a=$!
sleep 60
$RL --seeds 2 > $R/outputs/fast_sim/rl_cond_birds_hunter_rig_${V}_s2.log 2>&1 & b=$!
wait $a $b
CK=rl_corridor_eyes_lidar_gps_birds_hunter_rig_${V}_s0_mean.pt
[ -f $CK ] || { log "no $CK"; exit 1; }
log "C learner: $(ls rl_corridor_eyes_lidar_gps_birds_hunter_rig_${V}_s*_mean.pt | wc -l) checkpoints"
cd $R
OUT=planner_rig_$V SEEDS="0 1 2" DECIDERS=planner SENSE=rig INSTANCE=1 ./conditions/magpie/rigfly.sh > /dev/null 2>&1 & a=$!
sleep 20
OUT=learner_rig_$V SEEDS="0 1 2" DECIDERS=learner SENSE=rig CKPT=$CK INSTANCE=2 ./conditions/magpie/rigfly.sh > /dev/null 2>&1 & b=$!
wait $a $b
log "D done: planner $(grep -c outcome outputs/gazebo_rig/planner_rig_$V/flights.jsonl) learner $(grep -c outcome outputs/gazebo_rig/learner_rig_$V/flights.jsonl) flights"
