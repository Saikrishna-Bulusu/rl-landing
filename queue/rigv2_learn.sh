#!/usr/bin/env bash
# rigv2_learn.sh -- step 3 stage C (25 Sept): once stage B (PID 210867) has written urban/rig_calib_v2.json, the fast
# simulator flies rig v2: the magpie ladder (rule / planner / all-knowing with the rig), then the rig learner, 3 seeds
# (seeds 0 1 in one process, 2 in another, as rig_train.sh ran the first rig's), named ..._rig_v2_s*.
R=/home/sai/RL_projects/rl-landing; Q=$R/queue; P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
log () { echo "$(date +%H:%M) rigv2 learn: $*" >> $Q/rig.log; }
while kill -0 210867 2>/dev/null; do sleep 30; done
[ -f $R/urban/rig_calib_v2.json ] || { log "no rig_calib_v2.json -- stage B failed"; exit 1; }
cd $R/urban
export RIG_CALIB=$R/urban/rig_calib_v2.json PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "ladder"
$P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter --rig --tag _v2 > headroom_lidar_gps_birds_hunter_rig_v2.log 2>&1     || { log "ladder failed"; exit 1; }
log "ladder done: $(grep -m1 -A4 'corridor world' headroom_lidar_gps_birds_hunter_rig_v2.log | tail -4 | tr -s ' ' | cut -c1-120 | tr '\n' '|')"
RL="$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --birds --hunter --rig --tag _v2"
$RL --seeds 0 1 > rl_cond_birds_hunter_rig_v2_s01.log 2>&1 & a=$!
sleep 60
$RL --seeds 2 > rl_cond_birds_hunter_rig_v2_s2.log 2>&1 & b=$!
wait $a; ra=$?; wait $b; rb=$?
log "learner done (exit $ra $rb): $(ls rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2_s*_mean.pt 2>/dev/null | wc -l) checkpoints"
