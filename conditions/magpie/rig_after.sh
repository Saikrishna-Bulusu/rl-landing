#!/usr/bin/env bash
# conditions/magpie/rig_after.sh -- after conditions/magpie/rig_queue.sh's ladders, the rig learner: seeds 0-1 at once, seed 2 fifteen
# minutes later (a training's start takes ~1 GB extra while it copies the planner). (The rig ladder
# started 04:33, after the 0.5 calibration was written at 04:32:36 -- one calibration throughout.)
R=/home/sai/RL_projects/rl-landing; U=$R/shared/fast_sim; Q=$R/outputs/queue
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
step () { local name=$1; shift; log "start $name"; if "$@"; then touch $Q/$name.done; log "done  $name"; else log "FAILED $name ($?)"; fi; }
cd $U
until grep -q "rig ladders finished" $Q/rig.log; do sleep 30; done
RL="$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --birds --hunter --rig"
( step train_rig_s01 $RL --seeds 0 1 > $R/outputs/fast_sim/rl_cond_birds_hunter_rig_s01.log 2>&1 ) &
sleep 900
step train_rig_s2 $RL --seeds 2 > $R/outputs/fast_sim/rl_cond_birds_hunter_rig_s2.log 2>&1
wait
log "rig training finished"
