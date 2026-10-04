#!/usr/bin/env bash
# rig_train.sh -- the learner trained against the hunter magpie, knowing it only through the
# six-camera rig (--birds --hunter --rig): copy the planner, then practice, 3 seeds, corridor world,
# as every other condition. Two GPU slots:
#   A: once overnight_24sep.sh's train_unmapped ends -- stops that chain's own-cameras training
#      (train_cameras_s2: straight-diving magpie, superseded) as it starts -- seeds 0 and 1
#   B: once rig_queue.sh's ladders are done -- seed 2
R=/home/sai/RL_projects/rl-landing; U=$R/urban; Q=$R/queue
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
step () { local name=$1; shift; log "start $name"; if "$@"; then touch $Q/$name.done; log "done  $name"; else log "FAILED $name ($?)"; fi; }
RL="$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --birds --hunter --rig"
cd $U
slotA () {
    until [ -f $Q/rig_calib2.done ]; do sleep 60; done           # the mast rig's calibration
    until [ -f $Q/train_unmapped.done ] || grep -q "FAILED train_unmapped" $Q/overnight.log; do sleep 60; done
    until pid=$(pgrep -f "urban_rl[.]py .*--seeds 2 --birds --cameras"); do sleep 10; done
    kill $pid && log "stopped train_cameras_s2 (pid $pid): straight-diving magpie, superseded"
    # GPU: never a training's start (~1 GB extra while it copies the planner) beside a ladder (3.3 GB)
    # and the Gazebo rig flights (1.6 GB) -- wait for the ladders
    until grep -q "rig ladders finished" $Q/rig.log; do sleep 60; done
    step train_rig_s01 $RL --seeds 0 1 > rl_cond_birds_hunter_rig_s01.log 2>&1
}
slotB () {
    until grep -q "rig ladders finished" $Q/rig.log; do sleep 60; done
    sleep 900                    # a training's first minutes (copying the planner) take ~1 GB more: never two at once
    step train_rig_s2 $RL --seeds 2 > rl_cond_birds_hunter_rig_s2.log 2>&1
}
slotA & slotB & wait
log "rig training finished"
