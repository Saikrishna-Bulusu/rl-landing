#!/usr/bin/env bash
# rigv2_fly.sh -- step 3 stage D (25 Sept): once stage C (PID 212679) has trained the rig v2 learner, the live magpie
# flights on rig v2 -- the hunter magpie known ONLY through rig v2's cameras, v2 detector and tracker: the planner and
# the learner (seed 0's, as for the first rig), seeds 0-2 as the first rig flew, two simulators side by side at 0.1x.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue
log () { echo "$(date +%H:%M) rigv2 fly: $*" >> $Q/rig.log; }
while kill -0 212679 2>/dev/null; do sleep 30; done
CK=rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2_s0_mean.pt
[ -f $R/shared/fast_sim/$CK ] || { log "no $CK -- stage C failed"; exit 1; }
cd $R
log "planner and learner, seeds 0-2"
RIG=v2 OUT=planner_rig_v2 SEEDS="0 1 2" DECIDERS=planner SENSE=rig INSTANCE=1 ./conditions/magpie/rigfly.sh > /dev/null 2>&1 & a=$!
sleep 20
RIG=v2 OUT=learner_rig_v2 SEEDS="0 1 2" DECIDERS=learner SENSE=rig CKPT=$CK INSTANCE=2 ./conditions/magpie/rigfly.sh > /dev/null 2>&1 & b=$!
wait $a $b
log "done: planner $(grep -c outcome outputs/gazebo_rig/planner_rig_v2/flights.jsonl) learner $(grep -c outcome outputs/gazebo_rig/learner_rig_v2/flights.jsonl) flights"
