#!/usr/bin/env bash
# land_gate.sh -- the final-descent gate (keep descending only when centred and nearly still; 26 Sept): the calm crane
# landings again (rig v2, seeds 1-6, fast) and the live magpie flights on the up-camera rig again (planner, learner
# seed 0's checkpoint; seeds 0-2; the magpie known only through the rig), on three simulators side by side.
R=/home/sai/RL_projects/rl-landing; cd $R
CK=rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt
RIG=v2 OUT=land_gate/v2 SEEDS="1 2 3 4 5 6" RUNS="rule:local3d" SPEED=1.0 FAST=1 FULL=1 INSTANCE=4 ./conditions/static_obstacles/voxfly.sh > /dev/null 2>&1 & a=$!
sleep 20
RIG=v2l OUT=planner_rig_v2l_gate SEEDS="0 1 2" DECIDERS=planner SENSE=rig INSTANCE=1 ./conditions/magpie/rigfly.sh > /dev/null 2>&1 & b=$!
sleep 20
RIG=v2l OUT=learner_rig_v2l_gate SEEDS="0 1 2" DECIDERS=learner SENSE=rig CKPT=$CK INSTANCE=2 ./conditions/magpie/rigfly.sh > /dev/null 2>&1 & c=$!
wait $a $b $c
echo "$(date +%H:%M) land_gate done" >> outputs/queue/rig.log
