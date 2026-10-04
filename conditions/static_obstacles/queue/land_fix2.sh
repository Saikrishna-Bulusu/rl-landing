#!/usr/bin/env bash
# land_fix2.sh -- the landing fix (heading locked in the terminal area; lag-aware centring), same A/B as land_ab:
# rig v2 with the gimbal and with the fixed camera, seeds 1-6, local 3D planner, whole deliveries, instances 4 and 5.
R=/home/sai/RL_projects/rl-landing; cd $R
RIG=v2f OUT=land_fix2/v2f SEEDS="1 2 3 4 5 6" RUNS="rule:local3d" SPEED=1.0 FAST=1 FULL=1 INSTANCE=4 ./voxfly.sh > /dev/null 2>&1 & a=$!
sleep 20
RIG=v2 OUT=land_fix2/v2 SEEDS="1 2 3 4 5 6" RUNS="rule:local3d" SPEED=1.0 FAST=1 FULL=1 INSTANCE=5 ./voxfly.sh > /dev/null 2>&1 & b=$!
wait $a $b
echo "$(date +%H:%M) land_fix2 done" >> queue/rig.log
