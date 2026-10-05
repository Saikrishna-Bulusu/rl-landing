#!/usr/bin/env bash
# set10b_missing.sh -- the two crane seeds whose instances stalled (arming refused after a failsafe landing, then an
# unbounded climb loop; 25 Sept): seed 2 on instance 5, seed 8 on instance 6, 15 s apart; then the table's file again.
R=/home/sai/RL_projects/rl-landing; cd $R
OUT=set10b/i4b SEEDS="2" RUNS="rule:local3d planner:survey planner:voxel" SPEED=1.0 FAST=1 SHORT=1 INSTANCE=5 ./conditions/static_obstacles/voxfly.sh > /dev/null 2>&1 &
a=$!; sleep 15
OUT=set10b/i7b SEEDS="8" RUNS="rule:local3d planner:survey planner:voxel" SPEED=1.0 FAST=1 SHORT=1 INSTANCE=6 ./conditions/static_obstacles/voxfly.sh > /dev/null 2>&1 &
wait $a $!
cat outputs/gazebo_voxel/set10b/i[4-8]/flights.jsonl outputs/gazebo_voxel/set10b/i4b/flights.jsonl outputs/gazebo_voxel/set10b/i7b/flights.jsonl > outputs/gazebo_voxel/set10b/flights.jsonl
echo "$(date +%H:%M) set10b complete: $(grep -c outcome outputs/gazebo_voxel/set10b/flights.jsonl) flights" >> outputs/queue/rig.log
