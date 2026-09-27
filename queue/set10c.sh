#!/usr/bin/env bash
# set10c.sh -- the 10 crane positions with today's drone (rig v2, gimbal) and code: survey-only planner, voxel look-ahead
# planner, rule + local 3D planner; short fast flights at 1x on instances 4-8, starts 15 s apart (for the film's table).
R=/home/sai/RL_projects/rl-landing; cd $R
pids=()
for k in 0 1 2 3 4; do
    RIG=v2 OUT=set10c/i$((k + 4)) SEEDS="$((2 * k + 1)) $((2 * k + 2))" RUNS="rule:local3d planner:survey planner:voxel" \
        SPEED=1.0 FAST=1 SHORT=1 INSTANCE=$((k + 4)) ./voxfly.sh > /dev/null 2>&1 &
    pids+=($!); sleep 15
done
wait "${pids[@]}"
cat gazebo_voxel/set10c/i*/flights.jsonl > gazebo_voxel/set10c/flights.jsonl
echo "$(date +%H:%M) set10c done: $(grep -c outcome gazebo_voxel/set10c/flights.jsonl) flights" >> queue/rig.log
