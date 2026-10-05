#!/usr/bin/env bash
# rigv2_land2.sh -- rig v2 (gimbal landing camera, OAK-D under the plate) full deliveries after the self-return fix
# (depth within 1 m dropped: the front propellers' tips had put voxels at the drone's own spot, 25 Sept): seeds 1-3,
# one per instance (4-6), each flown survey-only, voxel look-ahead and local 3D planner.
R=/home/sai/RL_projects/rl-landing; cd $R
pids=(); i=4
for s in 1 2 3; do
    RIG=v2 OUT=rigv2_land2/i$i SEEDS="$s" RUNS="rule:survey rule:voxel rule:local3d" SPEED=1.0 FAST=1 FULL=1 INSTANCE=$i \
        ./conditions/static_obstacles/voxfly.sh > /dev/null 2>&1 &
    pids+=($!); i=$((i + 1)); sleep 15
done
wait "${pids[@]}"
cat outputs/gazebo_voxel/rigv2_land2/i*/flights.jsonl > outputs/gazebo_voxel/rigv2_land2/flights.jsonl
cp outputs/gazebo_voxel/rigv2_land2/i*/*.csv outputs/gazebo_voxel/rigv2_land2/
echo "$(date +%H:%M) rigv2_land2 done: $(grep -c outcome outputs/gazebo_voxel/rigv2_land2/flights.jsonl) flights" >> outputs/queue/rig.log
