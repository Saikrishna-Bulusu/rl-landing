#!/usr/bin/env bash
# after_rerun.sh -- once vox_rerun.sh (PID 168905) ends: the three film flights again (the video no longer starves the
# flight loop: 29 -> 4 Hz on 25 Sept, two false crashes) on instances 1-3, and the first rig v2 flights (the landing
# camera on PX4's gimbal) on instance 4: two full deliveries past the crane, fast, no video.
R=/home/sai/RL_projects/rl-landing; Q=$R/queue; cd $R
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
while kill -0 168905 2>/dev/null; do sleep 5; done
sleep 10
log "film3 (seeds 2 3 6, instances 1-3) + rig v2 test (seeds 1 2 full deliveries, instance 4)"
pids=()
i=1; for s in 2 3 6; do
    OUT=film3/i$i SEEDS="$s" RUNS="rule:local3d planner:survey" SPEED=0.4 SHORT=1 INSTANCE=$i ./voxfly.sh > /dev/null 2>&1 &
    pids+=($!); i=$((i + 1)); sleep 15
done
RIG=v2 OUT=rigv2_test SEEDS="1 2" RUNS="rule:local3d" SPEED=1.0 FAST=1 INSTANCE=4 ./voxfly.sh > /dev/null 2>&1 &
pids+=($!)
wait "${pids[@]}"
cat gazebo_voxel/film3/i*/flights.jsonl > gazebo_voxel/film3/flights.jsonl
cp gazebo_voxel/film3/i*/*.mp4 gazebo_voxel/film3/
log "film3 + rig v2 test done"
