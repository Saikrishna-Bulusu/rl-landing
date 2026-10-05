#!/usr/bin/env bash
# vox_rerun.sh -- the crane film and the 10 crane positions again, after two fixes (25 Sept): PX4's GPS delay set to
# the simulator's (its estimate ran 0.11 s ahead of the drone) and crashes judged where Gazebo has the drone.
# Film seeds 2 3 6 at 0.4x on instances 1-3, the 10 seeds fast (1x, no video) on instances 4-8; starts 15 s apart.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue; cd $R
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
log "vox rerun: film seeds 2 3 6 (instances 1-3) + 10 crane seeds fast (instances 4-8)"
pids=()
i=1; for s in 2 3 6; do
    OUT=film2/i$i SEEDS="$s" RUNS="rule:local3d planner:survey" SPEED=0.4 SHORT=1 INSTANCE=$i ./conditions/static_obstacles/voxfly.sh > /dev/null 2>&1 &
    pids+=($!); i=$((i + 1)); sleep 15
done
for k in 0 1 2 3 4; do
    OUT=set10b/i$((k + 4)) SEEDS="$((2 * k + 1)) $((2 * k + 2))" RUNS="rule:local3d planner:survey planner:voxel" \
        SPEED=1.0 FAST=1 SHORT=1 INSTANCE=$((k + 4)) ./conditions/static_obstacles/voxfly.sh > /dev/null 2>&1 &
    pids+=($!); sleep 15
done
wait "${pids[@]}"
cat outputs/gazebo_voxel/film2/i*/flights.jsonl > outputs/gazebo_voxel/film2/flights.jsonl
cat outputs/gazebo_voxel/set10b/i*/flights.jsonl > outputs/gazebo_voxel/set10b/flights.jsonl
cp outputs/gazebo_voxel/film2/i*/*.mp4 outputs/gazebo_voxel/film2/
log "vox rerun done: $(grep -c outcome outputs/gazebo_voxel/film2/flights.jsonl) film + $(grep -c outcome outputs/gazebo_voxel/set10b/flights.jsonl) fast flights"
