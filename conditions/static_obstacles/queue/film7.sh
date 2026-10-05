#!/usr/bin/env bash
# film7.sh -- the crane film again with today's drone (rig v2: OAK-D depth under the plate, gimbal landing camera) and
# today's code (async mover, truth-judged, nose rules): seeds 2 3 6, survey-only planner vs rule + local 3D planner,
# short flights at 0.4x on instances 1-3, starts 15 s apart.
R=/home/sai/RL_projects/rl-landing; cd $R
pids=(); i=1
for s in 2 3 6; do
    RIG=v2 OUT=film7/i$i SEEDS="$s" RUNS="rule:local3d planner:survey" SPEED=0.4 SHORT=1 INSTANCE=$i ./conditions/static_obstacles/voxfly.sh > /dev/null 2>&1 &
    pids+=($!); i=$((i + 1)); sleep 15
done
wait "${pids[@]}"
cat outputs/gazebo_voxel/film7/i*/flights.jsonl > outputs/gazebo_voxel/film7/flights.jsonl
cp outputs/gazebo_voxel/film7/i*/*.mp4 outputs/gazebo_voxel/film7/
echo "$(date +%H:%M) film7 done: $(grep -c outcome outputs/gazebo_voxel/film7/flights.jsonl) flights" >> outputs/queue/rig.log
