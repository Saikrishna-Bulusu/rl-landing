#!/usr/bin/env bash
# film6.sh -- the crane film's flights again (25 Sept), alone on the machine, now that Gazebo's blocking set_pose runs in
# a helper process (Mover(asyn=True)): it had frozen the flight loop at every chase-camera move (29 -> 2-10 Hz).
R=/home/sai/RL_projects/rl-landing; cd $R
pids=(); i=1
for s in 2 3 6; do
    OUT=film6/i$i SEEDS="$s" RUNS="rule:local3d planner:survey" SPEED=0.4 SHORT=1 INSTANCE=$i ./voxfly.sh > /dev/null 2>&1 &
    pids+=($!); i=$((i + 1)); sleep 15
done
wait "${pids[@]}"
cat gazebo_voxel/film6/i*/flights.jsonl > gazebo_voxel/film6/flights.jsonl
cp gazebo_voxel/film6/i*/*.mp4 gazebo_voxel/film6/
echo "$(date +%H:%M) film6 done: $(grep -c outcome gazebo_voxel/film6/flights.jsonl) flights" >> queue/rig.log
