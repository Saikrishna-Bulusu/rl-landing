#!/usr/bin/env bash
# vox_seeds3.sh -- the seed-1 survey flight finishes (PID 133026); close that Gazebo session by PID; then the
# remaining voxel flights at 0.3x (these carry no rig cameras, so the loop keeps up): seed 1 rule+local3d and
# voxel, then seeds 2-5 rule+local3d, survey only, voxel look-ahead.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
while kill -0 133026 2>/dev/null; do sleep 5; done
kill 132437 132679 2>/dev/null
while kill -0 132437 2>/dev/null || kill -0 132679 2>/dev/null; do sleep 2; done
sleep 15
log "voxel flights at 0.3x: seed 1 rule+local3d, voxel; then seeds 2-5"
cd $R && SPEED=0.3 OUT=seeds SEEDS="1" RUNS="rule:local3d planner:voxel" ./conditions/static_obstacles/voxfly.sh
sleep 15
cd $R && SPEED=0.3 OUT=seeds SEEDS="2 3 4 5" RUNS="rule:local3d planner:survey planner:voxel" ./conditions/static_obstacles/voxfly.sh
log "voxel seeds done"
