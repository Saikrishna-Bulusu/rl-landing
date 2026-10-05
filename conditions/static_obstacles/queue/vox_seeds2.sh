#!/usr/bin/env bash
# vox_seeds2.sh -- the seed-1 local-planner flight (MPC over the planner, before the fixes) finishes; then close
# that Gazebo session by PID and fly seeds 1-5 again: the RULE flying the local 3D planner's path (with path
# commitment, clearance slow-down, nose along the path), then survey only and voxel look-ahead for comparison.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
while kill -0 129835 2>/dev/null; do sleep 5; done
kill 129262 129490 2>/dev/null
while kill -0 129262 2>/dev/null || kill -0 129490 2>/dev/null; do sleep 2; done
sleep 15
log "voxel seeds 1-5 again: rule + local 3D planner (fixed), survey, voxel"
cd $R && OUT=seeds SEEDS="1 2 3 4 5" RUNS="rule:local3d planner:survey planner:voxel" ./conditions/static_obstacles/voxfly.sh
