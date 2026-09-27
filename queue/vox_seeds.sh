#!/usr/bin/env bash
# vox_seeds.sh -- after the queue that is flying YOLO's seed 2 (PID 109983) ends: the crane moved per seed,
# seeds 1-5, each flown with the local 3D planner, trusting the survey only, and on the voxel map alone.
R=/home/sai/RL_projects/rl-landing; Q=$R/queue
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
while kill -0 109983 2>/dev/null; do sleep 10; done
sleep 20
log "voxel seeds 1-5: crane moved, local3d / survey / voxel"
cd $R && OUT=seeds SEEDS="1 2 3 4 5" RUNS="planner:local3d planner:survey planner:voxel" ./voxfly.sh
