#!/usr/bin/env bash
# vox_after_yolo1.sh -- once YOLO's seed-1 flight is logged, stop the rig series before seed 2 (PIDs, not
# patterns: a broad pkill once killed the bench), fly the three voxel flights, then resume YOLO seed 2.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
F=$R/outputs/gazebo_rig/planner_rig_yolo/flights.jsonl
until [ "$(grep -c '"outcome"' $F 2>/dev/null)" -ge 2 ]; do sleep 10; done
for pid in $(pgrep -f "bash ./conditions/magpie/rigfly.sh"); do kill $pid; done
sleep 2
for pid in $(pgrep -f "shared/px4_gazebo/gz_learner.py --decider planner --seed 2 --magpie hunter"); do kill $pid; done
for pid in $(pgrep -f "px4_sitl_default/bin/px4") $(pgrep -f "gz si[m].*shared/px4_gazebo/gz_logan/rigfly/aruco.sdf"); do kill $pid; done
log "yolo rig flights paused after seed 1 (seed 2 after the voxel flights)"
sleep 20
cd $R && OUT=first SEEDS="0" RUNS="planner:survey planner:voxel learner:voxel" ./conditions/static_obstacles/voxfly.sh
cd $R && RIG_YOLO=$R/shared/px4_gazebo/gz_logan/rig/yolo/run/weights/best.pt RIG_THRESH=0.5 OUT=planner_rig_yolo SEEDS="2" DECIDERS=planner SENSE=rig ./conditions/magpie/rigfly.sh
cd $R/shared/fast_sim && /home/sai/miniconda3/envs/mujoco_drones/bin/python $R/conditions/magpie/rig_replay.py $R/outputs/gazebo_rig/planner_rig_yolo/*_rig.jsonl > $R/outputs/gazebo_rig/phantoms_yolo.txt 2>&1
log "yolo seed 2 flown; phantom replay redone"
