#!/usr/bin/env bash
# vox_yawfix.sh -- the first voxel flights flew tail first (PX4 held its -102 deg spawn yaw, so the depth camera
# looked back): wait for that conditions/static_obstacles/voxfly.sh to wind down, fly the three again with the nose along the path and the
# voxel columns padded by 1 m, then the YOLO rig flight for seed 2 and its phantom replay.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
while kill -0 104826 2>/dev/null; do sleep 5; done
while pgrep -f "gz si[m].*shared/px4_gazebo/gz_logan/voxfly/aruco.sdf" > /dev/null || pgrep -f "px4_sitl_default/bin/px4" > /dev/null; do sleep 5; done
sleep 10
log "voxel flights again: nose along the path, columns padded 1 m"
cd $R && OUT=yawfix SEEDS="0" RUNS="planner:survey planner:voxel learner:voxel" ./conditions/static_obstacles/voxfly.sh
cd $R && RIG_YOLO=$R/shared/px4_gazebo/gz_logan/rig/yolo/run/weights/best.pt RIG_THRESH=0.5 OUT=planner_rig_yolo SEEDS="2" DECIDERS=planner SENSE=rig ./conditions/magpie/rigfly.sh
cd $R/shared/fast_sim && /home/sai/miniconda3/envs/mujoco_drones/bin/python $R/conditions/magpie/rig_replay.py $R/outputs/gazebo_rig/planner_rig_yolo/*_rig.jsonl > $R/outputs/gazebo_rig/phantoms_yolo.txt 2>&1
log "yolo seed 2 flown; phantom replay redone"
