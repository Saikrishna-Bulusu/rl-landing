#!/usr/bin/env bash
# tello_train.sh -- once YOLO's fine-tune exits (its PID file), train the Tello cage learner: 3 seeds, 120 rounds.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue; P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
PID=$(cat $R/shared/px4_gazebo/gz_logan/rig/yolo/train.pid)
while kill -0 $PID 2>/dev/null; do sleep 30; done
cd $R/archive/tello && log "start tello cage training" && $P train.py --seeds 0 1 2 --gens 120 > train.log 2>&1 \
  && log "done  tello cage training" || log "FAILED tello cage training"
