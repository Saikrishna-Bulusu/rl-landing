#!/usr/bin/env bash
# after_yolo.sh -- once YOLO's fine-tune (PID in shared/px4_gazebo/gz_logan/rig/yolo/train.pid) exits: score it on the same held-out
# pictures as the small CNN, then the per-condition failure breakdowns for the paper (why_cond.json).
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue; P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
PID=$(cat $R/shared/px4_gazebo/gz_logan/rig/yolo/train.pid)
while kill -0 $PID 2>/dev/null; do sleep 30; done
log "yolo training ended"
cd $R && PYTHONPATH=.yolo_pkgs YOLO_AUTOINSTALL=false $P conditions/magpie/rig_yolo.py --score shared/px4_gazebo/gz_logan/rig/data2 > shared/px4_gazebo/gz_logan/rig/yolo/score.log 2>&1 \
  && log "yolo scored" || log "FAILED yolo score"
cd $R/shared/fast_sim && $P why_cond.py > $R/outputs/fast_sim/why_cond.log 2>&1 && log "why_cond done" || log "FAILED why_cond"
