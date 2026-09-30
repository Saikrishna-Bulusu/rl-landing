#!/usr/bin/env bash
# All sensitivity variations, one at a time, low priority beside Gazebo. Log: sens/run_all.log
cd /home/sai/RL_projects/rl-landing/urban; PY="nice -n 15 /home/sai/miniconda3/envs/mujoco_drones/bin/python"
run () { cond=$1; label=$2; shift 2; [ -f sens/${cond}_${label}.json ] && return; env "$@" $PY sens.py $cond $label 2>&1 | grep -v -i warn | tail -1 >> sens/run_all.log; }
for c in clear fog rig; do
  run $c base
  run $c leave10_120 SENS_LEAVE=10,120
  run $c leave60_480 SENS_LEAVE=60,480
  run $c tmax200 SENS_T_MAX=200
  run $c tmax400 SENS_T_MAX=400
  run $c sep50 SENS_DRONE_SEP=50
done
run rig commit0.1 HUNTER_T_COMMIT=0.1
run rig commit0.5 HUNTER_T_COMMIT=0.5
run rig halfrange RIG_CALIB=rig_calib_half.json
echo "ALL DONE $(date +%H:%M)" >> sens/run_all.log
