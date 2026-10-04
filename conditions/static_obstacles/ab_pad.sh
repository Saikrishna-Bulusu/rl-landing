#!/usr/bin/env bash
# ab_pad.sh -- the same five missions on the old pad, for the comparison.
#
# Runs are seeded by index (Mission(seed=i)), so mission i draws the SAME
# sigma and the SAME localization error in both arms. Nothing but the paint
# changes. SITL must be restarted because Gazebo loads the model once.
#
# This used to move SEARCH_ALT too, which made the arms differ in two ways at
# once and confounded the only significant result in the comparison. It now
# touches the pad and nothing else -- change flight parameters deliberately,
# not as a side effect of swapping paint.
set -e
M=/home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/models/arucotag
R=/home/sai/RL_projects/rl-landing
case "$1" in
  old) cp -a $M.bak-1.2m/arucotag.png $M/arucotag.png
       cp -a $M.bak-1.2m/model.sdf    $M/model.sdf
       echo "installed the 1.20 m pad" ;;
  new) cd $R && /home/sai/miniconda3/envs/mujoco_drones/bin/python make_pad.py --write >/dev/null
       echo "installed the generated pad" ;;
  *) echo "usage: $0 old|new"; exit 1 ;;
esac
grep -n "^SEARCH_ALT" $R/px4_mission.py  # for the record, not changed here
