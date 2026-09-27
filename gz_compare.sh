#!/usr/bin/env bash
# gz_compare.sh -- PX4 in Gazebo against the hunter magpie, seeds 0-2: the planner knowing the bird only
# through the mast rig (detector + tracker, no true bird position in the loop) vs handed the truth.
# Starts once the mast rig's detector and calibration exist (rig_v2.sh).
R=/home/sai/RL_projects/rl-landing; cd $R
until [ -f queue/rig_calib2.done ]; do sleep 60; done
OUT=planner_rig SEEDS="0 1 2" DECIDERS=planner SENSE=rig ./rigfly.sh
OUT=planner_omni SEEDS="0 1 2" DECIDERS=planner SENSE=omni ./rigfly.sh
echo "$(date +%H:%M) gz_compare done" >> queue/rig.log
