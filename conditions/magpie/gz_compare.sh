#!/usr/bin/env bash
# conditions/magpie/gz_compare.sh -- PX4 in Gazebo against the hunter magpie, seeds 0-2: the planner knowing the bird only
# through the mast rig (detector + tracker, no true bird position in the loop) vs handed the truth.
# Starts once the mast rig's detector and calibration exist (conditions/magpie/rig_v2.sh).
R=/home/sai/RL_projects/rl-landing; cd $R
until [ -f outputs/queue/rig_calib2.done ]; do sleep 60; done
OUT=planner_rig SEEDS="0 1 2" DECIDERS=planner SENSE=rig ./conditions/magpie/rigfly.sh
OUT=planner_omni SEEDS="0 1 2" DECIDERS=planner SENSE=omni ./conditions/magpie/rigfly.sh
echo "$(date +%H:%M) gz_compare done" >> outputs/queue/rig.log
