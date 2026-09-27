#!/usr/bin/env bash
# gz_pairs.sh -- one video per seed: the planner knowing the magpie through the rig (top) vs handed
# the truth (bottom), same world, magpie and seed. Runs after gz_compare.sh.
R=/home/sai/RL_projects/rl-landing; cd $R
n0=2       # "gz_compare done" lines before the current run (one left by a stopped run at 02:31, +1 per run)
until [ "$(grep -c "gz_compare done" queue/rig.log)" -ge $n0 ]; do sleep 60; done
mkdir -p gazebo_rig/pairs
for s in 0 1 2; do
    [ -f gazebo_rig/planner_rig/planner_s$s.mp4 ] && [ -f gazebo_rig/planner_omni/planner_s$s.mp4 ] && \
        /usr/bin/python3 gz_learner.py --compose gazebo_rig/planner_rig/planner_s$s.mp4 gazebo_rig/planner_omni/planner_s$s.mp4 \
            gazebo_rig/pairs/rig_vs_truth_s$s.mp4 >> gazebo_rig/pairs/pairs.log 2>&1
done
echo "$(date +%H:%M) pair videos done" >> queue/rig.log
