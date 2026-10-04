#!/usr/bin/env bash
# wind_gz.sh -- the Gazebo wind ladders once GPU training is done (Gazebo and JAX training share the GPU badly):
# the rule's ladder resumed, then the flying learner (filtered, seed 0) on the same 20 yards x 4 winds. Waits on PIDs, not names.
R=/home/sai/RL_projects/rl-landing; cd $R
for p in 518243 519647 579972 ; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
echo "$(date +%H:%M) wind_gz: training done, Gazebo ladders start" >> queue/rig.log
EPISODES="$(seq 0 19 | tr '\n' ' ')" WINDS="9 12 15 18" DECIDERS="rule wind/runs/fly_hp_s0_best.npz" N=4 OUT=ladder_rule ./wind/gz_ladder.sh
echo "$(date +%H:%M) wind_gz: done" >> queue/rig.log
