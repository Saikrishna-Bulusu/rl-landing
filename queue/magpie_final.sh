#!/usr/bin/env bash
# magpie_final.sh -- the six live magpie flights on the up-camera rig with the final landing (26 Sept): centred-and-still
# gate below 6 m (< 0.1 m/s in the last 2 m), heading locked at the marker read, turning tapered off 60 -> 35 m out.
R=/home/sai/RL_projects/rl-landing; cd $R
CK=rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt
RIG=v2l OUT=planner_rig_v2l_final SEEDS="0 1 2" DECIDERS=planner SENSE=rig INSTANCE=1 ./rigfly.sh > /dev/null 2>&1 & a=$!
sleep 20
RIG=v2l OUT=learner_rig_v2l_final SEEDS="0 1 2" DECIDERS=learner SENSE=rig CKPT=$CK INSTANCE=2 ./rigfly.sh > /dev/null 2>&1 & b=$!
wait $a $b
echo "$(date +%H:%M) magpie_final done" >> queue/rig.log
