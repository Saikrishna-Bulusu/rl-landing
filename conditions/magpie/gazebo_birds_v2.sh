#!/usr/bin/env bash
# gazebo_birds_v2.sh -- the eight Gazebo magpie flights again, filmed smoothly. The first videos
# (gazebo_birds/, kept) were framed from the flight loop, which pauses while the decider thinks:
# 5-10 new pictures a second, frames repeated up to 16 times. Now every chase-camera image is a
# frame (gz_learner.Film) and the simulation runs at a tenth of real speed, so the decider's
# thinking costs ~0.03 s of simulated time instead of 0.1-0.3 s.
# Needs the SITL up at 0.1x in the magpie world:
#   PX4_SIM_SPEED_FACTOR=0.1 WORLDS_DIR=$PWD/gz_logan/birds EXTRA_MODELS=$PWD/gz_logan/models ./launch_px4.sh
R=/home/sai/RL_projects/rl-landing; D=$R/gazebo_birds/v2
cd $R && mkdir -p $D
for s in 0 1 2 3; do for d in planner learner; do
    TREES_JSON=gz_logan/trees_logan.json timeout 2400 /usr/bin/python3 -u gz_learner.py --decider $d --seed $s \
        --video $D/${d}_s$s.mp4 >> $D/flights.jsonl 2>> $D/flights.err
done; done
for s in 0 1 2 3; do /usr/bin/python3 gz_learner.py --compose $D/planner_s$s.mp4 $D/learner_s$s.mp4 $D/pair_s$s.mp4 \
    >> $D/flights.err 2>&1; done
/usr/bin/python3 gz_learner.py --final $D/gazebo_magpie_smooth.mp4 >> $D/flights.err 2>&1
# only THIS world's simulator: a bare "gz sim" pattern also killed the rig bench (25 Sept)
kill $(pgrep -f "bin/p[x]4 ") $(pgrep -f "gz si[m].*gz_logan/birds/aruco.sdf") 2>/dev/null
[ "$(grep -c '"outcome"' $D/flights.jsonl)" = 8 ] && [ -s $D/gazebo_magpie_smooth.mp4 ] && touch $R/queue/gazebo_birds_v2.done
