#!/usr/bin/env bash
# run_n20.sh -- both arms of the forest comparison at 20 seeds, unattended.
#
# Sequential by necessity: there is one SITL and one Gazebo, and they cannot be
# parallelised. Arm A (1.80 m) is already flying when this starts; this waits
# for it, swaps the pad, restarts SITL because Gazebo loads the model once, and
# flies arm B on the SAME 20 seeds.
set -u
R=/home/sai/RL_projects/rl-landing
S=/tmp/claude-1000/-home-sai-RL-projects/a49f4b34-af42-4eb1-9d40-d5610c103102/scratchpad
cd "$R"

while pgrep -f "px4_missio[n].py" >/dev/null; do sleep 20; done
echo "ARM A done: $(grep -cE '^  =>' trees_pad18_n20.log)/20 flights"

./ab_pad.sh old
sed -i 's/^TAG_SIZES = .*/TAG_SIZES = {0: 1.20, 1: 0.15}/' px4_mission.py
sed -i 's/^PAD_IDS = .*/PAD_IDS = [0, 1]                  # coarse to fine/' px4_mission.py

pkill -f "bin/p[x]4 "; pkill -f "gz si[m]"; sleep 6
nohup ./launch_px4.sh > "$S/px4_n20b.log" 2>&1 &
sleep 45
pgrep -f "bin/p[x]4 " >/dev/null || { echo "SITL FAILED to restart"; exit 1; }
echo "ARM B: 1.20 m pad, SITL restarted, 10 trees in world"

MISSION_JSON=trees_pad12_n20.json /usr/bin/python3 -u px4_mission.py --runs 20 \
    > trees_pad12_n20.log 2>&1
echo "ARM B done: $(grep -cE '^  =>' trees_pad12_n20.log)/20 flights"
echo ALL_DONE
