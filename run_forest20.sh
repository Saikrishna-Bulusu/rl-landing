#!/usr/bin/env bash
# run_forest20.sh -- both arms, 20 trials, a different forest every trial.
#
# ONE MISSION PER PROCESS. The batch version died at trial 3 in both arms:
# scene/info stopped answering inside 2 s while Gazebo rendered the camera the
# same process was subscribed to. standing() now retries with backoff, and each
# trial gets a fresh interpreter as well, so no gz transport state accumulates
# across trials. A python start costs ~2 s against the 45 s a SITL restart
# costs, so SITL still comes up once per arm.
#
# Trial i flies forest seed 300+i and sigma seed i in BOTH arms, so a pair
# differs only in the coarse marker.
R=/home/sai/RL_projects/rl-landing
S=/tmp/claude-1000/-home-sai-RL-projects/a49f4b34-af42-4eb1-9d40-d5610c103102/scratchpad
export PATH=/opt/ros/jazzy/opt/gz_tools_vendor/bin:$PATH
# set -u only AFTER this: setup.bash reads an unset var and would kill us.
source /opt/ros/jazzy/setup.bash 2>/dev/null
set -u
cd "$R"
N=${N:-20}

arm () {            # $1 label   $2 json prefix   $3 log
  pkill -f "bin/p[x]4 "; pkill -f "gz si[m]"; sleep 6
  nohup ./launch_px4.sh > "$S/px4_$1.log" 2>&1 &
  sleep 45
  pgrep -f "bin/p[x]4 " >/dev/null || { echo "SITL FAILED for $1"; return 1; }
  echo "ARM $1 flying"
  : > "$3"
  for i in $(seq 0 $((N - 1))); do
    echo "--- $1 trial $i (forest $((300 + i)), sigma seed $i) ---" >> "$3"
    MISSION_JSON="${2}_${i}.json" timeout 420 /usr/bin/python3 -u px4_mission.py \
        --runs 1 --vary-forest --forest0 $((300 + i)) --seed0 "$i" --search-alt 18 >> "$3" 2>&1 \
        || echo "    trial $i exited $?" >> "$3"
    echo "  $1 $((i + 1))/$N done ($(grep -cE '^  =>' "$3") flights recorded)"
  done
}

./ab_pad.sh new
sed -i 's/^TAG_SIZES = .*/TAG_SIZES = {2: 1.80, 0: 1.20, 1: 0.15}/' px4_mission.py
sed -i 's/^PAD_IDS = .*/PAD_IDS = [2, 0, 1]               # coarse to fine/' px4_mission.py
arm A180 forest18 forest_pad18_n20.log

./ab_pad.sh old
sed -i 's/^TAG_SIZES = .*/TAG_SIZES = {0: 1.20, 1: 0.15}/' px4_mission.py
sed -i 's/^PAD_IDS = .*/PAD_IDS = [0, 1]                  # coarse to fine/' px4_mission.py
arm B120 forest12 forest_pad12_n20.log

./ab_pad.sh new
sed -i 's/^TAG_SIZES = .*/TAG_SIZES = {2: 1.80, 0: 1.20, 1: 0.15}/' px4_mission.py
sed -i 's/^PAD_IDS = .*/PAD_IDS = [2, 0, 1]               # coarse to fine/' px4_mission.py
pkill -f "bin/p[x]4 "; pkill -f "gz si[m]"
echo "SITL stopped; machine quiet"
echo ALL_DONE
