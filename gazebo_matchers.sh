#!/usr/bin/env bash
# gazebo_matchers.sh -- does a better matcher deliver more parcels, ON REAL FIRMWARE?
#
# The point-mass environment could not answer this. It samples each episode by
# sigma decile and draws errors from that decile, so swapping the error pool
# also swaps which episodes the policy sees -- three separate confounds came
# from that, and normalising the sigma column only masked it.
#
# Gazebo has no such coupling. Each mission replays one scene's error through
# real PX4, a real camera and real ArUco decoding, and either lands or does not.
# The seeds are MATCHED across pools, so mission i sees the same scene in every
# arm and only the error magnitude differs.
cd /home/sai/RL_projects/rl-landing
PY=python3
N="${N:-10}"
OUT=gazebo_match
mkdir -p $OUT
# ROS's setup.bash reads unset variables, so `set -u` must come AFTER it or
# sourcing aborts the whole script with no output at all.
set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null
set -u

for pool in sigma_pool.npz sigma_pool_litesam_n.npz sigma_pool_roma_n.npz; do
  tag=$(basename "$pool" .npz | sed 's/sigma_pool_*//; s/_n$//; s/^$/xfeat/')
  echo "=== $tag ($pool) ==="
  $PY -u px4_mission.py --runs "$N" --pool "$pool" --aim mean \
      > "$OUT/$tag.log" 2>&1
  grep -E "^  =>" "$OUT/$tag.log" | tail -3
done
touch $OUT/DONE
