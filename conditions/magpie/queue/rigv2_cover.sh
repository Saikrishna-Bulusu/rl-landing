#!/usr/bin/env bash
# rigv2_cover.sh -- step 3 stage A (25 Sept): rig v2's magpie coverage on the bench (shared/px4_gazebo/gz_logan/rig_v2, partition rig2):
# 160 directions x 6 ranges, head-on and broadside; the camera-model check; the chart and the contact sheet.
R=/home/sai/RL_projects/rl-landing; cd $R; export RIG=v2
O=shared/px4_gazebo/gz_logan/rig_v2/cover; mkdir -p $O
G=$(/usr/bin/python3 shared/px4_gazebo/gz_rig.py --bench)                 # headless Gazebo, its own session; prints its PID
trap 'kill -- -$G 2>/dev/null; sleep 3; kill -9 -- -$G 2>/dev/null' EXIT
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --cover $O > shared/px4_gazebo/gz_logan/rig_v2/cover.log 2>&1 && \
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --report $O >> shared/px4_gazebo/gz_logan/rig_v2/cover.log 2>&1 && \
/usr/bin/python3 -u shared/px4_gazebo/gz_rig.py --views $O >> shared/px4_gazebo/gz_logan/rig_v2/cover.log 2>&1
echo "$(date +%H:%M) rigv2 cover: $(tail -1 shared/px4_gazebo/gz_logan/rig_v2/cover.log)" >> outputs/queue/rig.log
