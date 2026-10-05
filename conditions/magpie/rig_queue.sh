#!/usr/bin/env bash
# conditions/magpie/rig_queue.sh -- the headroom ladders against the HUNTER magpie (from behind, steering), for three
# kinds of bird sensing: handed the truth (omni: the old assumption, now an upper bound), the drone's
# own two cameras (lower bound), the six-camera rig through its detector and tracker (rig_calib.json,
# measured in Gazebo) -- plus the rig at HALF its detection range (rig_calib_half.json: a detector
# needing twice the pixels), the sensitivity check Sai was promised.
#
# GPU: runs in chain B's slot of all_conditions/queue/overnight_24sep.sh once train_all ends, stopping that chain's
# own-cameras training (train_cameras_s01) as it starts -- it trained against the straight-diving
# magpie Sai replaced on 24 Sept, so its results would not be used. Logs to outputs/queue/rig.log.
R=/home/sai/RL_projects/rl-landing; U=$R/shared/fast_sim; Q=$R/outputs/queue
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) $*" >> $Q/rig.log; }
step () { local name=$1; shift; log "start $name"; if "$@"; then touch $Q/$name.done; log "done  $name"; else log "FAILED $name ($?)"; fi; }
HR="$P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter"

log "waiting for train_all to end and the mast-rig calibration"
until [ -f $Q/rig_calib2.done ]; do sleep 60; done
until [ -f $Q/train_all.done ] || grep -q "FAILED train_all" $Q/overnight.log; do sleep 60; done
until pid=$(pgrep -f "urban_rl[.]py .*--seeds 0 1 --birds --cameras"); do sleep 10; done
kill $pid && log "stopped train_cameras_s01 (pid $pid): straight-diving magpie, superseded"
sleep 30
cd $U
step ladder_hunter_omni    $HR           > $R/outputs/fast_sim/headroom_lidar_gps_birds_hunter.log 2>&1
step ladder_hunter_cameras $HR --cameras > $R/outputs/fast_sim/headroom_lidar_gps_birds_cameras_hunter.log 2>&1
step ladder_hunter_rig     $HR --rig     > $R/outputs/fast_sim/headroom_lidar_gps_birds_hunter_rig.log 2>&1
export RIG_CALIB=$U/rig_calib_half.json
step ladder_hunter_rig_half $HR --rig --tag _half > $R/outputs/fast_sim/headroom_lidar_gps_birds_hunter_rig_half.log 2>&1
unset RIG_CALIB
log "rig ladders finished"
