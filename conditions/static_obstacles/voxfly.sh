#!/usr/bin/env bash
# conditions/static_obstacles/voxfly.sh -- stage 3 in Gazebo (shared/px4_gazebo/gz_voxel.py's world: a 58 m crane on the way in, street lights, a traffic
# light, new trees, a shed; the x500 carrying PX4's depth camera). No magpie. Each RUN is decider:map --
# map "survey" trusts the 2023 survey alone, "voxel" also the voxel map built from the depth camera's images,
# "local3d" the voxel map AND a local 3D planner (shared/fast_sim/local_planner.py) flying round what it shows.
# The crane stands where gz_voxel.crane_at(seed) puts it.
#   OUT=first SEEDS="0" RUNS="planner:survey planner:voxel learner:voxel" ./conditions/static_obstacles/voxfly.sh
# INSTANCE=N runs beside others: its own Gazebo partition, PX4 instance N (ports 14540+N), PX4 folder px4run_N
R=/home/sai/RL_projects/rl-landing; D=$R/outputs/gazebo_voxel/${OUT:?}
[ -n "$INSTANCE" ] && export PX4_INSTANCE=$INSTANCE
export GZ_PARTITION=voxfly${INSTANCE:-0}                   # always our own partition: how cleanup finds our Gazebo
# Cleanup however the script ends (25 Sept: five parallel runs left their Gazebo servers running, which shrug off
# a first SIGTERM; and one run died before its last line). PIDs and our partition only, never a name pattern.
cleanup () {
    kill $PX4PID 2>/dev/null
    mine () { for p in $(pgrep -f "gz si[m]"); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=$GZ_PARTITION" && echo $p; done; }
    for p in $(mine); do kill $p; done
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(mine)" ] && kill -0 $PX4PID 2>/dev/null || break; sleep 1; done
    for p in $(mine); do kill -9 $p; done; kill -9 $PX4PID 2>/dev/null
}
trap cleanup EXIT
cd $R && mkdir -p $D
# FAST=1: the voxfast world (no video cameras), no video, quick climb, flights end 15 m past the crane
# SHORT=1: the same short flights WITH video (voxfly world) -- for the films
# FULL=1 (with FAST=1): no video, but the whole delivery, landing on the pad
export PX4_SIM_SPEED_FACTOR=${SPEED:-0.1} WORLDS_DIR=$R/shared/px4_gazebo/gz_logan/$([ -n "$FAST" ] && echo voxfast || echo voxfly)${RIG:+_$RIG} EXTRA_MODELS=$R/shared/px4_gazebo/gz_logan/models
# Gazebo's GPS has no latency, but PX4 assumes the real receiver's 110 ms: its estimate then ran 0.11 s AHEAD of the
# drone -- 0.9 m at 8 m/s (measured 25 Sept, truth vs LOCAL_POSITION_NED: lag 108-128 ms, 0.06 m left once removed)
export PX4_PARAM_EKF2_GPS_DELAY=0
# RIG=v2: shared/px4_gazebo/gz_unified.py's rig (all cameras under the body; landing camera on PX4's CGO3 gimbal) -- the _v2 world, and
# PX4's gimbal module holding that camera straight down: MAVLink in (DO_MOUNT_CONTROL, sent after arming by
# px4_mission), AUX out = gimbal_controls, which PX4's gz bridge turns into joint angles, with the airframe's roll and
# pitch taken out (MNT_DO_STAB); pitch range +-135 deg as the joint's, so -90 +- a braking tilt still fits
case "$RIG" in v2|v2t|v2u|v2l) export PX4_PARAM_MNT_MODE_IN=4 PX4_PARAM_MNT_MODE_OUT=0 PX4_PARAM_MNT_DO_STAB=1 PX4_PARAM_MNT_RANGE_PITCH=270;; esac
[ -n "$RIG" ] && export RIG                                  # v2f: the fixed downward camera (gz_unified.VARIANTS)
export PX4_GZ_MODEL_NAME=x500_mono_cam_down_0           # PX4 attaches to the depth drone standing at the depot
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
CKPT=rl_corridor_eyes_lidar_gps_unmapped_s0_mean.pt      # the learner trained on the unmapped world (stage 3)
setsid ./shared/px4_gazebo/launch_px4.sh > $D/px4.log 2>&1 &                # its own session: PX4's shutdown can't reach this script
PX4PID=$!
# a busy instance (25 Sept: "PX4 server already running for instance 1") or a PX4 that never comes up must end the
# run, not leave it waiting for "home set" forever
for _ in $(seq 120); do grep -q "home set" $D/px4.log && break
    grep -q "already running" $D/px4.log && { echo "instance ${INSTANCE:-0} busy" >> $D/flights.err; exit 1; }; sleep 5; done
grep -q "home set" $D/px4.log || { echo "PX4 never set home" >> $D/flights.err; exit 1; }
source /opt/ros/jazzy/setup.bash 2>/dev/null
# attached to our drone PX4 skips the speed factor (conditions/magpie/rigfly.sh, 25 Sept): set it here
gz service -s /world/aruco/set_physics --reqtype gz.msgs.Physics --reptype gz.msgs.Boolean --timeout 5000 \
    --req "real_time_factor: $PX4_SIM_SPEED_FACTOR" > $D/speed.txt 2>&1
sleep 20
for s in ${SEEDS:?}; do for run in ${RUNS:?}; do
    d=${run%%:*}; m=${run##*:}
    n0=$(grep -c "Failsafe activated" $D/px4.log)
    TREES_JSON=shared/px4_gazebo/gz_logan/trees_logan.json timeout 3600 /usr/bin/python3 -u shared/px4_gazebo/gz_learner.py --decider $d --seed $s --magpie none \
        $([ "$m" != survey ] && echo --voxel) $([ "$m" = local3d ] && echo --local3d) \
        $([ "$d" = learner ] && echo --ckpt $CKPT) $([ -n "$FAST" ] && echo --fast) $([ -n "$SHORT" ] && echo --short) $([ -n "$FULL" ] && echo --full) \
        --video $D/${d}_${m}_s$s.mp4 \
        >> $D/flights.jsonl 2>> $D/flights.err
    echo "$d $m s$s: $(( $(grep -c "Failsafe activated" $D/px4.log) - n0 )) failsafes during the flight" >> $D/failsafes.txt
done; done
echo "$(date +%H:%M) voxfly $OUT done: $(grep -c '"outcome"' $D/flights.jsonl) flights" >> $R/outputs/queue/rig.log
