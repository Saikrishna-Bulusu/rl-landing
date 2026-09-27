#!/usr/bin/env bash
# rigfly.sh -- PX4's rig drone in Gazebo against the hunter magpie (from behind, steering). With
# SENSE=rig the decider knows the magpie ONLY through the rig: the detector on the rig cameras'
# pictures and the tracker (urban/rig_perceive.py) -- no true bird position anywhere in the loop.
# SENSE=omni hands it the truth (the old assumption), for comparison on the same seeds.
#   OUT=planner_rig SEEDS="0 1 2 3" DECIDERS=planner SENSE=rig ./rigfly.sh
#   OUT=learner_rig ... DECIDERS=learner CKPT=rl_corridor_eyes_lidar_gps_birds_hunter_rig_s0_mean.pt
# RIG=v2: gz_unified.py's rig and its detector (gz_logan/rig_v2/yolo); INSTANCE=N runs beside others (PX4 -i N,
# MAVLink 14540+N, its own Gazebo partition); SPEED: the simulation speed (0.1 for the six HD cameras).
R=/home/sai/RL_projects/rl-landing; D=$R/gazebo_rig/${OUT:?}
cd $R && mkdir -p $D
[ -n "$INSTANCE" ] && export PX4_INSTANCE=$INSTANCE
export GZ_PARTITION=rigfly${INSTANCE:-0}                   # our own partition: how cleanup finds our Gazebo
cleanup () {                                               # however the script ends (voxfly.sh, 25 Sept)
    kill $PX4PID 2>/dev/null
    mine () { for p in $(pgrep -f "gz si[m]"); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=$GZ_PARTITION" && echo $p; done; }
    for p in $(mine); do kill $p; done
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(mine)" ] && kill -0 $PX4PID 2>/dev/null || break; sleep 1; done
    for p in $(mine); do kill -9 $p; done; kill -9 $PX4PID 2>/dev/null
}
trap cleanup EXIT
# Gazebo's GPS has no latency, but PX4 assumes the real receiver's 110 ms: its estimate then ran 0.11 s AHEAD of the
# drone -- 0.9 m at 8 m/s (measured 25 Sept, truth vs LOCAL_POSITION_NED: lag 108-128 ms, 0.06 m left once removed)
export PX4_PARAM_EKF2_GPS_DELAY=0
# RIG=v2: gz_unified.py's rig (all cameras under the body; landing camera on PX4's CGO3 gimbal) -- the _v2 world, and
# PX4's gimbal module holding that camera straight down: MAVLink in (DO_MOUNT_CONTROL, sent every step by
# px4_mission), AUX out = gimbal_controls, which PX4's gz bridge turns into joint angles, with the airframe's roll and
# pitch taken out (MNT_DO_STAB); pitch range +-135 deg as the joint's, so -90 +- a braking tilt still fits
case "$RIG" in v2|v2t|v2u|v2l) export PX4_PARAM_MNT_MODE_IN=4 PX4_PARAM_MNT_MODE_OUT=0 PX4_PARAM_MNT_DO_STAB=1 PX4_PARAM_MNT_RANGE_PITCH=270;; esac
case "$RIG" in v2*) export RIG RIG_YOLO=${RIG_YOLO:-$R/gz_logan/rig_$RIG/yolo/run/weights/best.pt} \
                          RIG_CALIB=${RIG_CALIB:-$R/urban/rig_calib_$RIG.json};; esac
export PX4_SIM_SPEED_FACTOR=${SPEED:-0.1} WORLDS_DIR=$R/gz_logan/rigfly${RIG:+_$RIG} EXTRA_MODELS=$R/gz_logan/models
export PX4_GZ_MODEL_NAME=x500_mono_cam_down_0          # PX4 attaches to the rig drone standing at the depot
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
setsid ./launch_px4.sh > $D/px4.log 2>&1 &              # its own session: PX4's shutdown can't reach this script
PX4PID=$!
for _ in $(seq 120); do grep -q "home set" $D/px4.log && break
    grep -q "already running" $D/px4.log && { echo "instance ${INSTANCE:-0} busy" >> $D/flights.err; exit 1; }; sleep 5; done
grep -q "home set" $D/px4.log || { echo "PX4 never set home" >> $D/flights.err; exit 1; }
# PX4 sets the simulation speed only when it SPAWNS its drone; attached to ours it skipped it, the
# world ran several times faster than PX4's offboard timeouts assume (set for 0.1x), setpoints
# arrived "late" in simulated time and PX4's failsafe landed the drone where it was (25 Sept, the
# first two rig flights: ~250 "Failsafe activated"). So set it here, and count failsafes per flight.
source /opt/ros/jazzy/setup.bash 2>/dev/null
gz service -s /world/aruco/set_physics --reqtype gz.msgs.Physics --reptype gz.msgs.Boolean --timeout 5000 \
    --req "real_time_factor: $PX4_SIM_SPEED_FACTOR" > $D/speed.txt 2>&1
sleep 20
for s in ${SEEDS:?}; do for d in ${DECIDERS:?}; do
    n0=$(grep -c "Failsafe activated" $D/px4.log)
    TREES_JSON=gz_logan/trees_logan.json timeout 3600 /usr/bin/python3 -u gz_learner.py --decider $d --seed $s \
        --magpie hunter --sense ${SENSE:?} ${CKPT:+--ckpt $CKPT} --video $D/${d}_s$s.mp4 >> $D/flights.jsonl 2>> $D/flights.err
    echo "$d s$s: $(( $(grep -c "Failsafe activated" $D/px4.log) - n0 )) failsafes during the flight (0-2 at its end are disarm)" >> $D/failsafes.txt
done; done
echo "$(date +%H:%M) rigfly $OUT done: $(grep -c '"outcome"' $D/flights.jsonl) flights" >> $R/queue/rig.log
