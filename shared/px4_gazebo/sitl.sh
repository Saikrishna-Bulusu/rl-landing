#!/usr/bin/env bash
# shared/px4_gazebo/sitl.sh -- PX4 SITL + Gazebo with the rig v2 drone (voxfast_v2 world: no video cameras), then one command.
#   OUT=steps ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u shared/px4_gazebo/px4_steps.py --out outputs/gazebo_wind/steps/steps.csv
# WORLD=<dir under gz_logan> (default voxfast_v2); SERVER_CONFIG=<file> (the wind plugin); SPEED (default 1.0); KEEP_ULG=1 keeps PX4's log;
# INSTANCE=N runs beside others (PX4 -i N, MAVLink 14540+N, partition windN). Cleanup as conditions/static_obstacles/voxfly.sh.
R=/home/sai/RL_projects/rl-landing; D=$R/outputs/gazebo_wind/${OUT:?}
# absolute: PX4 starts gz from its run folder, where a relative config is silently not found (no scene, no wind; 6 Oct)
[ -n "$SERVER_CONFIG" ] && { SERVER_CONFIG=$(realpath -e "$SERVER_CONFIG") || exit 1; }
[ -n "$INSTANCE" ] && export PX4_INSTANCE=$INSTANCE
export GZ_PARTITION=wind${INSTANCE:-0}
cleanup () {
    kill $PX4PID 2>/dev/null
    mine () { for p in $(pgrep -f "gz si[m]"); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=$GZ_PARTITION" && echo $p; done; }
    for p in $(mine); do kill $p; done
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(mine)" ] && kill -0 $PX4PID 2>/dev/null || break; sleep 1; done
    for p in $(mine); do kill -9 $p; done; kill -9 $PX4PID 2>/dev/null
    [ -n "$KEEP_ULG" ] && cp $R/outputs/px4run${INSTANCE:+_$INSTANCE}/px4run/log/*/*.ulg $D/ 2>/dev/null  # launch_px4.sh wipes them; ~30 MB/flight
}
trap cleanup EXIT
trap 'exit 143' INT TERM              # a kill runs cleanup too (26 Sept: a stopped ladder left 4 Gazebos + 3 PX4s)
cd $R && mkdir -p $D
export PX4_SIM_SPEED_FACTOR=${SPEED:-1.0} WORLDS_DIR=$R/shared/px4_gazebo/gz_logan/${WORLD:-voxfast_v2} EXTRA_MODELS=$R/shared/px4_gazebo/gz_logan/models
export PX4_PARAM_EKF2_GPS_DELAY=0                              # Gazebo's GPS has no latency (conditions/static_obstacles/voxfly.sh)
export PX4_PARAM_MNT_MODE_IN=4 PX4_PARAM_MNT_MODE_OUT=0 PX4_PARAM_MNT_DO_STAB=1 PX4_PARAM_MNT_RANGE_PITCH=270 RIG=${RIG:-v2}
export PX4_GZ_MODEL_NAME=x500_mono_cam_down_0
export EXTRA_PLUGINS=$R/conditions/wind/gz_winddrag/build             # WindDrag, used when SERVER_CONFIG names it
[ -n "$SERVER_CONFIG" ] && export SERVER_CONFIG
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
setsid ./shared/px4_gazebo/launch_px4.sh > $D/px4.log 2>&1 &
PX4PID=$!
for _ in $(seq 120); do grep -q "home set" $D/px4.log && break
    grep -q "already running" $D/px4.log && { echo "instance ${INSTANCE:-0} busy" >> $D/run.err; exit 1; }; sleep 5; done
grep -q "home set" $D/px4.log || { echo "PX4 never set home" >> $D/run.err; exit 1; }
source /opt/ros/jazzy/setup.bash 2>/dev/null
gz service -s /world/aruco/set_physics --reqtype gz.msgs.Physics --reptype gz.msgs.Boolean --timeout 5000 \
    --req "real_time_factor: $PX4_SIM_SPEED_FACTOR" > $D/speed.txt 2>&1
sleep 10
MAV_PORT=$((14540 + ${INSTANCE:-0})) "$@" >> $D/run.log 2>> $D/run.err
