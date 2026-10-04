#!/usr/bin/env bash
# geofly.sh -- one GeoAnchor flight in Logan: PX4 SITL + Gazebo (a photo-floor world, geo/geo_world.py), GeoAnchor's own
# three layers (geoanchor-rt, unmodified) matching the live gimbal camera, a uXRCE-DDS agent and geo/agp_sim_bridge.py
# feeding PX4's EKF2 over AGP, then one flight command. Never GeoAnchor's run.sh (it syncs every run to the logs repo).
#   OUT=smoke WORLD=geo_clean INSTANCE=5 geo/geofly.sh /usr/bin/python3 -u geo/geo_cruise.py
# WORLD: geo_clean (no unmapped obstacles) or geo_floor (stage 3's crane, lights, trees, shed round the pad).
# SERVER_CONFIG: the WindDrag wind (wind/gz_wind.py). AGP_DELAY: EKF2_AGP_DELAY, ms (the fix's age on arrival, sim time).
# Its own DDS port (8890+N), ROS domain (40+N) and GeoAnchor bus, so GeoAnchor's own setup (8888) is never touched.
R=/home/sai/RL_projects/rl-landing; D=$R/gazebo_geo/${OUT:?}; GA=/home/sai/GeoAnchor/geoanchor-rt
I=${INSTANCE:-5}
mkdir -p $D/geoanchor /tmp/geoanchor_rl/$I
export PX4_INSTANCE=$I GZ_PARTITION=geo$I ROS_DOMAIN_ID=$((40 + I)) PX4_UXRCE_DDS_PORT=$((8890 + I))
PIDS=()
cleanup () {
    kill "${PIDS[@]}" $PX4PID 2>/dev/null
    mine () { for p in $(pgrep -f "gz si[m]"); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=$GZ_PARTITION" && echo $p; done; }
    for p in $(mine); do kill $p; done
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -z "$(mine)" ] && kill -0 $PX4PID 2>/dev/null || break; sleep 1; done
    for p in $(mine); do kill -9 $p; done; kill -9 "${PIDS[@]}" $PX4PID 2>/dev/null
}
trap cleanup EXIT
trap 'exit 143' INT TERM
cd $R
# PX4 + Gazebo as wind/sitl.sh, and EKF2 fusing GeoAnchor over AGP (geoanchor-rt/configs/px4_agp.params' values)
export PX4_SIM_SPEED_FACTOR=${SPEED:-1.0} WORLDS_DIR=$R/gz_logan/${WORLD:-geo_clean} EXTRA_MODELS=$R/gz_logan/models
export PX4_PARAM_EKF2_GPS_DELAY=0 PX4_PARAM_EKF2_NOAID_TOUT=10000000   # 10 s (its max) of dead reckoning between fixes
export PX4_PARAM_EKF2_RNG_A_HMAX=30 PX4_PARAM_EKF2_RNG_A_VMAX=2   # the rangefinder (x500_unified_rng) for the landing: from 30 m
export PX4_PARAM_EKF2_AGP_CTRL=1 PX4_PARAM_EKF2_AGP_NOISE=0.1 PX4_PARAM_EKF2_AGP_GATE=3.0 PX4_PARAM_EKF2_AGP_DELAY=${AGP_DELAY:-100}
export PX4_PARAM_MNT_MODE_IN=4 PX4_PARAM_MNT_MODE_OUT=0 PX4_PARAM_MNT_DO_STAB=1 PX4_PARAM_MNT_RANGE_PITCH=270 RIG=${RIG:-v2}
export PX4_GZ_MODEL_NAME=x500_mono_cam_down_0
export EXTRA_PLUGINS=$R/wind/gz_winddrag/build
[ -n "$SERVER_CONFIG" ] && export SERVER_CONFIG
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
setsid ./launch_px4.sh > $D/px4.log 2>&1 &
PX4PID=$!
for _ in $(seq 120); do grep -q "home set" $D/px4.log && break
    grep -q "already running" $D/px4.log && { echo "instance $I busy" >> $D/run.err; exit 1; }; sleep 5; done
grep -q "home set" $D/px4.log || { echo "PX4 never set home" >> $D/run.err; exit 1; }
source /opt/ros/jazzy/setup.bash 2>/dev/null
gz service -s /world/aruco/set_physics --reqtype gz.msgs.Physics --reptype gz.msgs.Boolean --timeout 5000 \
    --req "real_time_factor: $PX4_SIM_SPEED_FACTOR" > $D/speed.txt 2>&1
# GeoAnchor on the live gimbal camera; the vehicle state (height, heading) over PX4's GCS link, 18570+N
B=ipc:///tmp/geoanchor_rl/$I
export GEOANCHOR_CONFIG=$R/geo/geo_logan_replay.yaml GEOANCHOR_RUN_DIR=$D/geoanchor GEOANCHOR_SYNC_ON_EXIT=0
export GEOANCHOR_SET="data_layer.feed.type=gz;data_layer.feed.topic=/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image;\
data_layer.gps.source=mavlink;data_layer.gps.endpoint=udpout:127.0.0.1:$((18570 + I));\
bus.data_pub=$B/data.sock;bus.processing_pub=$B/processing.sock;bus.output_pub=$B/output.sock;\
bus.data_ctl=$B/data.ctl;bus.processing_ctl=$B/processing.ctl;bus.output_ctl=$B/output.ctl"
for layer in output processing data; do
    (cd $GA && exec .venv/bin/python -m geoanchor.${layer}_layer) > $D/ga_$layer.out 2>&1 &
    PIDS+=($!); sleep 3
done
MicroXRCEAgent udp4 -p $PX4_UXRCE_DDS_PORT > $D/xrce.log 2>&1 &
PIDS+=($!)
bash -c "source /home/sai/GeoAnchor/ros2_ws/install/setup.bash; exec python3 -u $R/geo/agp_sim_bridge.py \
    --config $GEOANCHOR_CONFIG --ns px4_$I --log $D/agp.jsonl --marker $B/marker.sock" > $D/bridge.out 2>&1 &
PIDS+=($!)
sleep 5
GEO_OUT=$D GEO_AGP_LOG=$D/agp.jsonl GEO_MARKER_PUB=$B/marker.sock MAV_PORT=$((14540 + I)) "$@" >> $D/run.log 2>> $D/run.err
RC=$?
cp $R/px4run_$I/px4run/log/*/*.ulg $D/ 2>/dev/null     # PX4's own flight log (QGC's plots; the dashboard's Flight log)
exit $RC
