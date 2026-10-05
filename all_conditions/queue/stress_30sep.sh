#!/usr/bin/env bash
# stress_30sep.sh -- the full-stack stress test (Sai, 30 Sept, from the mock review's list): GNSS denied once the visual
# fixes are in PX4's loop, WindDrag 9 m/s from a random direction, the crane on the approach at a random spot, the
# hunter magpie seen only through rig v2l's live YOLO and tracker, the magpie learner (probability-weighted mean) deciding
# on the voxel map with the local 3D planner, the marker landing. No flying learner (Sai). A repeatable integration
# benchmark, not a landing rate. One SITL at a time, INSTANCE 5.
#   all_conditions/queue/stress_30sep.sh 0              the smoke flight
#   all_conditions/queue/stress_30sep.sh 1 2 3 ... 9    the rest; touches outputs/queue/stress_30sep.done after the last one
# Waits for the wind session's Gazebo ladder (its PID, then outputs/queue/wind_gz2_30sep.done). Every flight: outputs/gazebo_geo/stress_e<N>/,
# with manifest.json (git commit and SHA-256 of the code, checkpoint, detector, world and wind config it flew).
R=/home/sai/RL_projects/rl-landing; cd $R; Q=$R/outputs/queue; LOG=$Q/stress_30sep.log
log () { echo "$(date +%H:%M) $*" >> $LOG; }
WIND_PID=${WIND_PID:-2337048}
log "queued: e$*"
while kill -0 $WIND_PID 2>/dev/null || [ ! -e $Q/wind_gz2_30sep.done ]; do sleep 60; done
until [ -z "$(pgrep -f 'gz si[m]')" ]; do sleep 30; done                      # nothing else flying
log "start: $(free -g | awk '/Mem/ {print $7}') GB free, disk $(df -h / | awk 'NR==2 {print $4}'), AC $(cat /sys/class/power_supply/ADP0/online)"
CKPT=rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt; YOLO=shared/px4_gazebo/gz_logan/rig_v2/yolo/run_geo/weights/best.pt
FROZEN="shared/px4_gazebo/gz_learner.py shared/px4_gazebo/px4_mission.py shared/px4_gazebo/gz_rig.py shared/px4_gazebo/gz_voxel.py gnss_denied/geo_cruise.py gnss_denied/geofly.sh gnss_denied/agp_sim_bridge.py shared/fast_sim/decider_server.py
        shared/fast_sim/urban_agents.py shared/fast_sim/urban_rl.py shared/fast_sim/urban_env.py shared/fast_sim/local_planner.py shared/fast_sim/$CKPT $YOLO shared/fast_sim/rig_calib_v2l.json
        shared/px4_gazebo/gz_logan/trees_logan.json shared/px4_gazebo/gz_logan/geo_stress_v2l/aruco.sdf shared/px4_gazebo/gz_logan/models/x500_unified_v2l_rng/model.sdf"
for E in "$@"; do
    OUT=stress_e$E; D=$R/outputs/gazebo_geo/$OUT
    if [ -e $D/run.log ] && grep -q '"outcome"' $D/run.log; then log "e$E already flown"; continue; fi
    mkdir -p $D
    # the crane: on the last leg's line (geo_cruise ROUTE[-1] to PAD), 30-60 m from the pad, up to 1.5 m off it, on
    # ground the survey shows clear (gz_voxel.crane_at's own check); geo_obst5's was 36 m out, on the line
    CRANE=$(python3 - $E <<'PY'
import os, sys
import numpy as np
sys.path.insert(0, "shared/px4_gazebo")
import gz_voxel as GV
e = int(sys.argv[1]); pad, ref = np.array([120.0, 0.0]), np.array([260.0, -120.0])
u = (ref - pad) / np.linalg.norm(ref - pad); n = np.array([-u[1], u[0]])
rng = np.random.default_rng(2000 + e)
while True:
    x, y = pad + rng.uniform(30.0, 60.0) * u + rng.uniform(-1.5, 1.5) * n
    os.environ["CRANE_AT"] = f"{x:.2f},{y:.2f}"
    try:
        GV.crane_at(0); print(os.environ["CRANE_AT"]); break
    except AssertionError:
        pass
PY
)
    DIR=$(python3 -c "import numpy as np; print(round(float(np.random.default_rng(3000 + $E).uniform(0.0, 360.0)), 1))")
    sed -e "s|<direction_deg>0.0</direction_deg>|<direction_deg>$DIR</direction_deg>|" -e "s|<seed>0</seed>|<seed>$E</seed>|" \
        shared/px4_gazebo/gz_logan/geo_clean/geo_w9.config > $D/wind.config
    if [ -z "$CRANE" ] || ! grep -q "<direction_deg>$DIR</direction_deg>" $D/wind.config || ! grep -q "<seed>$E</seed>" $D/wind.config \
       || ! grep -q "<mean_speed>9.0</mean_speed>" $D/wind.config; then log "e$E setup FAILED (crane '$CRANE', wind $DIR)"; exit 1; fi
    python3 - "$D" "$E" "$CRANE" "$DIR" $FROZEN $D/wind.config <<'PY'
import hashlib, json, subprocess, sys
d, e, crane, wdir, files = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5:]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
json.dump(dict(seed=e, crane_at=[float(v) for v in crane.split(",")], wind_mean_ms=9.0, wind_dir_deg=wdir, wind_seed=e,
               commit=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain", "--", *files[:-1]),
               sha256={f.replace(d + "/", ""): sha(f) for f in files}), open(f"{d}/manifest.json", "w"), indent=1)
PY
    # never on battery (30 Sept: the smoke run found the laptop unplugged; 4 h of Gazebo and YOLO drain it in under 1 h).
    # rig_perceive.py runs in shared/fast_sim/, so RIG_YOLO and RIG_CALIB must be absolute (the smoke run died on a relative one)
    if [ -z "$NO_AC_WAIT" ] && [ "$(cat /sys/class/power_supply/ADP0/online)" != 1 ]; then log "e$E waiting for AC power"   # NO_AC_WAIT=1: fly anyway
        until [ "$(cat /sys/class/power_supply/ADP0/online)" = 1 ]; do sleep 60; done; log "AC back"; fi
    log "e$E start: crane $CRANE, wind 9 m/s, direction $DIR deg"
    T0=$(date +%s)
    TREES_JSON=shared/px4_gazebo/gz_logan/trees_logan.json RIG=v2l RIG_YOLO=$R/$YOLO RIG_CALIB=$R/shared/fast_sim/rig_calib_v2l.json GEO_LAND_SPEED=0.1 \
    GEO_GOAL_OUT=90 CRANE_AT=$CRANE SERVER_CONFIG=$D/wind.config OUT=$OUT WORLD=geo_stress_v2l INSTANCE=5 \
        timeout 3600 gnss_denied/geofly.sh /usr/bin/python3 -u shared/px4_gazebo/gz_learner.py --geo --decider learner --seed $E --magpie hunter \
        --sense rig --voxel --local3d --ckpt $CKPT --video $D/stress.mp4
    RC=$?
    echo "{\"wall_s\": $(( $(date +%s) - T0 )), \"rc\": $RC}" > $D/wall.json
    log "e$E done, rc $RC, $(( ($(date +%s) - T0) / 60 )) min: $(grep -o '"outcome": "[a-z_]*"' $D/run.log | tail -1)"
    sleep 20
done
[ "$*" != "0" ] && touch $Q/stress_30sep.done
log "queue finished: e$*"
