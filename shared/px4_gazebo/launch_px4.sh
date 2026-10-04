#!/usr/bin/env bash
# Launch PX4 SITL + Gazebo: x500 with a DOWNWARD camera, in a world that
# already contains an ArUco tag.
#
#   airframe 4014_gz_x500_mono_cam_down
#   world    aruco.sdf   (model://arucotag, 0.5 x 0.5 m)
#   camera   99.7 deg FOV, 1280x960, 30 Hz, pitched 90 deg straight down
#
# TWO THINGS HAD TO BE FIXED TO MAKE THIS BOOT:
#
# 1. PX4 composes the world path as "${PX4_GZ_WORLDS}/${PX4_GZ_WORLD}.sdf"
#    (ROMFS/px4fmu_common/init.d-posix/px4-rc.gzsim:52). With PX4_GZ_WORLDS
#    unset that becomes "/aruco.sdf" and gz reports
#    "Failed to find world [/aruco.sdf]" then waits forever.
#
# 2. The generated build/px4_sitl_default/rootfs/gz_env.sh -- which is supposed
#    to set those variables -- points at /home/sai/thesis/PX4-Autopilot, a path
#    that NO LONGER EXISTS (the repo now lives at /home/sai/thesis_2.0). The
#    build was configured when the source was somewhere else. Sourcing it would
#    silently set the wrong directories, so the values are set explicitly here.
#    Rebuilding PX4 from the current path would regenerate it correctly.
P=/home/sai/thesis_2.0/PX4-Autopilot
S=/home/sai/RL_projects/rl-landing/px4run${PX4_INSTANCE:+_$PX4_INSTANCE}   # PX4_INSTANCE=N: several SITLs side by side

source /opt/ros/jazzy/setup.bash 2>/dev/null

export PX4_GZ_MODELS="$P/Tools/simulation/gz/models"
# WORLDS_DIR / EXTRA_MODELS: another world under the same name (e.g. gz_logan/, built
# from the Logan LiDAR by urban/gz_logan.py) without touching the PX4 tree.
export PX4_GZ_WORLDS="${WORLDS_DIR:-$P/Tools/simulation/gz/worlds}"
export PX4_GZ_PLUGINS="$P/build/px4_sitl_default/src/modules/simulation/gz_plugins"
export PX4_GZ_SERVER_CONFIG="$P/src/modules/simulation/gz_bridge/server.config"
export GZ_SIM_RESOURCE_PATH="$PX4_GZ_MODELS:$PX4_GZ_WORLDS${EXTRA_MODELS:+:$EXTRA_MODELS}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PX4_GZ_PLUGINS${EXTRA_PLUGINS:+:$EXTRA_PLUGINS}"   # wind/sitl.sh: the WindDrag plugin
# SERVER_CONFIG: a sky from urban/gz_weather.py (wind gusts, particles) -- plugins live
# there because gz loads the server config only when the world declares no plugins.
export GZ_SIM_SERVER_CONFIG_PATH="${SERVER_CONFIG:-$PX4_GZ_SERVER_CONFIG}"

export PX4_GZ_WORLD=aruco
export PX4_SYS_AUTOSTART=4014
export PX4_SIM_MODEL=gz_x500_mono_cam_down
export HEADLESS=1

# START FROM A CLEAN STATE EVERY TIME.
#
# PX4 SITL persists its parameters in px4run/parameters.bson and reloads them
# on boot. After a few dozen restarts that accumulated state began failing the
# EKF magnetometer check -- "Preflight Fail: Strong magnetic interference" --
# and arming was refused for every mission, including the plain bridge check.
# Nothing in the world or the code had changed. Wiping the directory fixed it
# immediately. There is no SITL state worth carrying between sessions here.
rm -rf "$S/px4run"
mkdir -p "$S/px4run"
cd "$S/px4run" || exit 1
exec "$P/build/px4_sitl_default/bin/px4" "$P/build/px4_sitl_default/etc" \
     -s etc/init.d-posix/rcS -d ${PX4_INSTANCE:+-i $PX4_INSTANCE}
