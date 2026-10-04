#!/usr/bin/env bash
# run_weather.sh SKY [N] -- fly N Logan deliveries under one sky from urban/gz_weather.py.
#
# One mission per process (run_forest20.sh explains why); seeds 0..N-1 in every sky, so
# the skies pair flight for flight. Flight 0 is also recorded to video. Writes
# weather_runs/SKY/{m_i.json, trace_i.csv, missions.log, px4.log, rtf.txt}, then
# queue/SKY.done -- the next sky's queue item waits for that file.
# Summarise with: python urban/gz_weather.py report
R=/home/sai/RL_projects/rl-landing
SKY=$1; N=${2:-10}; OUT=$R/weather_runs/$SKY
export PATH=/opt/ros/jazzy/opt/gz_tools_vendor/bin:$PATH
source /opt/ros/jazzy/setup.bash 2>/dev/null
set -u                       # only after setup.bash, which reads unset variables
mkdir -p "$OUT" "$R/queue"; cd "$R" || exit 1

pkill -f "bin/p[x]4 "; pkill -f "gz si[m]"; sleep 6
WORLDS_DIR=$R/gz_logan/weather/$SKY EXTRA_MODELS=$R/gz_logan/models \
SERVER_CONFIG=$R/gz_logan/weather/$SKY/server.config \
    nohup ./launch_px4.sh > "$OUT/px4.log" 2>&1 &
for _ in $(seq 1 60); do grep -q "home set" "$OUT/px4.log" 2>/dev/null && break; sleep 3; done
sleep 8
# Real-time factor while flying: the mission loop runs on wall clock, so a stalled
# simulator changes the flight. Every 20 s, for the record.
( while pgrep -f "bin/p[x]4 " >/dev/null; do
      timeout 5 gz topic -e -n 1 -t /world/aruco/stats 2>/dev/null | grep -oE "real_time_factor: [0-9.]+"
      sleep 20
  done ) > "$OUT/rtf.txt" 2>&1 &

for i in $(seq 0 $((N - 1))); do
    vid=(); [ "$i" = 0 ] && vid=(--video "$OUT/flight0.mp4")
    MISSION_TRACE=$OUT/trace_$i.csv TREES_JSON=gz_logan/trees_logan.json MISSION_JSON=$OUT/m_$i.json \
        timeout 480 /usr/bin/python3 -u px4_mission.py --runs 1 --seed0 "$i" "${vid[@]}" \
        >> "$OUT/missions.log" 2>&1 || echo "flight $i exited $?" >> "$OUT/missions.log"
done
pkill -f "bin/p[x]4 "; pkill -f "gz si[m]"
echo "$SKY: $(grep -c '  =>' "$OUT/missions.log") of $N flights recorded" > "$R/queue/$SKY.done"
