#!/usr/bin/env bash
# replay.sh -- GeoAnchor's own three layers (geoanchor-rt, unmodified) on one recorded sweep, with our config and run
# dir and no log sync: run.sh pushes every run to the GeoAnchor logs repo on exit, so it is never called here.
#     geo/replay.sh gazebo_geo/floor floor      -> geo/runs/floor_<stamp>/records.jsonl + the error summary
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
GA=/home/sai/GeoAnchor/geoanchor-rt
PY=$GA/.venv/bin/python
run=$R/geo/runs/${2}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$run" /tmp/geoanchor_rl
sed "s#/gazebo_geo/floor/#/$1/#" "$R/geo/geo_logan_replay.yaml" > "$run/config.yaml"
grep -q "/$1/flight.mp4" "$run/config.yaml" || { echo "feed path not set in $run/config.yaml"; exit 1; }
export GEOANCHOR_CONFIG=$run/config.yaml GEOANCHOR_RUN_DIR=$run GEOANCHOR_SYNC_ON_EXIT=0
cd "$GA"
$PY -m geoanchor.output_layer > "$run/output.out" 2>&1 & po=$!
$PY -m geoanchor.processing_layer > "$run/processing.out" 2>&1 & pp=$!
trap 'kill -TERM $po $pp 2>/dev/null' EXIT INT TERM
sleep 5
$PY -m geoanchor.data_layer > "$run/data.out" 2>&1          # returns at the end of the feed
# the last frames are through the matcher once the records stop growing
n=-1; while [ "$(cat "$run/records.jsonl" 2>/dev/null | wc -l)" != "$n" ]; do n=$(cat "$run/records.jsonl" 2>/dev/null | wc -l); sleep 10; done
kill -TERM $po $pp; wait $po $pp 2>/dev/null
echo "$run"
$PY "$R/geo/score.py" "$run" "$R/$1/flight_gt.jsonl"
