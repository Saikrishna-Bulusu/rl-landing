#!/usr/bin/env bash
# v2l_bench.sh -- the up camera with the back camera left level: coverage bench, then its calibration (rig v2's detector
# score, its own coverage and masks) and the fast simulator's magpie ladder after the v2t/v2u ladders.
R=/home/sai/RL_projects/rl-landing; Q=$R/queue; P=/home/sai/miniconda3/envs/mujoco_drones/bin/python; cd $R
log () { echo "$(date +%H:%M) v2l: $*" >> $Q/rig.log; }
O=gz_logan/rig_v2l/cover; mkdir -p $O
G=$(RIG=v2l /usr/bin/python3 gz_rig.py --bench)
RIG=v2l /usr/bin/python3 -u gz_rig.py --cover $O > gz_logan/rig_v2l/cover.log 2>&1 && \
RIG=v2l /usr/bin/python3 -u gz_rig.py --report $O >> gz_logan/rig_v2l/cover.log 2>&1
kill -- -$G 2>/dev/null; sleep 3; kill -9 -- -$G 2>/dev/null
log "cover: $(grep 'camera model check' gz_logan/rig_v2l/cover.log | cut -c1-90)"
RIG=v2l /usr/bin/python3 gz_rig.py --calib $O gz_logan/rig_v2/data --score score_yolo.json --out urban/rig_calib_v2l.json \
    > gz_logan/rig_v2l/calib.log 2>&1 || { log "calib failed"; exit 1; }
while pgrep -f "urban_headroom[.]py .*--rig --tag _v2[tu]" > /dev/null; do sleep 30; done   # one ladder at a time (GPU)
cd urban
RIG_CALIB=$R/urban/rig_calib_v2l.json PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P urban_headroom.py --map logan_map_lidar.npz \
    --nav gps --birds --hunter --rig --tag _v2l > headroom_lidar_gps_birds_hunter_rig_v2l.log 2>&1 || { log "ladder failed"; exit 1; }
log "ladder: $(grep -m2 -A2 'corridor world' headroom_lidar_gps_birds_hunter_rig_v2l.log | grep -m1 rule | tr -s ' ' | cut -c1-110)"
