#!/usr/bin/env bash
# rigv2_detector.sh -- step 3 stage B (25 Sept): rig v2's labelled pictures (600 drone places, magpie around it; 200
# empty places over the pads), the YOLO fine-tune started from the first rig's (same bird, new cameras), its held-out
# score in the detector's own pixels, and the calibration the fast simulator flies with (urban/rig_calib_v2.json).
R=/home/sai/RL_projects/rl-landing; cd $R; export RIG=v2 YOLO_AUTOINSTALL=false
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python; D=gz_logan/rig_v2/data; L=gz_logan/rig_v2
log () { echo "$(date +%H:%M) rigv2 detector: $*" >> queue/rig.log; }
while kill -0 209893 2>/dev/null; do sleep 10; done      # the coverage bench (PID 209893) first: same partition
mkdir -p $D
G=$(/usr/bin/python3 gz_rig.py --bench)
trap 'kill -- -$G 2>/dev/null; sleep 3; kill -9 -- -$G 2>/dev/null' EXIT
/usr/bin/python3 -u gz_rig.py --data $D --poses 600 > $L/data.log 2>&1 || { log "data failed"; exit 1; }
/usr/bin/python3 -u gz_rig.py --negatives $D > $L/negatives.log 2>&1 || { log "negatives failed"; exit 1; }
kill -- -$G 2>/dev/null; sleep 3; kill -9 -- -$G 2>/dev/null
log "pictures: $(wc -l < $D/labels.jsonl) labelled"
PYTHONPATH=.yolo_pkgs $P rig_yolo.py --data $D > $L/yolo_build.log 2>&1 || { log "build failed"; exit 1; }
PYTHONPATH=.yolo_pkgs $P rig_yolo.py --train --epochs 30 --init $R/gz_logan/rig/yolo/run/weights/best.pt > $L/yolo_train.log 2>&1 || { log "train failed"; exit 1; }
PYTHONPATH=.yolo_pkgs $P rig_yolo.py --score $D > $L/yolo_score.log 2>&1 || { log "score failed"; exit 1; }
/usr/bin/python3 gz_rig.py --calib $L/cover $D --score score_yolo.json --out urban/rig_calib_v2.json > $L/calib.log 2>&1 || { log "calib failed"; exit 1; }
log "done: urban/rig_calib_v2.json"
