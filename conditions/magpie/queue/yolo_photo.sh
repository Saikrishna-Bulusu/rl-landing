#!/usr/bin/env bash
# yolo_photo.sh -- the rig's magpie detector retrained on PHOTO ground (27 Sept, overnight). Sai: "the cameras kept
# detecting trees as magpies". Every training picture so far was plain Gazebo ground; over the photo floor it boxed
# crowns and roofs. One Gazebo at a time: nothing else may fly while step 2 runs.
#   1. the photo bench: geo_magpie's floor, roofs and crowns draped with it, the magpie; rig v2 standing static
#   2. 800 places of labelled pictures (the magpie 2-70 m away; two empty views per place: the trees as negatives)
#   3. the YOLO set; the weights in use fine-tuned on plain + photo ground, 30 epochs -> yolo/run_geo (yolo/run untouched)
#   4. scored on the photo pictures' held-out places (old and new weights) and the plain ones (new): found / false alarms
R=/home/sai/RL_projects/rl-landing; cd $R; Q=$R/queue
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export RIG=v2 RIG_BENCH=rig_v2_geo PYTHONPATH=$R/.yolo_pkgs YOLO_AUTOINSTALL=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log () { echo "$(date +%H:%M) $*" >> $Q/yolo_photo.log; }
D=gz_logan/rig_v2_geo/data; Y=gz_logan/rig_v2/yolo; O=gz_logan/rig_v2_geo
bench_down () {
    kill $BPID 2>/dev/null; sleep 5
    for p in $(pgrep -f "gz si[m].*rig_v2_geo"); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=rig2" && kill $p; done
}
log "start"
/usr/bin/python3 gz_rig.py --photo-bench > /dev/null
BPID=$(/usr/bin/python3 gz_rig.py --bench | tail -1); log "bench up (pid $BPID)"
source /opt/ros/jazzy/setup.bash 2>/dev/null
until [ "$(GZ_PARTITION=rig2 timeout 10 gz topic -l 2>/dev/null | grep -c '^/rig/')" -ge 5 ]; do sleep 5; done
/usr/bin/python3 -u gz_rig.py --data $D --poses 800 > $O/data.log 2>&1 || { log "FAILED data"; bench_down; exit 1; }
bench_down
log "data: $(wc -l < $D/labels.jsonl) pictures"
$P rig_yolo.py --data $D --ds ds_geo > $O/yolo_build.log 2>&1 || { log "FAILED build"; exit 1; }
$P rig_yolo.py --score $D --weights $R/$Y/run/weights/best.pt --tag _yolo_old > $O/score_old.log 2>&1 || log "FAILED score old"
$P rig_yolo.py --train --ds ds,ds_geo --name run_geo --epochs 30 --init $R/$Y/run/weights/best.pt > $O/yolo_train.log 2>&1 \
    || { log "FAILED train"; exit 1; }
$P rig_yolo.py --score $D --weights $R/$Y/run_geo/weights/best.pt --tag _yolo_new > $O/score_new.log 2>&1 || log "FAILED score new"
$P rig_yolo.py --score gz_logan/rig_v2/data --weights $R/$Y/run_geo/weights/best.pt --tag _yolo_new > $O/score_new_plain.log 2>&1 \
    || log "FAILED score new (plain)"
log "done: $(python3 - <<'PY'
import json
row = lambda f, k="0.5": (lambda d: f"found {100 * d['by_threshold'][k]['found']:.0f}%, false alarms {d['by_threshold'][k]['false_alarms_per_image']:.3f}/picture")(json.load(open(f)))
out = []
for tag, f in (("photo, old weights", "gz_logan/rig_v2_geo/data/score_yolo_old.json"), ("photo, new", "gz_logan/rig_v2_geo/data/score_yolo_new.json"),
               ("plain, old", "gz_logan/rig_v2/data/score_yolo.json"), ("plain, new", "gz_logan/rig_v2/data/score_yolo_new.json")):
    try:
        out.append(f"{tag}: {row(f)}")
    except Exception as e:
        out.append(f"{tag}: -- ({e.__class__.__name__})")
print(" | ".join(out))
PY
)"
