#!/usr/bin/env bash
# variants.sh -- Sai, 25 Sept evening: "Was adding a gimbal the wrong choice? ... Try with the back camera tilted up
# about 30 degrees, I am interested in another camera getting added."
#   1. coverage benches: v2t (back camera tilted up 30 deg) and v2u (v2t + a camera on the top plate looking up)
#   2. landing A/B, same code, same seeds 1-6, local 3D planner, whole deliveries: v2 (gimbal) vs v2f (fixed camera)
#   3. studio photos of v2u
# Starts when stage B's YOLO training (PID 216274) has finished, to leave the memory it was using.
R=/home/sai/RL_projects/rl-landing; Q=$R/queue; cd $R
log () { echo "$(date +%H:%M) variants: $*" >> $Q/rig.log; }
while kill -0 216274 2>/dev/null; do sleep 15; done
cover () {                                             # $1: the variant; its own bench partition (gz_rig.PARTITION)
    local v=$1 O=gz_logan/rig_$1/cover G
    mkdir -p $O
    G=$(RIG=$v /usr/bin/python3 gz_rig.py --bench)
    RIG=$v /usr/bin/python3 -u gz_rig.py --cover $O > gz_logan/rig_$v/cover.log 2>&1 && \
    RIG=$v /usr/bin/python3 -u gz_rig.py --report $O >> gz_logan/rig_$v/cover.log 2>&1 && \
    RIG=$v /usr/bin/python3 -u gz_rig.py --views $O >> gz_logan/rig_$v/cover.log 2>&1
    kill -- -$G 2>/dev/null; sleep 3; kill -9 -- -$G 2>/dev/null
    log "$v cover: $(grep -c . $O/cover.json 2>/dev/null) file(s), $(grep 'camera model check' gz_logan/rig_$v/cover.log | cut -c1-90)"
}
log "start: benches v2t v2u, landing A/B v2 vs v2f"
cover v2t & c1=$!
sleep 20
cover v2u & c2=$!
sleep 20
RIG=v2f OUT=land_ab/v2f SEEDS="1 2 3 4 5 6" RUNS="rule:local3d" SPEED=1.0 FAST=1 FULL=1 INSTANCE=4 ./voxfly.sh > /dev/null 2>&1 & f1=$!
sleep 20
RIG=v2 OUT=land_ab/v2 SEEDS="1 2 3 4 5 6" RUNS="rule:local3d" SPEED=1.0 FAST=1 FULL=1 INSTANCE=5 ./voxfly.sh > /dev/null 2>&1 & f2=$!
wait $c1 $c2
RIG=v2u /usr/bin/python3 gz_unified.py --photo > gz_logan/photo/photo_v2u.log 2>&1
wait $f1 $f2
log "done"
