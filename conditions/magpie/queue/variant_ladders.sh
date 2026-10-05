#!/usr/bin/env bash
# variant_ladders.sh -- once the v2t / v2u coverage benches are done: their calibrations (their own coverage, blind
# spots and propeller masks; rig v2's detector score and distance-from-size, same bird, same kinds of camera), then the
# fast simulator's magpie ladder for each, after rig v2's (PID 225563) -- which rig loses how many drones to the magpie.
R=/home/sai/RL_projects/rl-landing; Q=$R/outputs/queue; P=/home/sai/miniconda3/envs/mujoco_drones/bin/python; cd $R
log () { echo "$(date +%H:%M) variant ladders: $*" >> $Q/rig.log; }
until [ -f shared/px4_gazebo/gz_logan/rig_v2t/cover/rig_views.jpg ] && [ -f shared/px4_gazebo/gz_logan/rig_v2u/cover/rig_views.jpg ]; do sleep 30; done
for v in v2t v2u; do
    RIG=$v /usr/bin/python3 shared/px4_gazebo/gz_rig.py --calib shared/px4_gazebo/gz_logan/rig_$v/cover shared/px4_gazebo/gz_logan/rig_v2/data --score score_yolo.json         --out shared/fast_sim/rig_calib_$v.json > shared/px4_gazebo/gz_logan/rig_$v/calib.log 2>&1 || log "$v calib failed"
done
while kill -0 225563 2>/dev/null; do sleep 30; done
cd shared/fast_sim; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for v in v2t v2u; do
    RIG_CALIB=$R/shared/fast_sim/rig_calib_$v.json $P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter --rig --tag _$v         > $R/outputs/fast_sim/headroom_lidar_gps_birds_hunter_rig_$v.log 2>&1 || log "$v ladder failed"
    log "$v: $(grep -m2 -A2 'corridor world' $R/outputs/fast_sim/headroom_lidar_gps_birds_hunter_rig_$v.log | grep -m1 rule | tr -s ' ' | cut -c1-110)"
done
log "done"
