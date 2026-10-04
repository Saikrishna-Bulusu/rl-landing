#!/usr/bin/env bash
# overnight_24sep.sh -- everything left after the 16:53 power loss (charger out, battery flat).
#
#   1. Gazebo magpie flights (SITL already up in gz_logan/birds): planner and learner, seeds 0-3
#   2. two training chains in parallel, one GPU (8 GB: two trainings, or one training + one
#      headroom ladder -- never three things):
#        A: headroom ladders for stage 3 and own-cameras birds -> fog seed 2 (lost) ->
#           unmapped structures s0-2 -> own-cameras birds s2
#        B: birds+fog seed 2 (lost) -> everything s0-2 -> own-cameras birds s0-1
#   3. the conditions figure
# Each step touches queue/<step>.done ONLY on success; queue/overnight.log has the timeline.
R=/home/sai/RL_projects/rl-landing; U=$R/urban; Q=$R/queue
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/opt/ros/jazzy/opt/gz_tools_vendor/bin:$PATH
source /opt/ros/jazzy/setup.bash 2>/dev/null
log () { echo "$(date +%H:%M) $*" >> $Q/overnight.log; }
step () { local name=$1; shift; log "start $name"; if "$@"; then touch $Q/$name.done; log "done  $name"; else log "FAILED $name ($?)"; fi; }
RL="$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps"
HR="$P urban_headroom.py --map logan_map_lidar.npz --nav gps"

cd $R
gazebo () {
    for s in 0 1 2 3; do for d in planner learner; do
        TREES_JSON=gz_logan/trees_logan.json timeout 600 /usr/bin/python3 -u gz_learner.py --decider $d --seed $s \
            --video gazebo_birds/${d}_s$s.mp4 >> gazebo_birds/flights.jsonl 2>> gazebo_birds/flights.err
    done; done
    for s in 0 1 2 3; do /usr/bin/python3 gz_learner.py --compose gazebo_birds/planner_s$s.mp4 gazebo_birds/learner_s$s.mp4 \
        gazebo_birds/pair_s$s.mp4 >> gazebo_birds/flights.err 2>&1; done
    pkill -f "bin/p[x]4 "; pkill -f "gz si[m]"; sleep 5
    grep -c '"outcome"' gazebo_birds/flights.jsonl
}
step gazebo_birds gazebo

cd $U
chainA () {
    sleep 240                                # after chain B's first copy phase: its fit is the memory spike
    step ladder_unmapped      $HR --unmapped                 > headroom_lidar_gps_unmapped.log 2>&1
    step ladder_all           $HR --birds --fog --unmapped   > headroom_lidar_gps_birds_fog_unmapped.log 2>&1
    step ladder_cameras       $HR --birds --cameras          > headroom_lidar_gps_birds_cameras.log 2>&1
    step train_fog_s2         $RL --seeds 2 --fog            > rl_cond_fog_s2.log 2>&1
    step train_unmapped       $RL --seeds 0 1 2 --unmapped   > rl_cond_unmapped.log 2>&1
    step train_cameras_s2     $RL --seeds 2 --birds --cameras > rl_cond_birds_cameras_s2.log 2>&1
}
chainB () {
    step train_birds_fog_s2   $RL --seeds 2 --birds --fog            > rl_cond_birds_fog_s2.log 2>&1
    step train_all            $RL --seeds 0 1 2 --birds --fog --unmapped > rl_cond_all.log 2>&1
    step train_cameras_s01    $RL --seeds 0 1 --birds --cameras      > rl_cond_birds_cameras_s01.log 2>&1
}
chainA & chainB & wait
step figure $P plot_conditions.py --out conditions.png
log "overnight queue finished"
