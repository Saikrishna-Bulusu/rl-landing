#!/usr/bin/env bash
# overnight_27sep_b.sh -- after overnight_27sep.sh (pid 870392): the two GeoAnchor flights through gz_learner, filmed --
#   obstacles: geo_floor (stage 3's crane, lights, traffic light, trees, shed round the pad), rule + local 3D planner
#   magpie:    geo_magpie (hunter), the rig-trained learner (v2l s0 -- the only rig-trained checkpoint; flown on rig v2
#              hardware, Sai's choice of 26 Sept), the magpie known only through the rig; the sim at 0.1x after hand-over
# then the morning report again. Waits on the PID, not a name.
R=/home/sai/RL_projects/rl-landing; cd $R
while kill -0 870392 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) b: GeoAnchor obstacles + magpie start" >> queue/overnight.log
[ -f $R/gazebo_geo/geo_obst/obstacles_geo.json ] || TREES_JSON=gz_logan/trees_logan.json OUT=geo_obst WORLD=geo_floor INSTANCE=5 timeout 3000 geo/geofly.sh \
    /usr/bin/python3 -u gz_learner.py --geo --decider rule --seed 1 --magpie none --voxel --local3d \
    --video $R/gazebo_geo/geo_obst/obstacles.mp4
echo "$(date +%H:%M) b: obstacles done" >> queue/overnight.log
[ -f $R/gazebo_geo/geo_magpie/magpie_geo.json ] || TREES_JSON=gz_logan/trees_logan.json RIG=v2 RIG_YOLO=$R/gz_logan/rig_v2/yolo/run/weights/best.pt \
    RIG_CALIB=$R/urban/rig_calib_v2.json GEO_LAND_SPEED=0.1 OUT=geo_magpie WORLD=geo_magpie INSTANCE=5 timeout 5400 \
    geo/geofly.sh /usr/bin/python3 -u gz_learner.py --geo --decider learner --seed 0 --magpie hunter --sense rig \
    --ckpt rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt --video $R/gazebo_geo/geo_magpie/magpie.mp4
echo "$(date +%H:%M) b: magpie done" >> queue/overnight.log
/usr/bin/python3 wind/overnight_report.py > queue/overnight_report.md 2>> queue/overnight.log
echo "$(date +%H:%M) b: report written" >> queue/overnight.log
