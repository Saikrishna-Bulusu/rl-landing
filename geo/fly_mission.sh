#!/usr/bin/env bash
# fly_mission.sh -- one mission from the dashboard's planner (dashboard/app/mission), flown like the GeoAnchor flights:
# PX4 + Gazebo on the photo floor, GeoAnchor in the loop, GNSS denied over the suburb, the chosen lander on the chosen
# pad (any of the sixteen; the mission's route, height and speed via GEO_MISSION: geo_cruise.py, px4_mission.py).
#   geo/fly_mission.sh geo/missions/to_pad_4.json mission_to_pad_4        -> gazebo_geo/mission_to_pad_4/
# One flight at a time: the dashboard refuses while a Gazebo world is up (two at once broke PX4's estimate, 27 Sept).
set -e
R=/home/sai/RL_projects/rl-landing; cd $R
M=$(readlink -f "${1:?mission json}"); OUTN=${2:?output name}
read -r LANDER WIND <<< "$(python3 -c "import json; m = json.load(open('$M')); print(m['lander'].replace(' ', '_'), m['wind'])")"
D=$R/gazebo_geo/$OUTN; mkdir -p $D; cp $M $D/mission.json
CFG=
if python3 -c "import sys; sys.exit(0 if float('$WIND') > 0 else 1)"; then    # WindDrag at the mission's speed, from the west
    sed "s|<mean_speed>[^<]*</mean_speed>|<mean_speed>$WIND</mean_speed>|" gz_logan/geo_clean/geo_w9.config > $D/wind.config
    CFG=$D/wind.config
fi
case $LANDER in
    rule)           CMD=(wind/gz_windland.py --geo --episode 0 --w10 $WIND --decider rule --out $D/flights.jsonl --video $D/mission.mp4) ;;
    flying_learner) CMD=(wind/gz_windland.py --geo --episode 0 --w10 $WIND --decider wind/runs/fly_hp_s0_best.npz --out $D/flights.jsonl --video $D/mission.mp4) ;;
    none)           CMD=(geo/geo_cruise.py) ;;
    *)              echo "unknown lander $LANDER"; exit 2 ;;
esac
set +e
env GEO_MISSION=$M ${CFG:+SERVER_CONFIG=$CFG} OUT=$OUTN WORLD=geo_clean INSTANCE=${INSTANCE:-5} timeout 3000 \
    geo/geofly.sh /usr/bin/python3 -u "${CMD[@]}"
echo "mission $OUTN exit $?"
