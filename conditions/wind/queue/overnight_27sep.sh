#!/usr/bin/env bash
# overnight_27sep.sh -- Sai, 26 Sept 23:30: "run some overnight runs ... maybe get the MJX running more seeds".
# A. MJX on the GPU, three lanes; each run is scored on the ladder (1024 paired flights x 8 winds) as it ends:
#    - the flying learner (fly_hp: filtered push, flying only), seeds 3-7 -> 8 seeds with s0-s2
#    - the crash-priced decider (stage2_c100: 87% landed at 18 m/s vs the flyer's 44-55%, 2.1% crashed), seeds 1-2
#    - three variants of it: on the filtered flyer (stage2hp_c100), crash -300 (stage2_c300), hard -100 (stage2_c100h100)
# B. Gazebo once the GPU is free (the two share it badly): the crash-priced decider and a second flyer seed on the
#    same 20 yards x 4 winds as the rule and fly_hp_s0 (does the MJX gain survive Gazebo?); then two GeoAnchor flights,
#    filmed: calm (re-flown: no marker fixes below 3 m) and 9 m/s wind with the flying learner.
# C. The morning report: conditions/wind/overnight_report.py -> conditions/wind/queue/overnight_report.md
R=/home/sai/RL_projects/rl-landing; W=$R/conditions/wind; PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
LOG=$R/outputs/queue/overnight.log
cd $W
echo "$(date +%H:%M) overnight start" >> $LOG
train () {                                           # tag seed args... (skips what is already trained and scored)
    local tag=$1 seed=$2; shift 2
    if [ -f runs/${tag}_s${seed}_best_ladder.json ]; then return; fi
    echo "$(date +%H:%M) start $tag s$seed" >> $LOG
    $PY ppo.py --tag $tag --seed $seed "$@" --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100 > runs/${tag}_s${seed}.out 2>&1
    echo "$(date +%H:%M) trained $tag s$seed (exit $?)" >> $LOG
    $PY ppo.py --ladder runs/${tag}_s${seed}_best.npz > runs/${tag}_s${seed}_best_ladder.out 2>&1
    echo "$(date +%H:%M) scored $tag s$seed (exit $?)" >> $LOG
}
FH="--no-decide --hp-tau 4 --iters 1000"
S2="--flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600"
( train fly_hp 3 $FH; train stage2_c100 1 $S2 --crash-r -100 --hard-r -30; train fly_hp 6 $FH ) &
L1=$!
sleep 60
( train fly_hp 4 $FH; train stage2_c100 2 $S2 --crash-r -100 --hard-r -30; train fly_hp 7 $FH ) &
L2=$!
sleep 60
( train stage2hp_c100 0 --flyer runs/fly_hp_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -30
  train stage2_c300 0 $S2 --crash-r -300 --hard-r -30
  train stage2_c100h100 0 $S2 --crash-r -100 --hard-r -100
  train fly_hp 5 $FH ) &
L3=$!
wait $L1 $L2 $L3
echo "$(date +%H:%M) MJX done" >> $LOG

cd $R                                                # B. Gazebo
EPISODES="$(seq 0 19 | tr '\n' ' ')" WINDS="9 12 15 18" DECIDERS="conditions/wind/runs/stage2_c100_s0_best.npz conditions/wind/runs/fly_hp_s1_best.npz" \
    N=4 OUT=ladder_rule ./conditions/wind/gz_ladder.sh
echo "$(date +%H:%M) Gazebo ladder done: $(wc -l < outputs/gazebo_wind/ladder_rule/flights.jsonl) flights in all" >> $LOG
OUT=geo_calm2 WORLD=geo_clean INSTANCE=5 timeout 3000 gnss_denied/geofly.sh /usr/bin/python3 -u conditions/wind/gz_windland.py --geo \
    --episode 0 --w10 0 --decider rule --out $R/outputs/gazebo_geo/geo_calm2/flights.jsonl --video $R/outputs/gazebo_geo/geo_calm2/calm.mp4
echo "$(date +%H:%M) GeoAnchor calm: $(cat outputs/gazebo_geo/geo_calm2/flights.jsonl 2>/dev/null)" >> $LOG
OUT=geo_wind9 WORLD=geo_clean SERVER_CONFIG=$R/shared/px4_gazebo/gz_logan/geo_clean/geo_w9.config INSTANCE=5 timeout 3000 gnss_denied/geofly.sh \
    /usr/bin/python3 -u conditions/wind/gz_windland.py --geo --episode 0 --w10 9 --decider conditions/wind/runs/fly_hp_s0_best.npz \
    --out $R/outputs/gazebo_geo/geo_wind9/flights.jsonl --video $R/outputs/gazebo_geo/geo_wind9/wind9.mp4
echo "$(date +%H:%M) GeoAnchor wind 9: $(cat outputs/gazebo_geo/geo_wind9/flights.jsonl 2>/dev/null)" >> $LOG

/usr/bin/python3 $W/overnight_report.py > $R/conditions/wind/queue/overnight_report.md 2>> $LOG   # C.
echo "$(date +%H:%M) overnight done" >> $LOG
