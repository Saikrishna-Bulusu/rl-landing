#!/usr/bin/env bash
# baselines_6oct.sh -- what learning adds in wind (Sai, 6 Oct: "baselines first"). The PPO decider on the disturbance
# observer instead of the learned flyer (3 seeds, the paper's stage-2 recipe), the tuned rule (rule_tuned.py), then every
# arm on the same 1,024 test flights per wind (compare_baselines.py). At most two GPU jobs at once; waits on PIDs.
#   conditions/wind/queue/baselines_6oct.sh     (about 30-40 min; live curves in ClearML rl-landing/wind)
R=/home/sai/RL_projects/rl-landing; cd $R/conditions/wind; LOG=$R/outputs/queue/baselines_6oct.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer dob:1.0 --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998 --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100"
log () { echo "$(date +%H:%M) $*" >> $LOG; }
mkdir -p $R/outputs/queue
[ "$(cat /sys/class/power_supply/A*/online)" = 1 ] || { log "on battery: not starting"; exit 1; }
log "start"
nice -n 10 $PY ppo.py --tag stage2_dob_c100h100g --seed 0 $S2 > runs/stage2_dob_c100h100g_s0.out 2>&1 & a=$!
sleep 30
nice -n 10 $PY ppo.py --tag stage2_dob_c100h100g --seed 1 $S2 > runs/stage2_dob_c100h100g_s1.out 2>&1 & b=$!
wait $a $b; log "DOB decider seeds 0-1 done"
nice -n 10 $PY ppo.py --tag stage2_dob_c100h100g --seed 2 $S2 > runs/stage2_dob_c100h100g_s2.out 2>&1 & c=$!
nice -n 10 $PY rule_tuned.py > runs/rule_tuned.out 2>&1 & d=$!
wait $c $d; log "DOB decider seed 2 and the tuned rule done"
nice -n 10 $PY compare_baselines.py > runs/compare_baselines.out 2>&1 && log "compare done" && touch $R/outputs/queue/baselines_6oct.done
