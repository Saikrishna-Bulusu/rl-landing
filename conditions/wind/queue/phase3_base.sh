#!/usr/bin/env bash
# Phase 3, step 1 (PLAN.md; Sai, 6 Oct: "Phase 3 as planned, DOB first"): the paper's recipe retrained, unchanged, in
# the corrected and randomized simulator (--rand all): the control every upgrade is compared with. The flyer (1,000
# rounds), then the decider on it, seeds 0-2. The wind rises at 80% landed or after 40 stalled rounds (randomized
# drones cannot reach the paper's 95%; with 150-round steps the flyer was still at 12 m/s after 550 rounds, 7 Oct). At most two GPU jobs at once; waits on PIDs; live curves in ClearML.
#   nohup conditions/wind/queue/phase3_base.sh > outputs/queue/phase3_base.log 2>&1 &      (~2-2.5 h)
R=/home/sai/RL_projects/rl-landing; cd $R/conditions/wind; LOG=$R/outputs/queue/phase3_base.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
COMMON="--w-ok 0.8 --stuck 40 --envs 4096 --steps 64 --eval-every 100 --rand all"
FLY="--no-decide --iters 1000 $COMMON"
S2="--flyer runs/fly_rand_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998 $COMMON"
log () { echo "$(date +%H:%M) $*" >> $LOG; }
[ "$(cat /sys/class/power_supply/A*/online)" = 1 ] || { log "on battery: not starting"; exit 1; }
log "start"
nice -n 5 $PY ppo.py --tag fly_rand --seed 0 $FLY > runs/fly_rand_s0.out 2>&1 || { log "flyer failed"; exit 1; }
log "flyer done"
nice -n 5 $PY ppo.py --tag base_rand --seed 0 $S2 > runs/base_rand_s0.out 2>&1 & a=$!
sleep 30
nice -n 5 $PY ppo.py --tag base_rand --seed 1 $S2 > runs/base_rand_s1.out 2>&1 & b=$!
wait $a $b; log "deciders s0-s1 done"
nice -n 5 $PY ppo.py --tag base_rand --seed 2 $S2 > runs/base_rand_s2.out 2>&1
log "decider s2 done"; touch $R/outputs/queue/phase3_base.done
