#!/usr/bin/env bash
# Phase 3, upgrade 1 (Sai, 6 Oct: "DOB first"): the base recipe with the DOB's estimate of the wind's push as 2 more
# decider inputs (ppo.py --dob-in), on the SAME frozen flyer as the base (runs/fly_rand_s0_best.npz), seeds 0-2. Waits
# for phase3_base.sh to finish (its .done file), so at most two GPU trainings run at once.
#   nohup conditions/wind/queue/phase3_dobin.sh > outputs/queue/phase3_dobin.out 2>&1 &      (~1 h after the base)
R=/home/sai/RL_projects/rl-landing; cd $R/conditions/wind; LOG=$R/outputs/queue/phase3_base.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_rand_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998 --w-ok 0.8 --stuck 40 --envs 4096 --steps 64 --eval-every 100 --rand all --dob-in"
log () { echo "$(date +%H:%M) $*" >> $LOG; }
until [ -f $R/outputs/queue/phase3_base.done ]; do sleep 60; done
[ "$(cat /sys/class/power_supply/A*/online)" = 1 ] || { log "dobin: on battery, not starting"; exit 1; }
log "dobin start"
nice -n 5 $PY ppo.py --tag dobin_rand --seed 0 $S2 > runs/dobin_rand_s0.out 2>&1 & a=$!
sleep 30
nice -n 5 $PY ppo.py --tag dobin_rand --seed 1 $S2 > runs/dobin_rand_s1.out 2>&1 & b=$!
wait $a $b; log "dobin s0-s1 done"
nice -n 5 $PY ppo.py --tag dobin_rand --seed 2 $S2 > runs/dobin_rand_s2.out 2>&1
log "dobin s2 done"; touch $R/outputs/queue/phase3_dobin.done
