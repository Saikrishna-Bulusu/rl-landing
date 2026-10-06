#!/usr/bin/env bash
# Phase 3, upgrade 2 (PLAN.md candidate 2, moved up 7 Oct): the control recipe with a privileged critic (ppo.py
# --priv-critic: in training the critic also sees land_env.privileged, the hidden drone, sensing errors and wind; the
# actor is unchanged). The control trained in the randomized world went worse than the paper decider and kept its
# round-100 checkpoint: the critic cannot tell a drone that can land from one that cannot. Same frozen flyer, seeds
# 0-2. Waits for phase3_dobin.sh (.done), at most two GPU trainings at once.
#   nohup conditions/wind/queue/phase3_priv.sh > outputs/queue/phase3_priv.out 2>&1 &      (~40 min after the DOB runs)
R=/home/sai/RL_projects/rl-landing; cd $R/conditions/wind; LOG=$R/outputs/queue/phase3_base.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_rand_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998 --w-ok 0.8 --stuck 40 --envs 4096 --steps 64 --eval-every 100 --rand all --priv-critic"
log () { echo "$(date +%H:%M) $*" >> $LOG; }
until [ -f $R/outputs/queue/phase3_dobin.done ]; do sleep 60; done
[ "$(cat /sys/class/power_supply/A*/online)" = 1 ] || { log "priv: on battery, not starting"; exit 1; }
log "priv start"
nice -n 5 $PY ppo.py --tag priv_rand --seed 0 $S2 > runs/priv_rand_s0.out 2>&1 & a=$!
sleep 30
nice -n 5 $PY ppo.py --tag priv_rand --seed 1 $S2 > runs/priv_rand_s1.out 2>&1 & b=$!
wait $a $b; log "priv s0-s1 done"
nice -n 5 $PY ppo.py --tag priv_rand --seed 2 $S2 > runs/priv_rand_s2.out 2>&1
log "priv s2 done"; touch $R/outputs/queue/phase3_priv.done
