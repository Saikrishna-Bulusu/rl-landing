#!/usr/bin/env bash
# Phase 3, upgrades 3-4 (Sai, 7 Oct: "Warm start + critic" and "Re-price the outcomes"): fine-tune the paper decider
# (seed s starts from stage2_c100h100g_s<s>) in the randomized world with the privileged critic, all winds from the start.
# 7 Oct, first try (warm_rand s0-s1, stopped): unleashed, it unlearned landing in every wind within ~30 rounds of the
# actor's release (validation 0.00), then slowly relearned. Now: the critic learns alone for 50 rounds (a flight lasts
# 15-35), a fresh Adam at the release, and a leash (--anchor b: b x KL from the starting decider, per decision).
#   step 1  seed 0, leash 0 / 0.1 / 0.3 / 1: the leash chosen on VALIDATION flights (never the test yards)
#   step 2  (phase3_warm2.sh) seeds 1-2 at that leash; re-priced (--offpad-r -10) seeds 0-2
#   nohup conditions/wind/queue/phase3_warm.sh > outputs/queue/phase3_warm.out 2>&1 &      (~25 min)
R=/home/sai/RL_projects/rl-landing; cd $R/conditions/wind; LOG=$R/outputs/queue/phase3_base.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998 --w-start 20 --envs 4096 --steps 64 --eval-every 100 --rand all --priv-critic --critic-warmup 50"
log () { echo "$(date +%H:%M) $*" >> $LOG; }
run () { nice -n 5 $PY ppo.py --tag $1 --seed $2 $S2 --init runs/stage2_c100h100g_s$2_best.npz ${@:3} > runs/$1_s$2.out 2>&1; }
[ "$(cat /sys/class/power_supply/A*/online)" = 1 ] || { log "warm: on battery, not starting"; exit 1; }
log "warm leash sweep start"
run warm_leash0 0 --anchor 0 & a=$!; sleep 30; run warm_leash1 0 --anchor 1 & b=$!; wait $a $b
run warm_leash01 0 --anchor 0.1 & a=$!; sleep 30; run warm_leash03 0 --anchor 0.3 & b=$!; wait $a $b
log "warm leash sweep done"; touch $R/outputs/queue/phase3_warm.done
