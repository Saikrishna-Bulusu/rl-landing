#!/usr/bin/env bash
# Phase 3 warm start, step 2 (after phase3_warm.sh chose the leash on seed 0's VALIDATION flights): seeds 1-2 at that
# leash, the re-priced arm (a gentle touchdown off the pad -10, too fast and crashes still -100) seeds 0-2, then every
# arm on the test flights (phase3_compare.py; set its WARM / WARM_P to these tags first).
#   nohup conditions/wind/queue/phase3_warm2.sh 0.3 > outputs/queue/phase3_warm2.out 2>&1 &      (~45 min)
B=$1; T=$(echo "$B" | tr -d .)                  # leash 0.3 -> tags warm_leash03, warmp_leash03
R=/home/sai/RL_projects/rl-landing; cd $R/conditions/wind; LOG=$R/outputs/queue/phase3_base.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998 --w-start 20 --envs 4096 --steps 64 --eval-every 100 --rand all --priv-critic --critic-warmup 50 --anchor $B"
log () { echo "$(date +%H:%M) $*" >> $LOG; }
run () { nice -n 5 $PY ppo.py --tag $1 --seed $2 $S2 --init runs/stage2_c100h100g_s$2_best.npz ${@:3} > runs/$1_s$2.out 2>&1; }
[ -n "$B" ] && grep -q "\"warm_leash$T\", \"warmp_leash$T\"" phase3_compare.py || { echo "usage: $0 LEASH (and set phase3_compare.WARM)"; exit 1; }
[ "$(cat /sys/class/power_supply/A*/online)" = 1 ] || { log "warm2: on battery, not starting"; exit 1; }
log "warm2 start, leash $B"
run warm_leash$T 1 & a=$!; sleep 30; run warm_leash$T 2 & b=$!; wait $a $b
run warmp_leash$T 0 --offpad-r -10 & a=$!; sleep 30; run warmp_leash$T 1 --offpad-r -10 & b=$!; wait $a $b
run warmp_leash$T 2 --offpad-r -10
log "warm2 trained; comparing"
$PY phase3_compare.py > runs/phase3_compare.out 2>&1
log "warm2 done"; touch $R/outputs/queue/phase3_warm2.done
