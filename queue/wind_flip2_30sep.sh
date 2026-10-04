#!/usr/bin/env bash
# wind_flip2_30sep.sh -- second try at the reversal price. The first (wind_flip_30sep.sh, flip 0.2) drifted to 25-49% timeouts
# with training returns above a landing's: the shaping's discount 0.997 (learner 0.9998) pays ~+15 for hovering 90 s, and the
# price made descending dearer. Now shaping at the learner's discount (--shape-g 0.9998): flip 0.2 on seeds 0-2, and shaping
# alone (no price) on seed 0 as the control. Also scores the two first-try bests. Two runs at a time.
R=/home/sai/RL_projects/rl-landing; cd $R/wind; LOG=$R/queue/wind_flip.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998"
score () { $PY ppo.py --ladder runs/$1_best.npz > runs/$1_ladder.out 2>&1
    $PY timing_check.py runs/$1_best.npz --w 9 18 --timing 1:0,mix:1 > runs/$1_timing.out 2>&1; }
train () { tag=$1; seed=$2; shift 2
    $PY ppo.py --tag $tag --seed $seed $S2 "$@" --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100 > runs/${tag}_s${seed}.out 2>&1
    score ${tag}_s${seed}; echo "$(date +%H:%M) $tag s$seed done" >> $LOG; }
echo "$(date +%H:%M) flip2 start" >> $LOG
( train stage2_c100h100f02g 0 --flip-r 0.2; train stage2_c100h100f02g 2 --flip-r 0.2; score stage2_c100h100f02_s0 ) &
sleep 30
( train stage2_c100h100f02g 1 --flip-r 0.2; train stage2_c100h100g 0; score stage2_c100h100f02_s1 ) &
wait
echo "$(date +%H:%M) flip2 all done" >> $LOG
