#!/usr/bin/env bash
# wind_recipes_29sep.sh -- seeds 1-2 of the two decider recipes that beat stage2_c100 on VALIDATION with seed 0
# (stage2_c300: crash -300; stage2_c100h100: hard -100), same settings as overnight_27sep.sh, two runs at a time on
# the GPU (one run leaves it underused: 8192 drones cost 1.6x the time of 4096). Test-yard logging off (--log-test).
# Then each new best checkpoint on the 1024-flight test ladder. Log: queue/wind_recipes.log
R=/home/sai/RL_projects/rl-landing; cd $R/wind; LOG=$R/queue/wind_recipes.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600"
train () { tag=$1; seed=$2; shift 2
    $PY ppo.py --tag $tag --seed $seed $S2 "$@" --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100 > runs/${tag}_s${seed}.out 2>&1
    $PY ppo.py --ladder runs/${tag}_s${seed}_best.npz > runs/${tag}_s${seed}_ladder.out 2>&1
    echo "$(date +%H:%M) $tag s$seed done" >> $LOG; }
echo "$(date +%H:%M) start" >> $LOG
( train stage2_c300 1 --crash-r -300 --hard-r -30; train stage2_c300 2 --crash-r -300 --hard-r -30 ) &
sleep 30
( train stage2_c100h100 1 --crash-r -100 --hard-r -100; train stage2_c100h100 2 --crash-r -100 --hard-r -100 ) &
wait
echo "$(date +%H:%M) all done" >> $LOG
