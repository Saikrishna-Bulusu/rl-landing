#!/usr/bin/env bash
# wind_flip_30sep.sh -- Sai 30 Sept: "try the fixes and re-fly it in Gazebo". The hard -100 decider (stage2_c100h100) pulsed
# climb/descend (MJX 0.75 reversals/s, Gazebo 1.74/s at 9 m/s) and held PX4 at its thrust limit, where Gazebo sank the drone
# (cmd_replay.py). Same recipe plus a price on each climb<->descend reversal (land_env.FLIP_R): 0.2 on seeds 0-2, 0.5 on
# seed 0 as a price check. Two runs at a time, then each best checkpoint on the test ladder and timing_check (reversals).
R=/home/sai/RL_projects/rl-landing; cd $R/conditions/wind; LOG=$R/outputs/queue/wind_flip.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100"
train () { tag=$1; seed=$2; shift 2
    $PY ppo.py --tag $tag --seed $seed $S2 "$@" --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100 > runs/${tag}_s${seed}.out 2>&1
    $PY ppo.py --ladder runs/${tag}_s${seed}_best.npz > runs/${tag}_s${seed}_ladder.out 2>&1
    $PY timing_check.py runs/${tag}_s${seed}_best.npz --w 9 18 --timing 1:0,mix:1 > runs/${tag}_s${seed}_timing.out 2>&1
    echo "$(date +%H:%M) $tag s$seed done" >> $LOG; }
echo "$(date +%H:%M) start" >> $LOG
( train stage2_c100h100f02 0 --flip-r 0.2; train stage2_c100h100f02 2 --flip-r 0.2 ) &
sleep 30
( train stage2_c100h100f02 1 --flip-r 0.2; train stage2_c100h100f05 0 --flip-r 0.5 ) &
wait
echo "$(date +%H:%M) all done" >> $LOG
