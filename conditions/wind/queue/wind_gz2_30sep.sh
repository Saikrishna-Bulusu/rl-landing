#!/usr/bin/env bash
# wind_gz2_30sep.sh -- after the ACRA session's GNSS-free flights (queue/geo_calm10.done): two more seeds of the shaping-only
# control (stage2_c100h100g: shaping at the learner's discount, no reversal price), which on seed 0 did not pulse (0 reversals/s
# at 9 m/s, old 0.75) and landed 96.2% at 18 m/s in MJX. Then the Gazebo ladder, 20 yards x 4 winds, interleaved per yard:
# the reversal-priced decider chosen on validation (stage2_c100h100f02g_s1, val 0.978) and the best control seed on validation.
R=/home/sai/RL_projects/rl-landing; cd $R/wind; LOG=$R/queue/wind_flip.log
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
S2="--flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998"
train () { tag=$1; seed=$2; shift 2
    $PY ppo.py --tag $tag --seed $seed $S2 "$@" --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100 > runs/${tag}_s${seed}.out 2>&1
    $PY ppo.py --ladder runs/${tag}_s${seed}_best.npz > runs/${tag}_s${seed}_ladder.out 2>&1
    $PY timing_check.py runs/${tag}_s${seed}_best.npz --w 9 18 --timing 1:0,mix:1 > runs/${tag}_s${seed}_timing.out 2>&1
    echo "$(date +%H:%M) $tag s$seed done" >> $LOG; }
echo "$(date +%H:%M) gz2 queued: waiting for queue/geo_calm10.done" >> $LOG
until [ -f $R/queue/geo_calm10.done ]; do sleep 60; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "$(date +%H:%M) gz2: training control seeds 1-2" >> $LOG
train stage2_c100h100g 1 & sleep 30; train stage2_c100h100g 2 & wait
G=$($PY - <<'PYEOF'
import json, numpy as np, glob
rows = [(f, float(np.mean(json.loads(str(np.load(f)["meta"]))["val"]))) for f in sorted(glob.glob("runs/stage2_c100h100g_s[0-9]_best.npz"))]
print(*[f"{f} val mean {v:.4f}" for f, v in rows], sep="\n", file=open("/home/sai/RL_projects/rl-landing/queue/wind_flip.log", "a"))
print(max(rows, key=lambda r: r[1])[0])
PYEOF
)
cd $R
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "$(date +%H:%M) gz2 ladder start: wind/runs/stage2_c100h100f02g_s1_best.npz + wind/$G" >> $LOG
EPISODES="$(seq 0 19)" WINDS="9 12 15 18" DECIDERS="wind/runs/stage2_c100h100f02g_s1_best.npz wind/$G" N=1 OUT=ladder_rule ./wind/gz_ladder.sh
echo "$(date +%H:%M) gz2 ladder done" >> $LOG
touch $R/queue/wind_gz2_30sep.done
