#!/bin/bash
# Overnight chain, 18->19 Sept 2026. Runs unattended, no human between stages.
#
#   1. HP sweep at rho=0.8           (36 runs @ 600k, ALL at training seed 0)
#   2. 20-seed ablation at rho=0.8   (60 runs @ 2.5M)  <- the disambiguator
#   3. top rho=0.707 up to 20 seeds  (36 runs @ 2.5M)  <- equal seed counts
#
# Why this order: stage 2 changes ONLY rho relative to the rho=0.707 fair test,
# and ONLY the HP source relative to the earlier rho=0.8 run, so whichever way
# it lands the confound between those two is resolved. Stage 3 exists because
# the sigma mean at rho=0.707 was carried by two lucky seeds out of eight.
#
# NOTE ON SEEDS: every tuning candidate trains at seed 0. An earlier version of
# this script derived the seed from the candidate index, which would have
# selected the luckiest (settings, seed) PAIR rather than the best settings --
# the same class of unfair-comparison bug that has already produced three wrong
# conclusions in this project.
set -u
cd /home/sai/RL_projects/rl-landing
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a overnight.log; }

# jobs file format:  outdir|steps|rho|seed|hp...
run_jobs(){
  cat "$1" | xargs -P 12 -I{} bash -c '
    IFS="|" read -r out steps rho seed hp <<< "{}"
    arm="${out##*/}"; arm="${arm%%_*}"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    /home/sai/miniconda3/envs/mujoco_drones/bin/python train.py \
      --pool sigma_pool.npz --arm "$arm" --seed "$seed" --steps "$steps" \
      --n-envs 8 --eval-episodes 400 --t-max 45 --rho "$rho" --tb "" \
      --out "$out" $hp > "${out}.log" 2>&1
  '
}

say "=== OVERNIGHT CHAIN START ==="

# ---------- 1 ----------
say "STAGE 1: HP sweep at rho=0.8 (36 runs, seed 0 throughout)"
mkdir -p sweep08; : > sweep08/jobs.txt
for arm in none staleness sigma; do i=0
  for lr in 1e-4 3e-4 1e-3; do for ent in 0.003 0.02; do for net in "128 64" "256 128"; do
    i=$((i+1))
    echo "sweep08/${arm}_c${i}|600000|0.8|0|--lr $lr --ent-coef $ent --net $net" >> sweep08/jobs.txt
  done; done; done
done
run_jobs sweep08/jobs.txt
say "STAGE 1 done: $(ls sweep08/*/ablation.json 2>/dev/null | wc -l)/36"

# ---------- 2 ----------
say "STAGE 2: 20-seed ablation at rho=0.8 (60 runs)"
mkdir -p final08
$PY - <<'PYEOF' | tee -a overnight.log
import json, os
GRID=[f"--lr {lr} --ent-coef {e} --net {n}" for lr in ("1e-4","3e-4","1e-3")
      for e in ("0.003","0.02") for n in ("128 64","256 128")]
best={}
for arm in ("none","staleness","sigma"):
    rows=[]
    for i in range(1,13):
        p=f"sweep08/{arm}_c{i}/ablation.json"
        if os.path.exists(p):
            rows.append((json.load(open(p))["results"][arm]["success_high_sigma_%"], GRID[i-1]))
    rows.sort(reverse=True)
    best[arm]=rows[0][1] if rows else GRID[9]
    print(f"  rho=0.8 winner {arm:<10} {rows[0][0] if rows else float('nan'):5.1f}  {best[arm]}")
json.dump(best, open("sweep08/best.json","w"), indent=2)
with open("final08/jobs.txt","w") as fh:
    for s in range(20):
        for arm,hp in best.items():
            fh.write(f"final08/{arm}_s{s}|2500000|0.8|{s}|{hp}\n")
PYEOF
run_jobs final08/jobs.txt
say "STAGE 2 done: $(ls final08/*/ablation.json 2>/dev/null | wc -l)/60"

# ---------- 3 ----------
say "STAGE 3: rho=0.707 seeds 8-19 (36 runs)"
: > final07/jobs_more.txt
for s in $(seq 8 19); do
  echo "final07/none_s${s}|2500000|0.707|${s}|--lr 1e-3 --ent-coef 0.02 --net 256 128"       >> final07/jobs_more.txt
  echo "final07/staleness_s${s}|2500000|0.707|${s}|--lr 3e-4 --ent-coef 0.003 --net 256 128" >> final07/jobs_more.txt
  echo "final07/sigma_s${s}|2500000|0.707|${s}|--lr 3e-4 --ent-coef 0.02 --net 128 64"       >> final07/jobs_more.txt
done
run_jobs final07/jobs_more.txt
say "STAGE 3 done: $(ls final07/*/ablation.json 2>/dev/null | wc -l)/60"
say "=== ALL OVERNIGHT STAGES COMPLETE ==="
