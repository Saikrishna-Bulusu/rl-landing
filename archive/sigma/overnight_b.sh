#!/bin/bash
# Second half of the overnight run. Waits for overnight.sh to post its
# completion marker, then continues. Kept as a SEPARATE FILE because bash reads
# a script incrementally as it executes -- editing overnight.sh while it runs
# would make it execute garbage.
#
#   4. rho=0.363 topped up to 20 seeds   (36 runs @ 2.5M)
#   5. STAGE 2 cage: HP sweep            (36 runs @ 1M,  seed 0 throughout)
#   6. STAGE 2 cage: 16-seed ablation    (48 runs @ 2M)
#
# MEASURED before launching (never assume):
#   point-mass job 864 MB, cage job 972 MB, both CPU-only, GPU 15 MiB idle.
#   8 jobs -> ~7.8 GB of 23 GB, leaving 10+ GB available. Disk: 0.25 MB/run.
set -u
cd /home/sai/RL_projects/rl-landing/archive/sigma   # these drivers run from their own folder
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a ../../outputs/logs/overnight.log; }

# Refuse to start a stage if the machine is already short on memory.
guard(){
  local avail; avail=$(free -m | awk 'NR==2{print $7}')
  local free_g; free_g=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "  guard: ${avail} MB RAM available, ${free_g} GB disk free"
  if [ "$avail" -lt 3000 ]; then
    say "  ABORT: under 3 GB RAM available, refusing to launch more jobs"; exit 1
  fi
  if [ "$free_g" -lt 10 ]; then
    say "  ABORT: under 10 GB disk free, refusing to launch more jobs"; exit 1
  fi
}

run_jobs(){   # $1 = jobs file
  # train.py needs --tb "" -- tensorboard is not installed and SB3 dies if it is
  # asked for. An empty string cannot survive word-splitting inside the hp
  # field, so it is passed explicitly here. ../tello/train_cage.py has no --tb flag.
  cat "$1" | xargs -P 8 -I{} bash -c '
    IFS="|" read -r script out steps rho seed hp <<< "{}"
    arm="${out##*/}"; arm="${arm%%_*}"
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    PYBIN=/home/sai/miniconda3/envs/mujoco_drones/bin/python
    if [ "$script" = "train.py" ]; then
      nice -n 10 $PYBIN "$script" --pool sigma_pool.npz --arm "$arm" \
        --seed "$seed" --steps "$steps" --eval-episodes 400 --rho "$rho" \
        --out "$out" --tb "" $hp > "${out}.log" 2>&1
    else
      nice -n 10 $PYBIN "$script" --pool sigma_pool.npz --arm "$arm" \
        --seed "$seed" --steps "$steps" --eval-episodes 400 --rho "$rho" \
        --out "$out" $hp > "${out}.log" 2>&1
    fi
  '
}

say "=== SECOND HALF: waiting for the first chain to finish ==="
until grep -q "ALL OVERNIGHT STAGES COMPLETE" ../../outputs/logs/overnight.log 2>/dev/null; do sleep 60; done
say "=== SECOND HALF: first chain done, continuing ==="

# ---------- 4. rho=0.363 up to 20 seeds ----------
guard
say "STAGE 4: rho=0.363 seeds 8-19 (36 runs)"
: > final/jobs_more.txt
for s in $(seq 8 19); do
  echo "train.py|final/none_s${s}|2500000|0.363|${s}|--n-envs 8 --t-max 45 --lr 1e-3 --ent-coef 0.02 --net 256 128"      >> final/jobs_more.txt
  echo "train.py|final/staleness_s${s}|2500000|0.363|${s}|--n-envs 8 --t-max 45 --lr 3e-4 --ent-coef 0.02 --net 256 128" >> final/jobs_more.txt
  echo "train.py|final/sigma_s${s}|2500000|0.363|${s}|--n-envs 8 --t-max 45 --lr 1e-3 --ent-coef 0.02 --net 128 64"      >> final/jobs_more.txt
done
run_jobs final/jobs_more.txt
say "STAGE 4 done: $(ls final/*/ablation.json 2>/dev/null | wc -l)/60"

# ---------- 5. cage HP sweep ----------
guard
say "STAGE 5: CAGE hyperparameter sweep (36 runs @ 1M, seed 0)"
mkdir -p cage_sweep; : > cage_sweep/jobs.txt
for arm in none staleness sigma; do i=0
  for lr in 1e-4 3e-4 1e-3; do for ent in 0.003 0.02; do for net in "128 64" "256 128"; do
    i=$((i+1))
    echo "../tello/train_cage.py|cage_sweep/${arm}_c${i}|1000000|0.707|0|--n-envs 4 --lr $lr --ent-coef $ent --net $net" >> cage_sweep/jobs.txt
  done; done; done
done
run_jobs cage_sweep/jobs.txt
say "STAGE 5 done: $(ls cage_sweep/*/ablation.json 2>/dev/null | wc -l)/36"

# ---------- 6. cage ablation ----------
guard
say "STAGE 6: CAGE 16-seed ablation (48 runs @ 2M)"
mkdir -p cage_final
$PY - <<'PYEOF' | tee -a ../../outputs/logs/overnight.log
import json, os
GRID=[f"--lr {lr} --ent-coef {e} --net {n}" for lr in ("1e-4","3e-4","1e-3")
      for e in ("0.003","0.02") for n in ("128 64","256 128")]
best={}
for arm in ("none","staleness","sigma"):
    rows=[]
    for i in range(1,13):
        p=f"cage_sweep/{arm}_c{i}/ablation.json"
        if os.path.exists(p):
            r=json.load(open(p))["results"][arm]
            rows.append((r["success_%"], GRID[i-1]))
    rows.sort(reverse=True)
    best[arm]=rows[0][1] if rows else GRID[9]
    print(f"  cage winner {arm:<10} {rows[0][0] if rows else float('nan'):5.1f}%  {best[arm]}")
json.dump(best, open("cage_sweep/best.json","w"), indent=2)
with open("cage_final/jobs.txt","w") as fh:
    for s in range(16):
        for arm,hp in best.items():
            fh.write(f"../tello/train_cage.py|cage_final/{arm}_s{s}|2000000|0.707|{s}|--n-envs 4 {hp}\n")
PYEOF
run_jobs cage_final/jobs.txt
say "STAGE 6 done: $(ls cage_final/*/ablation.json 2>/dev/null | wc -l)/48"
say "=== SECOND HALF COMPLETE ==="
