#!/bin/bash
# Cage re-run, 19 Sept. Replaces the 84 runs wasted by the un-normalised
# action space. Preflight (checks 1-4) passed before launch; resources verified.
#   CPU-only, 8 parallel, niced -- see ../sigma/overnight.sh header for the measurements.
#   3M steps per ablation run: the repo's own train_landing.py uses 3M, and an
#   800k smoke run only reached 11% success (rising), so shorter is too short.
set -u
cd /home/sai/RL_projects/rl-landing/archive/tello   # these drivers run from their own folder
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a ../../outputs/logs/cage_rerun.log; }
guard(){
  local r=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  local d=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "  guard: ${r} MB RAM available, ${d} GB disk free"
  if [ "$r" -lt 3000 ] || [ "$d" -lt 5 ]; then say "  ABORT: resources too low"; exit 1; fi
}
run_jobs(){
  cat "$1" | xargs -P 8 -I{} bash -c '
    IFS="|" read -r out steps arm seed hp <<< "{}"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
    /home/sai/miniconda3/envs/mujoco_drones/bin/python train_cage.py \
      --pool sigma_pool.npz --arm "$arm" --seed "$seed" --steps "$steps" \
      --n-envs 8 --eval-episodes 400 --rho 0.707 --out "$out" $hp \
      > "${out}.log" 2>&1
  '
}

say "=== CAGE RE-RUN START (action space fixed, preflight passed) ==="
guard

say "STAGE A: cage HP sweep (36 runs @ 1.5M, seed 0)"
rm -rf cage_sweep2; mkdir -p cage_sweep2; : > cage_sweep2/jobs.txt
for arm in none staleness sigma; do i=0
  for lr in 1e-4 3e-4 1e-3; do for ent in 0.003 0.02; do for net in "128 64" "256 128"; do
    i=$((i+1))
    echo "cage_sweep2/${arm}_c${i}|1500000|${arm}|0|--lr $lr --ent-coef $ent --net $net" >> cage_sweep2/jobs.txt
  done; done; done
done
run_jobs cage_sweep2/jobs.txt
say "STAGE A done: $(ls cage_sweep2/*/ablation.json 2>/dev/null | wc -l)/36"
guard

say "STAGE B: cage 16-seed ablation (48 runs @ 3M)"
rm -rf cage_final2; mkdir -p cage_final2
$PY - <<'PYEOF' | tee -a ../../outputs/logs/cage_rerun.log
import json, os
GRID=[f"--lr {lr} --ent-coef {e} --net {n}" for lr in ("1e-4","3e-4","1e-3")
      for e in ("0.003","0.02") for n in ("128 64","256 128")]
best={}
for arm in ("none","staleness","sigma"):
    rows=[]
    for i in range(1,13):
        p=f"cage_sweep2/{arm}_c{i}/ablation.json"
        if os.path.exists(p):
            rows.append((json.load(open(p))["results"][arm]["success_%"], GRID[i-1]))
    rows.sort(reverse=True)
    best[arm]=rows[0][1] if rows else GRID[4]
    sp = (max(r[0] for r in rows)-min(r[0] for r in rows)) if rows else 0
    print(f"  cage winner {arm:<10} {rows[0][0] if rows else -1:5.1f}%  spread {sp:5.2f}  {best[arm]}")
    if sp <= 1.0:
        print(f"  !! WARNING {arm}: spread {sp:.2f} <= 1.0 -- policy may not be in the loop")
json.dump(best, open("cage_sweep2/best.json","w"), indent=2)
with open("cage_final2/jobs.txt","w") as fh:
    for s in range(16):
        for arm,hp in best.items():
            fh.write(f"cage_final2/{arm}_s{s}|3000000|{arm}|{s}|{hp}\n")
PYEOF
run_jobs cage_final2/jobs.txt
say "STAGE B done: $(ls cage_final2/*/ablation.json 2>/dev/null | wc -l)/48"
guard
say "=== CAGE RE-RUN COMPLETE ==="
