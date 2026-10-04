#!/bin/bash
# V2: "how much worse would localisation have to be before sigma pays?"
#
# At the REAL GeoAnchor accuracy the task is saturated: 75.9% of fixes already
# land inside the 17.7 m camera footprint and a dive policy scores 96.4%. So
# "sigma does not help" at scale 1.0 is not informative -- nothing helps,
# because nothing is needed. The discriminating band is 1.5-2.0x:
#     scale 1.0 -> dive 96.4%  hold 100.0%   (saturated)
#     scale 1.5 -> dive 62.8%  hold  95.2%
#     scale 2.0 -> dive 29.6%  hold  60.0%
#     scale 3.0 -> dive  3.6%  hold  12.0%   (floor)
#
# 2 scales x 4 arms (none/staleness/sigma/random) x 16 seeds, equal 12-candidate
# tuning budget per (scale, arm) so no arm and no scale is advantaged.
# Evaluation is BY SCENE; headline is the two scenes with no easy fixes at all
# (Scene_10, Scene_22).
set -u
cd /home/sai/RL_projects/rl-landing
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a v2_curve.log; }
guard(){
  local r=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)
  local d=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "  guard: ${r} MB RAM, ${d} GB disk"
  if [ "$r" -lt 3000 ] || [ "$d" -lt 5 ]; then say "  ABORT: resources low"; exit 1; fi
}
run_jobs(){
  cat "$1" | xargs -P 8 -I{} bash -c '
    IFS="|" read -r out steps arm seed scale hp <<< "{}"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
    /home/sai/miniconda3/envs/mujoco_drones/bin/python train_v2.py \
      --pool sigma_pool.npz --arm "$arm" --seed "$seed" --steps "$steps" \
      --n-envs 8 --eval-episodes 400 --t-max 45 --rho 0.707 \
      --sigma-scale "$scale" --out "$out" $hp > "${out}.log" 2>&1
  '
}
say "=== V2 CURVE FILL (1.75x, 2.25x) ==="; guard

for SC in 1.75 2.25; do
  TAG=$(echo $SC | tr -d .)
  say "STAGE A[$SC]: HP sweep, 4 arms x 12 @ 600k, seed 0 (48 runs)"
  rm -rf v2s${TAG}_sweep; mkdir -p v2s${TAG}_sweep; : > v2s${TAG}_sweep/jobs.txt
  for arm in none staleness sigma random; do i=0
    for lr in 1e-4 3e-4 1e-3; do for ent in 0.003 0.02; do for net in "128 64" "256 128"; do
      i=$((i+1))
      echo "v2s${TAG}_sweep/${arm}_c${i}|600000|${arm}|0|${SC}|--lr $lr --ent-coef $ent --net $net" >> v2s${TAG}_sweep/jobs.txt
    done; done; done
  done
  run_jobs v2s${TAG}_sweep/jobs.txt
  say "STAGE A[$SC] done: $(ls v2s${TAG}_sweep/*/ablation.json 2>/dev/null|wc -l)/48"; guard

  say "STAGE B[$SC]: 16-seed ablation, 4 arms @ 2.5M (64 runs)"
  rm -rf v2s${TAG}_final; mkdir -p v2s${TAG}_final
  TAG=$TAG SC=$SC $PY - <<'PYEOF' | tee -a v2_curve.log
import json, os
TAG=os.environ["TAG"]; SC=os.environ["SC"]
GRID=[f"--lr {lr} --ent-coef {e} --net {n}" for lr in ("1e-4","3e-4","1e-3")
      for e in ("0.003","0.02") for n in ("128 64","256 128")]
best={}
for arm in ("none","staleness","sigma","random"):
    rows=[]
    for i in range(1,13):
        p=f"v2s{TAG}_sweep/{arm}_c{i}/ablation.json"
        if os.path.exists(p):
            r=json.load(open(p))["results"][arm]
            rows.append((r.get("success_hard_scenes_%", r["success_%"]), GRID[i-1]))
    rows.sort(reverse=True)
    best[arm]=rows[0][1] if rows else GRID[9]
    sp=(max(r[0] for r in rows)-min(r[0] for r in rows)) if rows else 0
    top=rows[0][0] if rows else -1
    print(f"  scale {SC} winner {arm:<10} hard-scene {top:5.1f}%  spread {sp:5.2f}  {best[arm]}")
    if sp<=1.0: print(f"  !! WARNING {arm}: spread {sp:.2f} -- policy may not be in the loop")
    if top>95:  print(f"  !! WARNING {arm}: {top:.1f}% -- SATURATED, cannot discriminate")
json.dump(best, open(f"v2s{TAG}_sweep/best.json","w"), indent=2)
with open(f"v2s{TAG}_final/jobs.txt","w") as fh:
    for s in range(16):
        for arm,hp in best.items():
            fh.write(f"v2s{TAG}_final/{arm}_s{s}|2500000|{arm}|{s}|{SC}|{hp}\n")
PYEOF
  run_jobs v2s${TAG}_final/jobs.txt
  say "STAGE B[$SC] done: $(ls v2s${TAG}_final/*/ablation.json 2>/dev/null|wc -l)/64"; guard
done
say "=== V2 COMPLETE ==="
