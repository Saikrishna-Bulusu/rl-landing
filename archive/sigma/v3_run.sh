#!/bin/bash
# V3 GUARDRAIL EXPERIMENT. Separate labelled experiment; v1/v2 results stand.
#
# Tests two edge-case guardrails Sai specified:
#   1. outcome ordering  deliver(+100) > RTL(-10) > risky/low-batt(-30)
#      > risky/had-options(-80), so aborting early beats burning the reserve.
#   2. ArUco identity as a HARD SAFETY LAYER outside the policy: a wrong
#      marker is withheld, so it reads as "no marker" and the abort behaviour
#      the policy already has handles it.
#
# Smoke-tested first: at 200k steps the policy aborts 100% of episodes (a real
# local optimum -- RTL is safe, so why try). It escapes by 600k, reaching
# ~79% delivered / 21% RTL / 0% reckless. Anything shorter would have looked
# like total failure.
#
# Stage C is the measurement that matters for the safety layer: the SAME arm
# with id_check off, so the cost of omitting the layer is measured, not asserted.
set -u
cd /home/sai/RL_projects/rl-landing/archive/sigma   # these drivers run from their own folder
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a ../../outputs/logs/v3_run.log; }
guard(){
  local r=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)
  local d=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "  guard: ${r} MB RAM, ${d} GB disk"
  if [ "$r" -lt 3000 ] || [ "$d" -lt 5 ]; then say "  ABORT: resources low"; exit 1; fi
}
run_jobs(){
  cat "$1" | xargs -P 8 -I{} bash -c '
    IFS="|" read -r out steps arm seed idc hp <<< "{}"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
    /home/sai/miniconda3/envs/mujoco_drones/bin/python train_v3.py \
      --pool sigma_pool.npz --arm "$arm" --seed "$seed" --steps "$steps" \
      --n-envs 8 --eval-episodes 400 --t-max 45 --rho 0.707 \
      --sigma-scale 1.75 --id-check "$idc" --out "$out" $hp > "${out}.log" 2>&1
  '
}
say "=== V3 GUARDRAIL RUN START ==="; guard

say "STAGE A: HP sweep at the NEW reward, 4 arms x 12 @ 600k (48 runs)"
rm -rf v3_sweep; mkdir -p v3_sweep; : > v3_sweep/jobs.txt
for arm in none staleness sigma random; do i=0
  for lr in 1e-4 3e-4 1e-3; do for ent in 0.003 0.02; do for net in "128 64" "256 128"; do
    i=$((i+1))
    echo "v3_sweep/${arm}_c${i}|600000|${arm}|0|1|--lr $lr --ent-coef $ent --net $net" >> v3_sweep/jobs.txt
  done; done; done
done
run_jobs v3_sweep/jobs.txt
say "STAGE A done: $(ls v3_sweep/*/ablation.json 2>/dev/null|wc -l)/48"; guard

say "STAGE B+C: 16-seed ablation, 4 arms + sigma-without-safety-layer (80 runs)"
rm -rf v3_final; mkdir -p v3_final
$PY - <<'PYEOF' | tee -a ../../outputs/logs/v3_run.log
import json, os
GRID=[f"--lr {lr} --ent-coef {e} --net {n}" for lr in ("1e-4","3e-4","1e-3")
      for e in ("0.003","0.02") for n in ("128 64","256 128")]
best={}
for arm in ("none","staleness","sigma","random"):
    rows=[]
    for i in range(1,13):
        p=f"v3_sweep/{arm}_c{i}/ablation.json"
        if os.path.exists(p):
            r=json.load(open(p))["results"][arm]
            rows.append((r.get("success_hard_scenes_%", r["success_%"]), GRID[i-1]))
    rows.sort(reverse=True)
    best[arm]=rows[0][1] if rows else GRID[9]
    sp=(max(r[0] for r in rows)-min(r[0] for r in rows)) if rows else 0
    top=rows[0][0] if rows else -1
    print(f"  v3 winner {arm:<10} hard {top:5.1f}%  spread {sp:5.2f}  {best[arm]}")
    if sp<=1.0: print(f"  !! WARNING {arm}: spread {sp:.2f} -- policy may not be in the loop")
    if top>95:  print(f"  !! WARNING {arm}: {top:.1f}% -- SATURATED")
json.dump(best, open("v3_sweep/best.json","w"), indent=2)
with open("v3_final/jobs.txt","w") as fh:
    for s in range(16):
        for arm,hp in best.items():
            fh.write(f"v3_final/{arm}_s{s}|2500000|{arm}|{s}|1|{hp}\n")
        # STAGE C: identical sigma arm, safety layer OFF
        fh.write(f"v3_final/nosafety_s{s}|2500000|sigma|{s}|0|{best['sigma']}\n")
PYEOF
run_jobs v3_final/jobs.txt
say "STAGE B+C done: $(ls v3_final/*/ablation.json 2>/dev/null|wc -l)/80"; guard
say "=== V3 COMPLETE ==="
