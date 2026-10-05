#!/bin/bash
# CONTROL EXPERIMENT, 19 Sept.
#
# The cage result showed BOTH extra channels beating `none`, yet the staleness
# channel is byte-identical across episodes and correlates +0.003 with the
# actual error. A zero-information channel cannot confer uncertainty awareness,
# so the gain must come from somewhere else. These two arms separate it:
#
#   random -- same distribution as the sigma channel (median 0.262, range
#             [0.17,0.55]) but drawn from an INDEPENDENT bin, so it correlates
#             +0.016 with the episode error against sigma's +0.549. If this
#             beats `none`, the cage result is an input-shape artefact.
#   clock  -- bare elapsed time. If this reproduces the staleness gain, that
#             gain was timing.
#
# Both arms get the SAME 12-candidate tuning budget the real arms received.
# Skipping it would under-tune the control, make it score low, and falsely
# confirm the hypothesis -- the exact asymmetry this project keeps hitting.
set -u
cd /home/sai/RL_projects/rl-landing/archive/tello   # these drivers run from their own folder
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a ../../outputs/logs/cage_control.log; }
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
      --n-envs 8 --eval-episodes 400 --rho 0.707 --out "$out" $hp > "${out}.log" 2>&1
  '
}
say "=== CONTROL EXPERIMENT START ==="; guard

say "STAGE A: control HP sweep (24 runs @ 1.5M, seed 0)"
rm -rf ctrl_sweep; mkdir -p ctrl_sweep; : > ctrl_sweep/jobs.txt
for arm in random clock; do i=0
  for lr in 1e-4 3e-4 1e-3; do for ent in 0.003 0.02; do for net in "128 64" "256 128"; do
    i=$((i+1))
    echo "ctrl_sweep/${arm}_c${i}|1500000|${arm}|0|--lr $lr --ent-coef $ent --net $net" >> ctrl_sweep/jobs.txt
  done; done; done
done
run_jobs ctrl_sweep/jobs.txt
say "STAGE A done: $(ls ctrl_sweep/*/ablation.json 2>/dev/null | wc -l)/24"; guard

say "STAGE B: control 16-seed ablation (32 runs @ 3M)"
rm -rf ctrl_final; mkdir -p ctrl_final
$PY - <<'PYEOF' | tee -a ../../outputs/logs/cage_control.log
import json, os
GRID=[f"--lr {lr} --ent-coef {e} --net {n}" for lr in ("1e-4","3e-4","1e-3")
      for e in ("0.003","0.02") for n in ("128 64","256 128")]
best={}
for arm in ("random","clock"):
    rows=[]
    for i in range(1,13):
        p=f"ctrl_sweep/{arm}_c{i}/ablation.json"
        if os.path.exists(p):
            rows.append((json.load(open(p))["results"][arm]["success_%"], GRID[i-1]))
    rows.sort(reverse=True)
    best[arm]=rows[0][1] if rows else GRID[4]
    sp=(max(r[0] for r in rows)-min(r[0] for r in rows)) if rows else 0
    print(f"  control winner {arm:<8} {rows[0][0] if rows else -1:5.1f}%  spread {sp:5.2f}  {best[arm]}")
    if sp <= 1.0: print(f"  !! WARNING {arm}: spread {sp:.2f} -- policy may not be in the loop")
json.dump(best, open("ctrl_sweep/best.json","w"), indent=2)
with open("ctrl_final/jobs.txt","w") as fh:
    for s in range(16):
        for arm,hp in best.items():
            fh.write(f"ctrl_final/{arm}_s{s}|3000000|{arm}|{s}|{hp}\n")
PYEOF
run_jobs ctrl_final/jobs.txt
say "STAGE B done: $(ls ctrl_final/*/ablation.json 2>/dev/null | wc -l)/32"; guard
say "=== CONTROL EXPERIMENT COMPLETE ==="
