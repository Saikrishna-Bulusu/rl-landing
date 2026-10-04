#!/bin/bash
# OUTDOOR x500 CAMPAIGN -- the first sigma test with real physics.
#
# x500 from the real PX4 Gazebo model (2.064 kg, arm 0.246 m, T/W 1.69).
# Physics MJC_GND: ground effect ON, drag/downwash OFF (deferred with wind and
# domain randomisation to a v2, so this isolates whether sigma helps at all).
# Policy decides at 5 Hz, cascaded velocity controller flies at 48 Hz.
#
# Preflight passed: spread 32.0 between extreme settings, 100% landed, 32% not
# saturated. An earlier preflight FAILED at 0%/0% because episodes were up to
# 4320 policy steps against a 4096-8192 step rollout -- PPO never saw the
# terminal reward. Action repeat fixed that.
set -u
cd /home/sai/RL_projects/rl-landing
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a out_run.log; }
guard(){
  local r=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)
  local d=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "  guard: ${r} MB RAM, ${d} GB disk"
  if [ "$r" -lt 3000 ] || [ "$d" -lt 5 ]; then say "  ABORT: resources low"; exit 1; fi
}
run_jobs(){
  cat "$1" | xargs -P 8 -I{} bash -c '
    IFS="|" read -r out steps arm seed hp <<< "{}"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
    /home/sai/miniconda3/envs/mujoco_drones/bin/python train_out.py \
      --pool sigma_pool.npz --arm "$arm" --seed "$seed" --steps "$steps" \
      --n-envs 8 --eval-episodes 300 --rho 0.707 --sigma-scale 1.0 \
      --out "$out" $hp > "${out}.log" 2>&1
  '
}
say "=== OUTDOOR x500 CAMPAIGN START ==="; guard

say "STAGE A: HP sweep, 4 arms x 6 candidates @ 400k policy steps (24 runs)"
rm -rf out_sweep; mkdir -p out_sweep; : > out_sweep/jobs.txt
for arm in none staleness sigma random; do i=0
  for lr in 3e-4 1e-3; do for ent in 0.003 0.02; do for net in "128 64"; do
    i=$((i+1)); echo "out_sweep/${arm}_c${i}|400000|${arm}|0|--lr $lr --ent-coef $ent --net $net" >> out_sweep/jobs.txt
  done; done; done
  for lr in 3e-3; do for ent in 0.02; do for net in "256 128"; do
    i=$((i+1)); echo "out_sweep/${arm}_c${i}|400000|${arm}|0|--lr $lr --ent-coef $ent --net $net" >> out_sweep/jobs.txt
  done; done; done
  i=$((i+1)); echo "out_sweep/${arm}_c${i}|400000|${arm}|0|--lr 1e-3 --ent-coef 0.02 --net 256 128" >> out_sweep/jobs.txt
done
run_jobs out_sweep/jobs.txt
say "STAGE A done: $(ls out_sweep/*/ablation.json 2>/dev/null|wc -l)/24"; guard

say "STAGE B: 12-seed ablation, 4 arms @ 800k policy steps (48 runs)"
rm -rf out_final; mkdir -p out_final
$PY - <<'PYEOF' | tee -a out_run.log
import json, os, glob
best={}
for arm in ("none","staleness","sigma","random"):
    rows=[]
    for p in sorted(glob.glob(f"out_sweep/{arm}_c*/ablation.json")):
        r=json.load(open(p))["results"][arm]
        hp=open(p.replace("ablation.json","").rstrip("/")+".log").read() if False else None
        rows.append((r["success_%"], os.path.dirname(p)))
    rows.sort(reverse=True)
    sp=(max(r[0] for r in rows)-min(r[0] for r in rows)) if rows else 0
    best[arm]=rows[0][1] if rows else None
    print(f"  winner {arm:<10} {rows[0][0] if rows else -1:5.1f}%  spread {sp:5.2f}  {best[arm]}")
    if sp<=1.0: print(f"  !! WARNING {arm}: spread {sp:.2f} -- policy may not be in the loop")
    if rows and rows[0][0]>95: print(f"  !! WARNING {arm}: SATURATED")
json.dump(best, open("out_sweep/best_dirs.json","w"), indent=2)
PYEOF
$PY - <<'PYEOF'
import json, re, os
best=json.load(open("out_sweep/best_dirs.json"))
hp={}
for arm,d in best.items():
    log=open(d+".log").read()
    m=re.search(r"--lr (\S+) --ent-coef (\S+) --net ([\d ]+?)\s*--out", log)
    hp[arm]=f"--lr {m.group(1)} --ent-coef {m.group(2)} --net {m.group(3).strip()}" if m else "--lr 1e-3 --ent-coef 0.02 --net 128 64"
json.dump(hp, open("out_sweep/best.json","w"), indent=2)
with open("out_final/jobs.txt","w") as fh:
    for s in range(12):
        for arm,h in hp.items():
            fh.write(f"out_final/{arm}_s{s}|800000|{arm}|{s}|{h}\n")
print("  chosen:", json.dumps(hp, indent=2))
PYEOF
run_jobs out_final/jobs.txt
say "STAGE B done: $(ls out_final/*/ablation.json 2>/dev/null|wc -l)/48"; guard
say "=== OUTDOOR CAMPAIGN COMPLETE ==="
