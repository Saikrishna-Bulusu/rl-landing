#!/bin/bash
# FIXED-HP DEGRADATION CURVE.
#
# The previous curve was uninterpretable: each scale got its OWN max-of-12
# single-seed sweep, so every point carried independent winner's-curse noise.
# That noise exceeded the difficulty effect -- 1.75x scored LOWER than 2.25x for
# three of four arms, which is impossible if difficulty drove the numbers.
#
# Here each arm uses ONE hyperparameter set at every scale, selected by mean
# rank across all four existing sweeps (48 data points per arm, not 12). Three
# of four points therefore run on settings tuned for a different difficulty --
# a handicap, but applied EQUALLY to every arm, so arm comparisons stay fair and
# the curve becomes readable across scales for the first time.
set -u
cd /home/sai/RL_projects/rl-landing/archive/sigma   # these drivers run from their own folder
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a ../../outputs/logs/v2_fixed.log; }
guard(){
  local r=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)
  local d=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "  guard: ${r} MB RAM, ${d} GB disk"
  if [ "$r" -lt 3000 ] || [ "$d" -lt 5 ]; then say "  ABORT: resources low"; exit 1; fi
}
say "=== FIXED-HP CURVE START (4 scales x 4 arms x 16 seeds = 256 runs) ==="; guard
rm -rf v2fix; mkdir -p v2fix
$PY - <<'PYEOF'
import json
hp=json.load(open("fixed_hp.json"))
with open("v2fix/jobs.txt","w") as fh:
    for sc in ("1.5","1.75","2.0","2.25"):
        t=sc.replace(".","")
        for s in range(16):
            for arm,h in hp.items():
                fh.write(f"v2fix/s{t}_{arm}_s{s}|2500000|{arm}|{s}|{sc}|{h}\n")
print("  jobs:", sum(1 for _ in open("v2fix/jobs.txt")))
for k,v in hp.items(): print(f"    {k:<10} {v}")
PYEOF
cat v2fix/jobs.txt | xargs -P 8 -I{} bash -c '
  IFS="|" read -r out steps arm seed scale hp <<< "{}"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  /home/sai/miniconda3/envs/mujoco_drones/bin/python train_v2.py \
    --pool sigma_pool.npz --arm "$arm" --seed "$seed" --steps "$steps" \
    --n-envs 8 --eval-episodes 400 --t-max 45 --rho 0.707 \
    --sigma-scale "$scale" --out "$out" $hp > "${out}.log" 2>&1
'
say "DONE: $(ls v2fix/*/ablation.json 2>/dev/null|wc -l)/256"; guard
say "=== FIXED-HP CURVE COMPLETE ==="
