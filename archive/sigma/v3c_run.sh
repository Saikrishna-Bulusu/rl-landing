#!/bin/bash
# STAGE C RE-RUN: what does the ArUco identity safety layer actually buy?
#
# The first attempt measured nothing. The guidance hardcoded `target = self.pad`
# whenever anything was acquired, so a drone fooled by a neighbour's marker
# still flew to the correct address -- measured: placed ON a decoy 21.8 m from
# the pad, it flew 21.6 m to the real pad. Being deceived was free, and
# id_check on/off scored 91.3% vs 90.3%, i.e. identical.
#
# Fixed: approach_env gained a _visual_target() hook (default = self.pad, so v1
# and v2 are bit-identical -- verified by re-evaluating a trained policy to its
# exact stored score). v3 overrides it: if the only marker in frame is the
# neighbour's, the drone flies THERE. Now verified:
#     no layer  -> acquired=True,  ends 0.0 m from the DECOY, 21.8 m from the pad
#     layer on  -> acquired=False, ends 0.2 m from the PAD
#
# BOTH conditions are re-run, not just the broken one, so the comparison is
# entirely within one code version.
set -u
cd /home/sai/RL_projects/rl-landing
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a v3c_run.log; }
guard(){
  local r=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)
  local d=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  say "  guard: ${r} MB RAM, ${d} GB disk"
  if [ "$r" -lt 3000 ] || [ "$d" -lt 5 ]; then say "  ABORT: resources low"; exit 1; fi
}
say "=== STAGE C RE-RUN (deception now has a consequence) ==="; guard
HP=$(python3 -c "import json;print(json.load(open('v3_sweep/best.json'))['sigma'])")
say "  using the sigma arm's tuned HP: $HP"
rm -rf v3c; mkdir -p v3c; : > v3c/jobs.txt
for s in $(seq 0 15); do
  echo "v3c/safe_s${s}|1|${s}|$HP"   >> v3c/jobs.txt
  echo "v3c/unsafe_s${s}|0|${s}|$HP" >> v3c/jobs.txt
done
cat v3c/jobs.txt | xargs -P 8 -I{} bash -c '
  IFS="|" read -r out idc seed hp <<< "{}"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  /home/sai/miniconda3/envs/mujoco_drones/bin/python train_v3.py \
    --pool sigma_pool.npz --arm sigma --seed "$seed" --steps 2500000 \
    --n-envs 8 --eval-episodes 400 --t-max 45 --rho 0.707 \
    --sigma-scale 1.75 --id-check "$idc" --out "$out" $hp > "${out}.log" 2>&1
'
say "DONE: $(ls v3c/*/ablation.json 2>/dev/null|wc -l)/32"; guard
say "=== STAGE C COMPLETE ==="
