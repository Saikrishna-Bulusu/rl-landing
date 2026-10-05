#!/usr/bin/env bash
# run_overnight.sh -- the unattended run. Start it, go to sleep.
#
# Two job families, from one queue so a stall in either cannot starve the other:
#
#   obst_*   the new question: trees make the search underivable, so does a
#            learned policy beat the best hand-written avoidance rule?
#   cons_*   the old question, finally powered: sigma-aware vs sigma-blind at
#            enough seeds to survive the scrutiny that killed the 8-seed result.
#
# RESUME IS THE POINT. Every completed run drops an ablation.json, and `resume`
# rebuilds the queue from what is MISSING. So the watchdog can kill a job for
# memory, or the whole launcher can die, and restarting loses only the run that
# was in flight -- not the night.
set -u
cd /home/sai/RL_projects/rl-landing/archive/sigma   # these drivers run from their own folder
PY=/home/sai/miniconda3/envs/mujoco_drones/bin/python
OUT=overnight
PAR=10                      # Sai's cap: 10 jobs, keep 3 GB free
#
# REGIME, CHOSEN BY MEASUREMENT RATHER THAN TASTE (headroom.py):
#   35 s budget, sigma x3.5, 8 trees, canopies up to 5 m, crowding the pad.
#   There the best hand-written rule manages 80.5% against a 99.0% ceiling --
#   18.5 points of room. Trees alone were NOT enough: with a 45-60 s budget the
#   rule scored 91-99% whatever the crowding, because a vertical descent onto a
#   point that is not inside a trunk always works. Trees only start to cost
#   something once the budget is tight enough that the time they add matters.
MIN_MB=3000
mkdir -p "$OUT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OUT/launcher.log"; }

build_queue(){
  : > "$OUT/jobs_all.txt"
  # --- obstacles: 5M steps each, because this regime learns SLOWLY ---------
  #
  # A 700k-step probe reached 29.5% against a rule at 80.5%, climbing but
  # nowhere near converged -- sigma x3.5 makes the returns very noisy. Fewer
  # seeds that actually converge beat many that stop halfway, so the budget
  # goes into steps. The ent variant hedges the one hyperparameter that
  # matters most when exploration is the bottleneck.
  for s in 0 1 2; do
    echo "$OUT/obst_full_s${s}|sigma|1|1|1|1|5000000|0.003" >> "$OUT/jobs_all.txt"
    echo "$OUT/obst_ent_s${s}|sigma|1|1|1|1|5000000|0.02"   >> "$OUT/jobs_all.txt"
    echo "$OUT/obst_noobst_s${s}|sigma|0|1|1|1|5000000|0.003" >> "$OUT/jobs_all.txt"
    echo "$OUT/obst_blind_s${s}|none|1|1|1|1|5000000|0.003" >> "$OUT/jobs_all.txt"
  done
  # --- consolidation: no trees, easier regime, converges fast ---------------
  for s in 0 1 2 3 4; do
    for arm in sigma none random; do
      echo "$OUT/cons_${arm}_s${s}|${arm}|0|1|1|1|2500000|0.003" >> "$OUT/jobs_all.txt"
    done
  done
  say "queue built: $(wc -l < "$OUT/jobs_all.txt") runs"
}

remaining(){
  : > "$OUT/jobs_remaining.txt"
  while IFS='|' read -r out rest; do
    [ -f "$out/ablation.json" ] || echo "$out|$rest" >> "$OUT/jobs_remaining.txt"
  done < "$OUT/jobs_all.txt"
  say "remaining: $(wc -l < "$OUT/jobs_remaining.txt")"
}

[ "${1:-start}" = "resume" ] || build_queue
[ -f "$OUT/jobs_all.txt" ] || build_queue
remaining
rm -f "$OUT/ALLDONE"

cat "$OUT/jobs_remaining.txt" | xargs -P "$PAR" -I{} bash -c '
  IFS="|" read -r out arm obst en hi st steps ent <<< "{}"
  # Do not start a new job into a machine that is already tight. Sai is asleep
  # and his browser and VM are still holding memory; the run yields, not him.
  A=$(awk "/MemAvailable/{printf \"%d\",\$2/1024}" /proc/meminfo)
  if [ "$A" -lt '"$MIN_MB"' ]; then
    echo "[skip] $out: only ${A} MB free" >> '"$OUT"'/launcher.log
    exit 0
  fi
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 \
  '"$PY"' train_v5.py --pool sigma_pool.npz --arm "$arm" \
    --seed "${out##*_s}" --steps "$steps" --n-envs 8 --eval-episodes 600 \
    --t-max 35 --rho 0.707 --sigma-scale 3.5 \
    --obstacles "$obst" --energy "$en" --history "$hi" --steer "$st" \
    --n-obst 8 --obst-h 12 --pad-clear 0.5 --obst-r 5.0 \
    --lr 3e-4 --ent-coef "$ent" --net 128 64 --out "$out" > "${out}.log" 2>&1
'
say "launcher finished; $(ls "$OUT"/*/ablation.json 2>/dev/null | wc -l) runs complete"
if [ "$(wc -l < "$OUT/jobs_remaining.txt")" -gt 0 ]; then
  remaining
  [ -s "$OUT/jobs_remaining.txt" ] || touch "$OUT/ALLDONE"
fi
