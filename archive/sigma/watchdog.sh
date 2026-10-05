#!/usr/bin/env bash
# watchdog.sh -- hourly health check for the overnight run.
#
# Sai is asleep. The rules he set: check every hour, and if something is wrong,
# save what exists, fix it, and restart. So this does not just observe -- it
# acts, and it writes down what it did so the morning report is not guesswork.
#
# What can actually go wrong overnight, in order of likelihood:
#   1. memory pressure from the browsers/VM that were already holding 14 GB
#   2. a job dying silently and leaving a hole in the seed table
#   3. the whole launcher dying, leaving nothing running at all
set -u
cd /home/sai/RL_projects/rl-landing/archive/sigma   # these drivers run from their own folder
LOG=overnight/watchdog.log
mkdir -p overnight
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

avail_mb(){ awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo; }
disk_gb(){ df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
# Count JOBS, not processes. Each trainer forks vectorised-env workers that
# carry the same command line, so a plain pgrep -c reported 23 for 10 jobs and
# would have made the shed-on-memory logic act on a number that means nothing.
running(){ ps -eo args | grep "[t]rain_v5.py" \
             | grep -oE -- "--out overnight/[A-Za-z0-9_]+" \
             | sort -u | wc -l; }

say "check: $(running) jobs, $(avail_mb) MB RAM free, $(disk_gb) GB disk, load $(cut -d' ' -f1 /proc/loadavg)"

# --- resource guards ---------------------------------------------------
if [ "$(disk_gb)" -lt 10 ]; then
  say "  DISK LOW (<10 GB): stopping the run to protect the machine"
  pkill -f "train_v5.py"; exit 0
fi

A=$(avail_mb)
if [ "$A" -lt 1500 ]; then
  # Below the floor Sai asked me to keep. Shed the newest jobs rather than
  # let the OOM killer pick something of his.
  N=$(running)
  if [ "$N" -gt 2 ]; then
    VICTIM=$(ps -eo pid,args | grep "[t]rain_v5.py --pool" | awk '{print $1}' | tail -1)
    say "  RAM ${A} MB < 1500: killing newest job $VICTIM (was $N running)"
    kill "$VICTIM" 2>/dev/null
  else
    say "  RAM ${A} MB < 1500 with only $N jobs left -- not shedding further"
  fi
fi

# --- liveness ----------------------------------------------------------
if [ "$(running)" -eq 0 ]; then
  if [ -f overnight/ALLDONE ]; then
    say "  all jobs finished (ALLDONE present)"
  elif [ -f overnight/jobs_remaining.txt ] && [ -s overnight/jobs_remaining.txt ]; then
    say "  NOTHING RUNNING but work remains -- restarting the launcher"
    nohup ./run_overnight.sh resume >> overnight/launcher.log 2>&1 &
  else
    say "  nothing running and no work queued"
  fi
fi

# --- progress ----------------------------------------------------------
DONE=$(ls overnight/*/ablation.json 2>/dev/null | wc -l)
TOT=0; [ -f overnight/jobs_all.txt ] && TOT=$(wc -l < overnight/jobs_all.txt)
say "  progress: $DONE / $TOT runs complete"
