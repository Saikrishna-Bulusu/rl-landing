#!/usr/bin/env bash
# Phase 2 gate (PLAN.md 2b): Phase 1's six Gazebo smoke flights, each 3 times, one SITL at a time. --log-obs keeps every
# decision's estimates (p_est, v_est) beside Gazebo's truth: the sensing errors 2c's MJX randomization is fitted to.
# gz_ladder.sh cannot repeat a flight (it resumes by skipping flown ones), so each repeat has its own folder.
#   nohup conditions/wind/queue/smoke_phase2.sh > outputs/queue/smoke_phase2.log 2>&1 &      (~75 min)
R=/home/sai/RL_projects/rl-landing; cd $R
for r in 1 2 3; do
    for job in "0 18 s1" "1 18 s1" "2 15 s0" "2 18 s0" "3 15 s0" "3 18 s0"; do
        set -- $job
        OUT=smoke_phase2/r$r/l0 WORLD=windyard/e$1 SERVER_CONFIG=$R/shared/px4_gazebo/gz_logan/windyard/e$1/w$2.config INSTANCE=1 \
            KEEP_ULG=1 timeout 900 ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u conditions/wind/gz_windland.py --episode $1 \
            --w10 $2 --decider conditions/wind/runs/stage2_c100h100g_$3_best.npz --log-obs \
            --out $R/outputs/gazebo_wind/smoke_phase2/r$r/flights.jsonl
        echo "$(date +%H:%M) r$r e$1 w$2 $3: $(tail -1 outputs/gazebo_wind/smoke_phase2/r$r/flights.jsonl 2>/dev/null | cut -c1-90)"
    done
done
echo "$(date +%H:%M) smoke_phase2 done: $(cat outputs/gazebo_wind/smoke_phase2/r*/flights.jsonl | grep -c outcome) of 18 flights" >> outputs/queue/rig.log
