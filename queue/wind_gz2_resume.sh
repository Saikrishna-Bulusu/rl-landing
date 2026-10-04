#!/usr/bin/env bash
# wind_gz2_resume.sh -- resumes the Gazebo ladder of wind_gz2_30sep.sh, stopped 30 Sept 15:18 at 69 of 160 flights (Sai took the
# laptop home). gz_ladder.sh skips every flight already in gazebo_wind/ladder_rule/flights.jsonl, so this flies only the rest.
# One SITL; waits for an idle GPU. Done marker: queue/wind_gz2_30sep.done
R=/home/sai/RL_projects/rl-landing; cd $R; LOG=$R/queue/wind_flip.log
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "$(date +%H:%M) gz2 ladder resumed" >> $LOG
EPISODES="$(seq 0 19)" WINDS="9 12 15 18" DECIDERS="wind/runs/stage2_c100h100f02g_s1_best.npz wind/runs/stage2_c100h100g_s0_best.npz" \
    N=1 OUT=ladder_rule ./wind/gz_ladder.sh
echo "$(date +%H:%M) gz2 ladder done" >> $LOG
touch $R/queue/wind_gz2_30sep.done
