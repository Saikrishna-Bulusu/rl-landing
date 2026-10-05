#!/usr/bin/env bash
# geo_calm10_30sep.sh -- ten GNSS-denied calm flights for the ACRA paper (mock review 2: "one flight per scenario is a
# demonstration, not a rate"). The rule deciding, GeoAnchor feeding EKF2, the pad markers taking over, no film: the same
# command as geo_calm4 with episodes 0-9 (in --geo only the landing heading changes; route, pad and yard are Logan's).
# One frozen code version for all ten. Gazebo never shares the GPU with training, and one simulator at a time:
# waits for the wind queue (wind_flip2_30sep.sh, PID below) to exit, then for an idle GPU before every flight.
# Stops after flight 0 if the pipeline itself failed (no record), so a broken setup does not burn the night.
#   progress: outputs/queue/geo_calm10.log      results: outputs/gazebo_geo/geo_calm10_e<i>/     done (always, even on a stop): outputs/queue/geo_calm10.done
R=/home/sai/RL_projects/rl-landing; cd $R; LOG=$R/outputs/queue/geo_calm10.log
WAIT_PID=${WAIT_PID:-2048533}
log () { echo "$(date +%H:%M) $*" >> $LOG; }
# the wind session waits on the .done file before its own Gazebo/GPU work: it must appear however this queue ends
trap 'touch $R/outputs/queue/geo_calm10.done' EXIT
trap 'log "stopped by a signal"; exit 143' INT TERM
log "queued, waiting for PID $WAIT_PID"
while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
for e in 0 1 2 3 4 5 6 7 8 9; do
    D=outputs/gazebo_geo/geo_calm10_e$e
    [ -f $D/flights.jsonl ] && continue                        # flown already: a restarted queue resumes
    until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
    log "e$e start"
    OUT=geo_calm10_e$e WORLD=geo_clean INSTANCE=5 timeout 1800 gnss_denied/geofly.sh /usr/bin/python3 -u conditions/wind/gz_windland.py \
        --geo --episode $e --w10 0 --decider rule --out $R/$D/flights.jsonl
    if [ -f $D/flights.jsonl ]; then log "e$e $(tail -1 $D/flights.jsonl | cut -c1-110)"
    else log "e$e NO RECORD: $(tail -2 $D/run.err 2>/dev/null | tr '\n' ' ' | cut -c1-160)"
         [ $e -eq 0 ] && { log "first flight left no record: stopping (check the setup)"; exit 1; }; fi
    sleep 20                                                   # let PX4 and Gazebo release the instance
done
log "done: $(cat outputs/gazebo_geo/geo_calm10_e*/flights.jsonl 2>/dev/null | grep -c '"landed"') of 10 landed"
