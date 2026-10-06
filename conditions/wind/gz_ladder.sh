#!/usr/bin/env bash
# gz_ladder.sh -- the Gazebo check of the wind learners: every EPISODE x WIND x DECIDER flown in its own SITL (the wind
# restarts from its seed, so each decider meets the same gusts), N lanes side by side (PX4 instances 1..N).
#   EPISODES="$(seq 0 19)" WINDS="9 12 15 18" DECIDERS="rule conditions/wind/runs/both_s0.npz" N=4 OUT=ladder ./conditions/wind/gz_ladder.sh
# DECIDERS also takes "tuned" (the rule with rule_tuned.py's choice, flights tagged tuned_rule).
# Flights append to outputs/gazebo_wind/$OUT/flights.jsonl; each lane's PX4/Gazebo logs in outputs/gazebo_wind/$OUT/l<k>/.
R=/home/sai/RL_projects/rl-landing; cd $R
N=${N:-4}; OUT=${OUT:?}
jobs=()
for e in ${EPISODES:?}; do for w in ${WINDS:?}; do for d in ${DECIDERS:?}; do jobs+=("$e $w $d"); done; done; done
lane () {
    local k=$1 i=0
    for job in "${jobs[@]}"; do
        if [ $((i % N)) -eq "$k" ]; then
            set -- $job
            tag=$(case "$3" in rule) echo rule;; tuned) echo tuned_rule;; *) basename "$3" .npz;; esac)
            if grep -q "\"episode\": $1, \"w10\": $2.0, \"decider\": \"$tag\"" outputs/gazebo_wind/$OUT/flights.jsonl 2>/dev/null; then
                i=$((i + 1)); continue                                   # flown already: a restarted ladder resumes
            fi
            OUT=$OUT/l$k WORLD=windyard/e$1 SERVER_CONFIG=$R/shared/px4_gazebo/gz_logan/windyard/e$1/w$2.config INSTANCE=$((k + 1)) \
                timeout 900 ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u conditions/wind/gz_windland.py --episode $1 --w10 $2 --decider $3 \
                --out $R/outputs/gazebo_wind/$OUT/flights.jsonl
        fi
        i=$((i + 1))
    done
}
for k in $(seq 0 $((N - 1))); do lane $k & sleep 15; done
wait
echo "$(date +%H:%M) gz_ladder $OUT done: $(grep -c outcome outputs/gazebo_wind/$OUT/flights.jsonl) of ${#jobs[@]} flights" >> outputs/queue/rig.log
