# Wind: the agreed plan (Sai, 6 Oct 2026)

One phase at a time. Every phase ends at a **gate**: its MJX results (same 1,024 test flights per wind, seed 10000;
checkpoints chosen on the validation yards, seed 20000) and a **Gazebo/PX4 smoke test** (a few matched yards flown by
the candidates and the rule, with the same yards and gusts as MJX), shown to Sai before the next phase starts. Every
training run logs live to ClearML (`rl-landing/wind`). No transfer jump at the end: Gazebo checks at every gate.

Gazebo rules: one simulator at a time (`N=1`), the laptop on AC power, `conditions/wind/gz_ladder.sh` with
`OUT=smoke_phase<k>`; results in `outputs/gazebo_wind/smoke_phase<k>/flights.jsonl`.

## Phase 0: baselines (done 6 Oct; `queue/baselines_6oct.sh`, `compare_baselines.py`)

Landed % on the test flights (crashed % at 18 / 20 m/s):

| arm | W9 | W12 | W15 | W18 | W20 | crashed 18 / 20 |
|---|---|---|---|---|---|---|
| rule (paper) | 100 | 100 | 77.7 | 5.6 | 0.0 | 0.1 / 1.0 |
| tuned rule: gate x2, patience 10 s (`rule_tuned.py`, chosen on validation) | 100 | 100 | 99.6 | 75.7 | 14.5 | 0.2 / 0.5 |
| DOB, no learning | 100 | 100 | 100 | 80.6 | 18.3 | 0.3 / 0.5 |
| paper decider, 3 seeds (learned flyer + PPO decider) | 99.8-100 | 98.2-99.4 | 98.6-99.3 | 93.9-96.2 | 76.8-85.8 | 0.9-2.7 / 3.7-11.4 |
| DOB + PPO decider, 3 seeds (`--flyer dob:1.0`) | 100 | 100 | 99.7-99.8 | 92.5-95.9 | 67.9-75.8 | 1.9-2.1 / 7.5-9.7 |

Learning adds +12 to +20 points at W10 18 over the best no-learning arm and +50 to +70 at 20, at the price of more
crashes there.

Gazebo smoke (`smoke_phase0`, yards 0-1 at W10 18; MJX on the same yards in brackets): rule timeout 2/2 (MJX timeout,
but Gazebo drifts 224-282 m downwind while MJX hovers: the known 30 Sept gap); paper decider s0 landed 2/2, 0.35 and
0.12 m (MJX landed, 0.53 and 0.62 m; 30 Sept Gazebo 0.45 and 0.24 m: the move changed nothing); DOB decider s0 HARD
2/2: on the pad but sliding 0.90 m/s, and 11.9 m off at 2.86 m/s sink, 51 deg tilt, 0.43 m from an obstacle (MJX
landed both in ~29 s). Mechanism: the DOB's push sits at its 3 m/s^2 cap 78-86% of the time in BOTH simulators and
changes 3-8x more per step than the learned flyer's (cap 20-55%); MJX tolerates it, PX4 in Gazebo does not (the 30 Sept
lesson again: a gain built on an actuator at its limit does not transfer).

**Gate decision (Sai, 6 Oct): build on the learned flyer + decider.** The DOB returns in Phase 2: smoothed, gain 0.5,
under realistic sensing, re-flown in Gazebo before reconsidering. New rule for every gate: before any Gazebo flight,
the candidate's push-at-cap duty and step-to-step change are measured in MJX and compared with the paper decider's.

## Phase 1: clean the interface (no heavy GPU)

1. The rule asks 1.5 m/s above 8 m (`land_env.VZ_FAST`), so "rule's own" and "down 1.5" stop being twins
   (identical flights checked 6 Oct).
2. Pick table on the test flights: (rule's choice, learner's choice, height band) for the chosen decider; greedy vs
   sampled at test time.
3. New metrics in `land_env.evaluate`: thrust-saturation duty, command slew, climb/descend reversals, recovery time
   after a gust; an out-of-training wind (W10 22).
Gate: no drift, metrics in place. Gazebo smoke: the chosen base on 2 yards x W10 15 and 18.

## Phase 2: make MJX harder and more honest

1. Sensors: the marker read at a camera rate (~15 Hz) and held between frames, latency 40-120 ms, bursts of dropouts,
   rare outliers; rangefinder noise.
2. The drone per flight: mass +-10%, body drag x0.5-2 (the biggest unknown: `papers/research/x500_wind_limit.md`),
   thrust coefficient +-10%, motor lag +-30%, command latency 0-80 ms.
3. Gusts: direction-change events, gust moments.
4. Re-score rule, tuned rule, DOB and the chosen decider WITHOUT retraining: the robustness table.
Gate: how brittle is today's decider? Gazebo smoke: the same yards as Phase 1, to compare.

## Phase 3: learner upgrades, one at a time (3 seeds each, same test flights, kept only if the paired gain > 0)

1. The base retrained in the Phase 2 environment (the control).
2. Privileged critic: in training only, the critic also sees the true wind and state (the actor is unchanged).
3. Penalties for thrust saturation and command slew (the 30 Sept pulsing failure).
4. The DOB's wind estimate as 2 extra actor inputs.
5. A pad- or wind-aligned frame for inputs and outputs.
Gate: the final recipe. Gazebo smoke after each kept upgrade (2 yards x 2 winds), counting saturation and reversals
before flying.

## Phase 4: architecture ablation (Sai: not satisfied with the common default)

Width 64/128/256/512, depth 1-3, history length 0/4/8/16, tanh vs ReLU, history MLP vs GRU; 2-3 seeds each; ClearML
compare. Gate: the network the paper reports. Gazebo smoke of the winner.

## Phase 5: safety filter and hard cases

A deterministic filter after the actor (rate limits, thrust reserve, fade the push when the marker is lost, abort if
diverging), reporting how often it steps in; oversample hard cases (a gust just before touchdown, a crosswind reversal).
Gazebo smoke of the filtered policy.

## Phase 6: the full Gazebo/PX4 ladder

The finalists and the rule on 20 yards x 4 winds, flight by flight against MJX (`figs.py --pairs`).

## Phase 7: the wind paper

Not followed from the 6 Oct design note (and why): gamma 0.99-0.995 (at 25 decisions/s that looks 4-8 s ahead; we
measured 0.997 making the learner rush into hard landings); "the supervisor alone decides descent" (a tuned gate lands
75.7% at W10 18, the learned decider 92.5-96.2%: keep the learned decider, bound it with the Phase 5 filter); a GRU by
default (an ablation in Phase 4 instead); moving pads, yaw alignment, motor-level control (not our task).
