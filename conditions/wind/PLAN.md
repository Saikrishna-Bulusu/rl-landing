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

### Phase 1 results (6 Oct; `phase1_eval.py`, `runs/phase1.json`, `outputs/wind/phase1.png`)

- Greedy beats sampled for every seed (W10 18: 93.8-96.2 vs 90.4-94.9%; W10 20: 77-86 vs 62-66%; sampling adds 1.0-1.3
  climb/descend reversals per s). Greedy stays.
- Twin drift negligible: "down 1.5" below 6 m in at most 2.5% of decisions; "hold" vs "rule's own" stable (0-1%).
- Seed s1 pulses (0.69 reversals/s at W10 18 vs s0 0.13, s2 0.29) and crashes most (3.0% at 18, 11.1% at 20).
- Beyond training (W10 22): learned deciders crash 6-18% where the rules time out (<= 0.1% crashed): Phase 5 needs an
  abort for winds past the drone's limit.
- The decider descends while still approaching (rule holding height: 70% down 0.6, 26% down 1.5) and creeps down at
  0.6 m/s through the closed gate below 6 m (~55%): faster landings (38 vs 79 s), and the risk near obstacles.
- Gazebo smoke (`smoke_phase1`): s0 landed yards 2-3 at W10 15 and yard 2 at 18 (agreeing with MJX); s0 CRASHED on
  yard 3 at 18 (8.9 m off, 0.34 m from an obstacle, tilt 47.5 deg) where MJX lands and where Gazebo itself LANDED on
  30 Sept (0.70 m): one Gazebo flight at 18 m/s is one draw of a chaotic process, not a repeatable result. s1 landed
  yard 0, hard on yard 1 (12.3 m off, tilt 55.9 deg); it did not pulse in Gazebo (0.03-0.05/s). Every Gazebo failure so
  far (paper deciders and DOB) is a blow-away toward an obstacle with tilt past 45 deg; on 30 Sept 15 of 20 Gazebo
  flights at 18 m/s exceeded 45 deg (median 46.4), which MJX's emulated PX4 never does. MJX is more forgiving than PX4
  in Gazebo in strong gusts (96% vs 90% landed at W10 18).

**Gate decisions (Sai, 6 Oct):** the 1.5 m/s fix folds into Phase 3's retraining; Phase 2 becomes: close the
MJX-Gazebo gap first, every Gazebo smoke flight flown 3 times from now on, then the planned randomization.

## Phase 2: make MJX honest, then harder

2a. Close the MJX-Gazebo gap at 18 m/s first (Sai, 6 Oct). Measure the gap with matched tests that remove chance: a
    wind step at hover (steady wind switched on, no gusts) in both simulators at several speeds, comparing peak tilt,
    stray and recovery; then the gust response. Find the cause by switching one candidate at a time in MJX (PX4's
    controller fed estimated rather than true states, where drag and rotor forces act relative to the centre of mass,
    the tilt limit on the setpoint vs the actual attitude, motor saturation and desaturation) until MJX reproduces
    Gazebo's tilt past 45 deg and its blow-aways. Re-check the paper decider in the corrected MJX.
2b. Gazebo spread: every smoke flight flown 3 times from now on.
2c. Then randomize as planned:
1. Sensors: the marker read at a camera rate (~15 Hz) and held between frames, latency 40-120 ms, bursts of dropouts,
   rare outliers; rangefinder noise.
2. The drone per flight: mass +-10%, body drag x0.5-2 (the biggest unknown: `papers/research/x500_wind_limit.md`),
   thrust coefficient +-10%, motor lag +-30%, command latency 0-80 ms.
3. Gusts: direction-change events, gust moments.
4. Re-score rule, tuned rule, DOB and the chosen decider WITHOUT retraining: the robustness table.
Gate: how brittle is today's decider? Gazebo smoke: the same yards as Phase 1, each flight 3 times.

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
