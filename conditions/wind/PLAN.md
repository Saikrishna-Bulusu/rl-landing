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
  far (paper deciders and DOB) is a blow-away toward an obstacle with tilt past 45 deg. Tilt past 45 deg alone is NOT
  the gap (corrected 6 Oct: MJX also exceeds it in 86-95% of its W10 18 flights, median 45.8-46.0, peak 48.2 deg;
  Gazebo 30 Sept median 46.4): the gap is the TAIL, Gazebo's failures reach 47.5-56 deg and drift 9-12 m, MJX never
  passes 48.2 deg. MJX is more forgiving than PX4 in Gazebo in strong gusts (96% vs 90% landed at W10 18).

**Gate decisions (Sai, 6 Oct):** the 1.5 m/s fix folds into Phase 3's retraining; Phase 2 becomes: close the
MJX-Gazebo gap first, every Gazebo smoke flight flown 3 times from now on, then the planned randomization.

## Phase 2: make MJX honest, then harder

2a. Close the MJX-Gazebo gap at 18 m/s first (Sai, 6 Oct). Measure the gap with matched tests that remove chance: a
    wind step at hover (steady wind switched on, no gusts) in both simulators at several speeds, comparing peak tilt,
    stray and recovery; then the gust response. Find the cause by switching one candidate at a time in MJX (PX4's
    controller fed estimated rather than true states, where drag and rotor forces act relative to the centre of mass,
    the tilt limit on the setpoint vs the actual attitude, motor saturation and desaturation) until MJX reproduces
    Gazebo's tail of large tilts (past ~48 deg) and its blow-aways. Re-check the paper decider in the corrected MJX.
### Phase 2a results (6 Oct; `x500_mjx.py --check` runs `check_hold()`; Gazebo holds in `outputs/gazebo_wind/hold/`)

Matched test: steady wind at 30 m, no turbulence or gusts, velocity setpoint 0 (`px4_steps.py --hover 60 --alt 30`
under gz_winddrag; `sitl.sh KEEP_ULG=1` keeps PX4's log). Lean and drift over the last 30 s:

| wind at 30 m | comes from (deg off the nose) | Gazebo | MJX before the fix | MJX after |
|---|---|---|---|---|
| 20 m/s | 230 (back-right arm) | 31.7 deg, 0.07 m/s | 31.6, 0 | 31.6, 0 |
| 22 m/s | 230 | 36.1, 0.10 | 35.9, 0 | 35.9, 0 |
| 24 m/s | 227 | 37.7, **1.50** (blown away) | 40.2, 0 | 37.3, **1.28** |
| 26 m/s | 223 | 38.9, **3.09** | 44.2, 0 | 38.0, **2.93** |
| 24 m/s | 89 (left side) | 36.6, **1.57** | | 36.5, **1.64** |
| 24 m/s | 320 / 49 (front arms) | 39.9, 0.11 | | 40.2, 0 |
| 26 m/s | 356 (nose) | 43.9, 0.15 | | 44.2, 0 |

The four motor commands now match PX4's own log within 0.003 at three wind directions.

Cause, found by elimination: not the heading alone, not gz-sim 8.15's motor model (its source has no thrust loss with
speed, that was gazebo-classic, and its unsigned rolling moment was already copied), not the hover throttle (calm
hover collective 0.836 in both). It was where forces act: gz_winddrag pushes the body drag at base_link's CoM
(`Link::AddWorldForce`), 1.6 cm ABOVE the whole drone's CoM because the gimbal hangs below, and the CoM sits 3.4 mm
ahead of the rotor centre (the SDF masses and a calm hover's front-back motor split, +0.0099 per motor, agree; the
6.8 mm noted on 30 Sept was wrong). MJX had both at the CoM. Fixed: `COM_X`, `DRAG_AT`; the unused `TQ_BODY` and
`SPEED_LOSS` knobs are gone.

What it means: the drone's steady wind limit depends on where the wind comes from, in both simulators. MJX, all
eight directions: nose 26 m/s, front arms and behind 25, sides 22, back arms 21-23. Side winds cost through yaw: the
drag acts 3.4 mm behind the CoM, propellers have little yaw authority, and PX4 cuts thrust to keep its heading.
Facing the wind is a cheap candidate upgrade (Phase 3 or 5).

End to end, the same 20 test yards flown in the Gazebo ladder (30 Sept) and in MJX, worst stray from the pad after
first reaching it, median / worst flight (`stray_check.py`):

| arm, wind | Gazebo | MJX before the fix | MJX after |
|---|---|---|---|
| rule, W10 15 | 2.0 / 29 m | 2.0 / 12 m | 2.0 / 34 m |
| rule, W10 18 | 3.4 / 286 m | 2.0 / 20 m | 4.0 / 325 m |
| paper decider s0, W10 15 | 2.0 / 7.3 m | 2.0 / 2.0 m | 2.0 / 5.2 m |
| paper decider s0, W10 18 | 2.0 / 36 m | 2.0 / 8.7 m | 2.0 / 40 m |

The tail the old twin could not produce (blow-aways) is now there, in the same flights. Gap closed.

Flight by flight on those yards (rule, paper decider s0, stage2_c100 s0, fly_hp_s0 at W10 15 and 18; 158 Gazebo
flights): Gazebo landed 107, the old twin 120, the corrected twin 110. The same ending in both simulators: 127 -> 125
of 158. Single flights near the limit stay a coin toss (Gazebo itself did not repeat yard 3 between 30 Sept and
6 Oct), which is what 2b's three repeats are for; the twin's totals are now right where they were optimistic.

The paper decider re-scored on the corrected twin (same 1,024 test flights per wind, `runs/twin_fix_6oct.json`),
landed % (crashed %), old twin -> corrected:

| arm | W10 15 | W10 18 | W10 20 |
|---|---|---|---|
| rule | 77.7 -> 65.5 | 5.6 -> 2.6 | 0 -> 0 |
| tuned rule | 99.6 -> 95.0 | 75.7 -> 41.0 | 14.5 -> 5.1 |
| paper decider s0 | 98.6 -> 98.8 | 96.2 (1.0) -> 88.2 (4.3) | 85.8 (6.5) -> 58.4 (8.8) |
| paper decider s1 | 99.3 -> 97.9 (0.9) | 93.9 (2.7) -> 81.9 (8.4) | 76.8 (11.4) -> 50.9 (15.4) |
| paper decider s2 | 98.7 -> 98.2 (0.6) | 96.1 (0.9) -> 83.8 (2.8) | 78.3 (3.7) -> 50.0 (5.0) |

W10 9 and 12 are unchanged (>= 98%). Learning's lead over the tuned rule at W10 18 grows from ~20 to 41-47 points,
but the learned deciders crash more (up to 8.4%); a motor is saturated 32-40% of their flying time at W10 18 (rules
52-60%). Retraining on the corrected twin is Phase 3's control; the crashes are Phase 5's job.

**Gate decision (Sai, 6 Oct):** continue with 2b and 2c as planned; nose-into-wind becomes Phase 3 candidate 6.

Also fixed: `sitl.sh` passed a relative `SERVER_CONFIG` to gz, which runs in PX4's folder and silently found no
config (no wind, no scene: "Waiting for Gazebo world"). Every queue script used absolute paths, so no earlier flight
was affected.

2b. Gazebo spread: every smoke flight flown 3 times from now on.
2c. Then randomize as planned:
1. Sensors: the marker read at a camera rate (~15 Hz) and held between frames, latency 40-120 ms, bursts of dropouts,
   rare outliers; rangefinder noise.
2. The drone per flight: mass +-10%, body drag x0.5-2 (the biggest unknown: `papers/research/x500_wind_limit.md`),
   thrust coefficient +-10%, motor lag +-30%, command latency 0-80 ms.
3. Gusts: direction-change events, gust moments.
4. Re-score rule, tuned rule, DOB and the chosen decider WITHOUT retraining: the robustness table.
Gate: how brittle is today's decider? Gazebo smoke: the same yards as Phase 1, each flight 3 times.

### Phase 2b-2c as built (6 Oct)

2b. The gate's Gazebo smoke, Phase 1's six flights 3 times each (`queue/smoke_phase2.sh`, with `--log-obs`): 15 of 18
    landed; all three failures are one case, yard 1 at W10 18 with decider s1 (hard, crashed, hard; hard in Phase 1).

2c. `land_env.RAND` (empty = the Phase 0-2a environment, bit for bit: `evaluate()` outputs compared exactly):
- "sense" (`sensors_new`), fitted to Gazebo's own errors (gz_windland --log-obs against truth). W10 9: pad bias
  0.02-0.05 m, slow error sd 0.01-0.05 m over 0.2-1 s, velocity sd 0.06-0.12 m/s, no visible delay. W10 15-18 gate
  flights: bias up to 0.37 m, sd up to 0.13 m over 1-10 s, velocity sd up to 0.29 m/s (the lean and the 5 Hz gimbal).
  Per flight: bias sd 0.02-0.2 m, slow error 1-6 x (0.01 + 0.001 h) over 0.3-5 s, dropouts (~1 per 20 s, 0.5-3 s, the
  estimate drifts), one-frame outliers (0.3-1 m, ~1 per 13 s), height bias and slow error, velocity noise 0.05-0.3 m/s,
  0-120 ms delay. Advanced once per decision in step(), so the rule and the learner see the same estimate (as in
  Gazebo; before, each drew its own white noise).
- "drone" (`drone_new`, `x500_mjx.PER_DRONE`): mass and thrust coefficient x0.9-1.1, motor lag x0.7-1.3, body drag
  x0.5-2, 0-80 ms command delay. PX4's hover throttle follows mass and thrust (heavy + weak hovers at 94%).
- "gust": the mean wind turns up to 60 deg over 3-10 s, about once per 30 s. Gust moments are not added: since 2a
  the drag acts 1.6 cm above the CoM, so gusts already twist the drone as in Gazebo.
- Checks: `python conditions/wind/land_env.py --check-rand` (each group behaves as promised) and
  `x500_mjx.py --check` (check_drone: every corner of the fleet hovers level at its expected throttle).
- `evaluate()` takes a list of winds: one compile per list (about 14 s), about 3 s per wind.
- The tuned rule now also flies in Gazebo (`gz_windland.py --decider tuned`, checked against land_env's). On the
  ladder's 20 yards and gusts (`outputs/gazebo_wind/ladder_rule`): W10 15 landed 20/20, W10 18 7/20 (12 gave up, 1
  hard). Same yards: rule 12/19 and 0/19, paper decider s0 19/20 and 18/20. MJX said 95 / 41 (tuned), 66 / 3 (rule),
  98 / 88 (decider s0): the ranking and the gaps carry over.

Robustness, without retraining (`robustness.py` -> `runs/robustness_2c.json`, ClearML "phase 2c"). Landed % at W10 18
(the learned decider as the mean of its 3 seeds, crashed in brackets):

| condition | rule | tuned rule | DOB 0.5 | learned decider |
|---|---|---|---|---|
| nominal | 2.6 | 41.0 | 42.9 | 84.6 (5.2) |
| sensing | 0.8 | 41.4 | 20.3 | 78.2 (6.1) |
| drone | 15.5 | 38.7 | 39.2 | 61.1 (4.3) |
| gusts (wind turns) | 1.8 | 39.5 | 41.7 | 84.9 (5.5) |
| all three | 3.9 | 37.2 | 24.4 | 55.4 (5.9) |

At W10 15 under all three: rule 21.4, tuned 67.4, DOB 50.0, learned 79.4 (3.7 crashed); at W10 12 the learned
deciders now crash 1.2-2.1% (0 nominal). Readings:
- Sensing breaks the plain rule (W10 15: 65.5 -> 22.7; its strict gate waits on noisy estimates) and the DOB (it reads
  the noisy velocities); the tuned rule shrugs it off; the learned decider loses about 6 points.
- Turning wind changes nothing much.
- Drone differences cost the learned decider 23 points, and it is physics, not fragility: by drone (seed 0, W10 18)
  body drag x0.5-0.7 lands 94%, x1.4-2 lands 16%; hover throttle 0.74-0.80 lands 80%, 0.88-0.95 lands 32%; motor lag
  and command delay (0-80 ms) change nothing. A draggier or heavier x500 cannot hold a gale; pushing on regardless is
  where its crashes come from. Body drag, the one number nobody measured, is the biggest uncertainty again.
- The learned decider stays best in every condition at W10 >= 12, but its lead over the tuned rule at W10 18 shrinks
  from 44 to 18 points under all three.

Gate (6 Oct): today's decider is not fragile to sensing, timing or turning wind; it is to a weaker drone, as any
controller must be, but it crashes instead of giving up. For Phase 3: retrain in the randomized environment (the
control), and let it see what it cannot see now (the DOB's push estimate, candidate 4, is a direct readout of the
drone's drag). For Phase 5: abort when the drone cannot hold the wind.

## Phase 3: learner upgrades, one at a time (3 seeds each, same test flights, kept only if the paired gain > 0)

1. The base retrained in the Phase 2 environment (the control).
2. Privileged critic: in training only, the critic also sees the true wind and state (the actor is unchanged).
3. Penalties for thrust saturation and command slew (the 30 Sept pulsing failure).
4. The DOB's wind estimate as 2 extra actor inputs.
5. A pad- or wind-aligned frame for inputs and outputs.
6. Nose into the wind (Sai, 6 Oct, from Phase 2a): the drone holds 26 m/s on the nose but 22 from a side, so yaw
   the nose toward the estimated wind during the approach; first as a rule on top of the decider, then (if it pays)
   as something the learner chooses.
Gate: the final recipe. Gazebo smoke after each kept upgrade (2 yards x 2 winds), counting saturation and reversals
before flying.

### Phase 3 as run (from 7 Oct; Sai: "Phase 3 as planned, DOB first")

- `ppo.py --rand all` trains and validates in the Phase 2 world; checkpoints record `rand` (and `dob_in`).
- The control (`queue/phase3_base.sh`): the paper's recipe, flyer `fly_rand` then deciders `base_rand` s0-s2. One
  change was forced: with randomized drones the curriculum's 95% bar is out of reach in strong wind (some drones
  cannot hold it), and with 150-round steps the first flyer was still at 12 m/s after 550 rounds (the paper's reached
  20 by 500). Now `--w-ok 0.8 --stuck 40`: the wind rises at 80% landed or after 40 stalled rounds.
- Upgrade 1 (`queue/phase3_dobin.sh`): `--dob-in`, the DOB's estimate of the wind's push as 2 more decider inputs
  (`ppo.with_dob`, mirrored in gz_windland and checked by `--check-obs`), same frozen flyer, seeds 0-2.
- `phase3_compare.py`: every arm on the test flights in both worlds; the upgrade is kept only if its paired gain over
  the control is above zero (bootstrap 95% interval over flights, seeds averaged per flight); plus Tasim & Sun's split
  (the same policy with its wind input zeroed, `load_policy(wind_in=False)`).

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
