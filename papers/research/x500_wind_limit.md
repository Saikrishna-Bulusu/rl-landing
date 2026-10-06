# How much wind can the x500 take?

> **Correction, 6 Oct 2026 (conditions/wind/PLAN.md, Phase 2a).** The numbers below came from an MJX twin that pushed
> the body drag at the centre of mass. Gazebo pushes it at the body plate, 1.6 cm higher, and the centre of mass sits
> 3.4 mm ahead of the rotor centre. With both copied, MJX matches PX4 in Gazebo motor by motor, and the steady limit
> depends on where the wind comes from: 26 m/s on the nose, 25 on a front arm or from behind, 22 from a side, 21-23
> on a back arm. Below 26 m/s it is a motor running out (one saturates, PX4 trims thrust), not the 45 deg tilt cap.
> Checked in Gazebo at four directions. The "about 26 m/s" headline holds only for wind on the nose.

## The answer in one sentence

In our simulator, the x500 (2.58 kg with the camera rig) can hold its position against a steady wind of up to about 26 m/s at the drone (about 95 km/h), but only because we ASSUMED a body-drag number; nobody measured it, no real x500 was tested, and neither Holybro nor PX4 publishes a wind rating. Once gusts are included, the practical limit at 30 m height is a mean wind at the drone of roughly 20 m/s, which is W10 = 15 m/s. At W10 = 18 and 20 m/s the drone is past its limit at 30 m but fine near the ground.

(W10 = the average wind speed measured 10 m above ground. "Wind at the drone" = the wind at the height the drone is flying.)

## 1. The idea, with an everyday example

Think of a person walking into a strong wind. They lean forward so the wind pushes them back as hard as they push forward. A drone does the same: it tilts into the wind so that part of its rotor thrust pushes sideways against the wind.

A drone has two limits on how far it can lean:

1. A software limit. PX4 (the flight software) never lets the drone lean more than 45 degrees. This is the setting `MPC_TILTMAX_AIR`.
2. A muscle limit. The more it leans, the more total thrust it needs, because the rotors must still hold its weight up. When the motors are at 100%, it cannot lean further.

Whichever limit arrives first sets the maximum wind. Past it, the drone cannot push back hard enough and slides downwind.

## 2. The table: steady wind at the drone vs what the drone does

How these numbers were made (so they can be checked):
- Tilt and thrust: a force-balance calculation (`steady.py`) with the x500's constants: weight 25.35 N, maximum total thrust 34.19 N (so thrust-to-weight is 1.35), rotor drag 8.06e-5, and body drag CdA 0.04 m2 (the assumed value; "CdA" is the size of the drone's wind-catching area times its drag factor). I re-ran it and the output is identical to the earlier one.
- "Holds position": the full MJX drone with PX4's control loops (`hold_steady.py`). Start 30 m up in still air, switch the wind on at once, hold the pad position for 30 s using the rule's approach law. Where the drone has a position to hold, it settles; the last column says how far it strayed on the way, and where it ends.
- "Thrust % of max" = total thrust divided by 34.19 N. Hover alone is 74%.

| steady wind U at the drone (m/s) | km/h | tilt (deg) | thrust, % of max | holds? | largest stray from the spot, then final drift |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 74 | yes | 0 |
| 5 | 18 | 4.5 | 74 | yes | 0.3 m, then 0 |
| 10 | 36 | 11.6 | 74 | yes | 0.8 m, then 0 |
| 12 | 43 | 15.1 | 74 | yes | 1.1 m, then 0 |
| 15 | 54 | 20.9 | 75 | yes | 1.4 m, then 0 |
| 18 | 65 | 27.2 | 77 | yes | 1.8 m, then 0 |
| 20 | 72 | 31.4 | 78 | yes | 2.1 m, then 0 |
| 22 | 79 | 35.7 | 80 | yes | 2.4 m, then 0 |
| 24 | 86 | 39.9 | 83 | yes | 2.7 m, then 0 |
| 26 | 94 | 43.9 | 87 | yes, just | 4.3 m, then 0.02 m |
| 26.6 | 96 | 45.0 (the cap) | 88 | edge | limit |
| 27 | 97 | would need 45.8 (just over the cap) | would need 89 | NO, blown away | drifts downwind, 18 m after 30 s |
| 30 | 108 | would need 51.2 | would need 97 | NO | 103 m after 30 s |
| 35 | 126 | would need 58.9 | would need 113 (impossible) | NO | 245 m after 30 s |

Reading the "holds within 1 m" question carefully, there are two honest answers:
- Does it come back to the spot and stay within 1 m? Yes, up to about 26 m/s.
- Does it never stray more than 1 m while the wind arrives? Only up to about 11 m/s if the wind switches on instantly from calm (14.5 m/s if it builds over 10 s). Wind does not switch on instantly in real life, but gusts do arrive fast. The stray happens because the drone has to lean first, and while it leans it is pushed downwind.

The simulation and the calculation agree: the calculation says tilt hits 45 deg at U = 26.6 m/s; the MJX drone settles at 26.0 and fails at 26.5.

## 3. The limiting factor

- With CdA 0.04, the tilt limit (PX4's 45 deg cap) binds first, at U = 26.6 m/s. Thrust, averaged over the four motors, would not run out until 31.2 m/s, but the drone never gets there, because PX4 refuses to lean past 45 deg.
- The motors are closer to the edge than that average suggests. Holding a lean needs the windward motors to push harder than the others. In the simulation the top motor reaches 997 to 1000 rad/s (its maximum is 1000) already at 25 to 26 m/s.
- The same holds for other drag assumptions: tilt is first, thrust a close second. At CdA 0.12 the tilt limit is at 16.6 m/s and mean thrust runs out at 17.8 m/s, and in that case the collective is pinned at 100% about half the time.
- A point worth knowing: even in still air, the x500 could not hold height at more than 42.1 deg of lean (the cosine of that angle times 34.19 N equals its weight). The rotor drag helps a little (it has an upward part when the drone leans), which is why the real limit is a bit higher.

### PX4's emulated settings (what the simulator copies)

| setting | value | meaning |
|---|---|---|
| tilt limit in flight (`MPC_TILTMAX_AIR`) | 45 deg | the cap that decides the limit |
| hover throttle (`MPC_THR_HOVER`) | 0.60 in the Gazebo airframe file; the MJX twin uses 0.836 | the throttle that holds weight (0.836 is what the thrust estimate converges to for 2.58 kg) |
| throttle limits | min 0.12, max 1.0 | the most thrust PX4 will ask for |
| horizontal margin (`MPC_THR_XY_MARG`) | 0.3 | PX4 keeps vertical thrust first and gives the rest to sideways thrust |
| max horizontal speed | 12 m/s, vertical up 3 m/s, down 1.5 m/s | speed limits of PX4's velocity controller |
| motor model | thrust 8.55e-6 w2 with w = 150 + 850 u rad/s | Gazebo's motor model |
| rotor drag | 8.06e-5 | Gazebo's propeller drag against the air |
| body drag | CdA 0.04 horizontal, 0.05 vertical | ASSUMED, added by us |

Where these come from: the 45 deg value is in the local PX4 source at `/home/sai/thesis_2.0/PX4-Autopilot/src/modules/mc_pos_control/multicopter_position_control_limits_params.c` line 96 (`PARAM_DEFINE_FLOAT(MPC_TILTMAX_AIR, 45.f)`), and the PX4 main-branch parameter file reports the same default. The hover throttle and motor limits are in the local airframe file `ROMFS/px4fmu_common/init.d-posix/airframes/4001_gz_x500`. The rotor-drag, thrust and mass numbers are from `conditions/wind/x500_mjx.py`. I did not reconstruct them from scratch.

### How sensitive the answer is to the body-drag guess

| CdA (m2) | what it means | steady wind it can hold | tilt reaches 45 deg at |
|---|---|---|---|
| 0 | Gazebo's x500 with no body drag (wind acts only through the propellers) | at least 40 m/s (tested to 40: settled lean 23 deg, 68% thrust; worst moment on the way 29 deg) | 109 m/s |
| 0.04 | our assumption | 26 m/s | 26.6 m/s |
| 0.08 | twice as draggy | 19.5 m/s | 19.9 m/s |
| 0.12 | three times as draggy | 16 m/s | 16.6 m/s |

So the limit is not a property of "the x500". It is a property of one number we guessed. Halving or doubling that guess moves the limit anywhere between about 20 and "no limit".

## 4. W10 to the wind the drone feels

The wind in the study rises with height (a logarithmic "log profile", roughness 0.3 m, a suburb): at 30 m it is 1.31 times W10, at 6 m it is 0.85 times W10. On top of that come turbulence and gusts (random bursts lasting 1.5 to 5 s, about one per 15 s, 20% to 60% of W10). Table made by `wind_stats.py`, 512 runs of 90 s per case. "Mean wind" is the profile value (the simulator's measured mean is about 3 to 4% higher because the average of a wind speed with turbulence is a little above its steady value). "Peak gust" is the strongest horizontal wind during 90 s of flying: the middle run, then the worst 1 run in 20.

| W10 (m/s) | mean wind at 30 m | peak gust at 30 m (median, worst 1 in 20) | mean wind at 6 m | peak gust at 6 m (median, worst 1 in 20) |
|---|---|---|---|---|
| 12 | 15.8 | 23.4, 26.4 | 10.2 | 18.7, 21.5 |
| 15 | 19.7 | 29.3, 33.1 | 12.8 | 23.5, 27.0 |
| 18 | 23.6 | 35.3, 39.8 | 15.4 | 28.3, 32.6 |
| 20 | 26.3 | 39.3, 44.2 | 17.1 | 31.6, 36.3 |

The steady limit (26.6 m/s, CdA 0.04) is crossed by the mean wind at 30 m near W10 = 20.3, but by the peak gusts (median) already near W10 = 13.6.

## 5. Does the drone survive a gusty wind (the harder, more realistic test)

`hold_turb.py` (its output file was missing, so I ran it here): 128 drones per case, random wind directions, each holding the pad position at a fixed height for 90 s in the full study wind (turbulence plus gusts). "Lost" = ended up more than 30 m from the spot, or fell 10 m, or tilted past 68 deg. "At the cap" = share of the time the lean sat at the 45 deg limit.

Body drag 0.04 (our assumption):

| W10 | height | lost | median worst stray | worst 1 in 20 | time at the 45 deg cap |
|---|---|---|---|---|---|
| 12 | 30 m | 0% | 1.6 m | 2.2 m | 0% |
| 15 | 30 m | 0% | 2.5 m | 8.7 m | 5% |
| 18 | 30 m | 37.5% | 22 m | 85 m | 41% |
| 20 | 30 m | 95.3% | 98 m | 268 m | 75% |
| 12 | 6 m | 0% | 1.0 m | 1.5 m | 0% |
| 15 | 6 m | 0% | 1.4 m | 2.1 m | 0.1% |
| 18 | 6 m | 0% | 2.1 m | 4.9 m | 1.9% |
| 20 | 6 m | 0% | 4.3 m | 15 m | 6.3% |

Body drag 0.08 (twice our assumption):

| W10 | height | lost | time at the 45 deg cap |
|---|---|---|---|
| 12 | 30 m | 0% | 11% |
| 15 | 30 m | 92% | 70% |
| 18 | 30 m | 100% | 75% |
| 20 | 30 m | 100% | 72% |
| 12 | 6 m | 0% | 0.3% |
| 15 | 6 m | 0% | 6% |
| 18 | 6 m | 27% | 26% |
| 20 | 6 m | 77% | 46% |

These are 90 s holds, which is harsher than a landing: the real descent from 30 m takes less time at the top, where the wind is strongest.

## 6. Published numbers: what exists and what does not

I searched and opened the following. None gives a wind rating for the X500 V2.

- PX4 documentation page for the Holybro X500 V2 (Pixhawk 5X build): https://docs.px4.io/v1.14/en/frames_multicopter/holybro_x500V2_pixhawk5x.html . It lists motors 2216 KV880, propellers 1045, a 4S 5000 mAh battery, and 144 x 144 mm plates. No wind speed, wind resistance or weather rating anywhere.
- Elektor's Holybro X500 V2 ARF kit product page: https://www.elektor.com/products/holybro-x500-v2-arf-drone-kit . It lists wheelbase 500 mm, weight 610 g, flight time about 18 minutes in hover with no payload, payload 1500 g (without battery, at 70% throttle), motors 2216 KV920, propellers 1045. It does not list maximum takeoff weight, thrust per motor or any wind or weather rating.
- PX4 parameter file on GitHub (main branch): https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/src/modules/mc_pos_control/multicopter_position_control_limits_params.yaml . It gives `MPC_TILTMAX_AIR` default 45 deg ("Absolute maximum for all velocity or acceleration controlled modes") and `MPC_TILTMAX_LND` default 12 deg.
- NASA Ames wind-tunnel paper: Carl Russell, Jaewoo Jung, Gina Willink and Brett Glasner (2016), "Wind Tunnel and Hover Performance Test Results for Multicopter UAS Vehicles", presented at the AHS 72nd Annual Forum, West Palm Beach, May 16-19, 2016: https://ntrs.nasa.gov/citations/20160007399 . Five multicopters (1.3 to 5.8 kg) were tested in the 7 x 10 ft wind tunnel. It does NOT state a drag-area number in text. It plots the bare airframe's "Drag/q" (drag divided by dynamic pressure, which is the same thing as CdA) in square feet in Figures 29 and 30. Reading those plots by eye (approximate): the three small quadcopters (SOLO 1.5 kg, Phantom 3 and Iris+ 1.3 kg) sit near 0.2 to 0.25 ft2 (0.02 m2) at zero pitch and rise to about 0.3 to 0.45 ft2 (0.03 to 0.04 m2) at 40 deg of pitch; the Endurance (2.7 kg, 510 mm across, the closest in size to the x500) is 0.31 ft2 (0.029 m2) at zero pitch, about 0.4 ft2 (0.037 m2) at 20 deg and 0.65 to 0.73 ft2 (0.06 to 0.07 m2) at 40 deg. So our 0.04 m2 is the right order of size and inside what NASA's bare airframes show when tilted, but it is NOT a number the paper states, the x500 carries a camera rig and gimbal that add drag, and these are bare airframes without the rotors. Safe wording: "of the same order as the bare-airframe drag areas measured in NASA wind-tunnel tests of 1.3 to 2.7 kg quadcopters (0.02 to 0.07 m2, read from plots)".

So, plainly: no published wind rating for the X500 V2 exists in anything I opened. The 45 deg tilt cap is a software setting, and the real vehicle's limit would come from its true drag and its true thrust.

## 7. Caveats

1. Body drag (CdA 0.04 m2 horizontal, 0.05 vertical) is an assumption. Gazebo's x500 has no body drag at all; we added it in the plugin `conditions/wind/gz_winddrag` (and the same law in MJX). The limit moves from about 20 m/s (CdA 0.08) to none (CdA 0). This is the biggest uncertainty.
2. The motor model is Gazebo's (thrust = 8.55e-6 w2, w = 150 + 850 u rad/s; first-order lag 12.5 ms up, 25 ms down). Real 2216 KV920 motors, real 1045 propellers and a real battery sag are not in it. The motor model has no loss of thrust with forward speed (the code has a switch, `SPEED_LOSS`, set to 0).
3. No real x500 was flown in wind. The only checks of the model against a reference are against Gazebo (repo README): steady tilt in wind agrees within 0.15 deg (5.76 vs 5.73, 10.36 vs 10.43, 22.48 vs 22.6 deg), and tilt during velocity steps within about 1.5 deg (44.1 vs 45.6 deg for a 0 to 8 m/s step). Nothing was checked against a real drone or a wind tunnel. Gazebo's check cannot test the body drag, because Gazebo uses our added drag plugin.
4. The weight is 2.584 kg (the x500 plus the camera rig). A lighter x500 without the rig has more spare thrust; one with a gimbal and payload has less.
5. Wind is modelled as a horizontal flow with a log profile, MIL-F-8785C turbulence and 1-cosine gusts; real urban wind around houses (turbulence from buildings, sudden vertical gusts) is rougher than that.
6. The `vel0` rows in `hold_steady.json` (velocity command zero, no position hold) are not about the wind limit: that mode never tries to return to the spot, so it reads as "drifts 3 m". I ignore them here.
7. Near the limit the MJX drone climbs about 1 m because the vertical controller adds thrust while saturated. Harmless for this question.
8. What I re-ran and what I did not. Re-ran: `steady.py` (identical output) and `hold_turb.py` (its output was missing; now produced). Read and spot-checked against each other: `hold_steady.json`, `hold_ramp.json` and `wind_stats.json` (consistent with the calculation to within about 0.5 m/s, for example 26.6 versus 26.0 to 26.5). Did not re-run `hold_steady.py`, `hold_ramp.py` or `wind_stats.py`. The scripts `t.py`, `t2.py` and `xcs.py` are only workarounds for a GPU compile error (unrolling the 10 physics steps) and do not change any physics.

## 8. What this means for the paper's ladder up to W10 = 20 m/s

- Up to W10 = 12 m/s: no concern with our assumed drag. The median run's largest lean at 30 m is 38.7 deg (below the 45 deg cap; 0% lost). With twice the drag it already sits at the cap 11% of the time.
- W10 = 15: the drone is near its edge at 30 m in the gusts (peak gust 29 m/s, above the steady 26.6 limit, 5% of the time at the cap), but it is not lost in a 90 s hold. This is where the rule starts to fail (78%), so it is a real test of the decider rather than of the drone, as long as the drag assumption holds.
- W10 = 18: at 30 m, the drone itself cannot always hold (37.5% lost in a 90 s hold; mean wind 23.6 and peaks 35 m/s). Below, near 6 m, it holds comfortably (0% lost, 1.9% of the time at the cap). The paper's story, that the rule fails at 18 because it waits for stillness, is correct for the last metres: the drone can hold there; but a policy that lingers high up in the strongest wind is also at the physical limit.
- W10 = 20: at 30 m the drone cannot hold (95% lost in a 90 s hold). Near 6 m it holds (0% lost, drift up to 15 m in the worst 1 in 20, 6% of the time at the cap). A landing is only possible if the flight spends little time up high.
- The result depends on one guess. If the true body drag were twice the assumed one, the x500 would already be at its limit at W10 = 12 to 15 at 30 m (92% lost at W10 15) and 18 at 6 m (27% lost); the ladder would end earlier. I would add this sensitivity as a short paragraph or a small table, and I would say "assumed CdA 0.04 m2; limit about 26 m/s steady" rather than "the x500 handles 20 m/s".
- Suggested wording for the paper (plain, hedged): "With an assumed body-drag area of 0.04 m2 and PX4's default 45 degree tilt limit, the simulated x500 can hold position in a steady wind of up to about 26 m/s at the drone. Our top rungs (W10 = 18 and 20 m/s, i.e. 24 to 26 m/s mean wind at 30 m, gusts to 35 to 45 m/s) therefore lie at or beyond the vehicle's limit at the start of the descent and within it near the ground. No wind rating for the real X500 V2 is published, and we did not test it."
- A safe next step if a number is wanted: fly the real x500 in a measured steady wind and read its hover tilt; that gives the true drag and replaces the assumed 0.04. Gazebo cannot supply it, because its steady-tilt figures in the README use the same assumed drag.

## Files

- Scripts and outputs: `papers/research/scripts/` (`steady.py`, `steady_out.txt`, `hold_steady.json`, `hold_ramp.json`, `hold_turb.py`, `hold_turb_out.txt`, `hold_turb.json` if finished, `wind_stats.json`).
- Repo, read only: `conditions/wind/x500_mjx.py`, `land_env.py`, `README.md`.


## Verification

Checked by a second agent, 4 Oct 2026. Scripts I wrote or re-ran are in `papers/research/scripts/chk/` (`indep.py`, `rerun.py`, `rerun2.py`). Nothing in the repo was changed; no PX4 or Gazebo was started.

### Sources (opened)

1. PX4 docs, Holybro X500 V2 (Pixhawk 5X), v1.14 page: VERIFIED. Motors "2216 KV880" (marked superseded on the page), propellers 1045, 4S 5000 mAh recommended (not included), bottom plate 144 x 144 mm, no wind or weather rating. Opened https://docs.px4.io/v1.14/en/frames_multicopter/holybro_x500V2_pixhawk5x.html .
2. Elektor Holybro X500 V2 ARF kit page: VERIFIED. 500 mm wheelbase, 610 g, about 18 min hover (5000 mAh, no payload), payload 1500 g (without battery, 70% throttle), motors 2216 KV920, 1045 propellers (6 listed), no wind rating, no thrust per motor. https://www.elektor.com/products/holybro-x500-v2-arf-drone-kit
3. PX4 main `multicopter_position_control_limits_params.yaml`: VERIFIED. MPC_TILTMAX_AIR default 45 deg (min 20, max 89), text "Absolute maximum for all velocity or acceleration controlled modes"; MPC_TILTMAX_LND default 12 deg. Opened the raw GitHub URL given in section 6. Local checkout `/home/sai/thesis_2.0/PX4-Autopilot` (commit 54f0455, 22 Apr 2026): line 96 is `PARAM_DEFINE_FLOAT(MPC_TILTMAX_AIR, 45.f)`, VERIFIED.
4. NASA paper, Russell, Jung, Willink, Glasner: title, four authors in that order, 2016, 7 x 10 ft tunnel, five vehicles: VERIFIED (https://ntrs.nasa.gov/citations/20160007399, and the full PDF https://ntrs.nasa.gov/api/citations/20160007399/downloads/20160007399.pdf, read with pdftotext and Figures 29 and 30 viewed). CORRECTED: the report named no venue (it is a conference paper, AHS 72nd Annual Forum, May 2016), and said the PDF could not be opened. The "0.04 m2 range" claim is only partly supported: the paper gives no 0.04, only plotted Drag/q of 0.2 to 0.7 ft2 (0.02 to 0.07 m2). The section 6 bullet is rewritten with those numbers, marked approximate (read off a plot).
5. "One fetch of the PX4 docs gave 60 deg for MPC_TILTMAX_AIR": REMOVED. I could not reproduce or trace it (the docs reference page came back truncated, the v1.14 raw source path returned 404), and the report itself discarded it.
6. Local airframe file `ROMFS/px4fmu_common/init.d-posix/airframes/4001_gz_x500`: VERIFIED `MPC_THR_HOVER 0.60`, `SIM_GZ_EC_MIN 150`, `SIM_GZ_EC_MAX 1000`. Defaults in the local source: MPC_THR_MIN 0.12, MPC_THR_MAX 1.0, MPC_THR_XY_MARG 0.3, MPC_XY_VEL_MAX 12, MPC_Z_VEL_MAX_UP 3, MPC_Z_VEL_MAX_DN 1.5: all VERIFIED. Gazebo `x500/model.sdf`: motorConstant 8.54858e-06, rotorDragCoefficient 8.06428e-05, time constants 0.0125 / 0.025 s, max 1000 rad/s: VERIFIED.
7. Gazebo's x500 has no body drag, and the drag is our plugin: VERIFIED in `conditions/wind/gz_winddrag/WindDrag.cc` (body drag 0.5 rho CdA |v| v, cda_h 0.04, cda_v 0.05) and README table.
8. "Thrust drops with forward speed in real rotors": REMOVED (no source). The repo has `SPEED_LOSS = 0` (x500_mjx.py line 52). I also looked in gz-sim 8's `MulticopterMotorModel.cc` (GitHub) and found no forward-speed thrust loss there, only the rotor drag term, which agrees with the switch being off. The repo comment that credits gz-sim for this term looks mistaken (I think it refers to the older Gazebo Classic plugin; not confirmed).

### Numbers about our code (re-derived from the repo)

9. Mass 2.5843 kg, KT 8.54858e-6, w = 150 + 850 u, lag 12.5 / 25 ms, rotor drag 8.06428e-5, CdA 0.04 / 0.05, tilt 45 deg, thrust limits 0.12 to 1.0, margin 0.3, speeds 12 / 3 / 1.5, hover throttle 0.836 (W_HOVER 861 rad/s): VERIFIED in `conditions/wind/x500_mjx.py`. z0 0.3 m, MIL-F-8785C, 1-cosine gusts: VERIFIED in `conditions/wind/land_env.py` and README.
10. Gust length "one to five seconds": CORRECTED to 1.5 to 5 s, one per 15 s on average, 0.2 to 0.6 W10 (land_env.py lines 36, 73 to 75; README section 2).
11. Log profile ratios 1.31 at 30 m and 0.85 at 6 m: VERIFIED by hand (ln(100)/ln(33.3) = 1.313; ln(20)/ln(33.3) = 0.854). Mean-wind columns of the section 4 table (15.8, 19.7, 23.6, 26.3 and 10.2, 12.8, 15.4, 17.1) match; the simulator mean is 3.5% above the profile: VERIFIED from `wind_stats.json`.
12. Max thrust 34.19 N, weight 25.35 N, T/W 1.349, hover 74%, still-air lean limit 42.1 deg: VERIFIED (own calculation in `chk/indep.py`: 4 x 8.54858e-6 x 1000^2 = 34.194; acos(25.352/34.194) = 42.15 deg).
13. Steady-wind table (tilt, thrust %): VERIFIED against my independent solver (fresh code, `chk/indep.py`) for U = 5, 18, 26, 26.6, 28, 30, 35, and the 45 deg crossing at 26.57 m/s (report: 26.6). One row CORRECTED: at U = 27 the report had copied the 28 m/s values (47.7 deg, 91%); the right values are 45.8 deg and 89%.
14. MJX hold results (the "largest stray" column, 26 settles, 26.5 fails, 27 drifts 18 m, 30 drifts 103 m, 35 drifts 245 m): VERIFIED from `hold_steady.json`, and re-run on the GPU for U = 10 to 12 and 26 to 27 (e.g. 26.0: 4.19 m stray, 0.02 m final; 26.5: 4.27 m final; 27: 18.3 m; the report's 4.29 differs by 0.1 m from GPU run-to-run noise). Note, not an error: with a 10 s wind ramp the drone still settles at 26.5 (`hold_ramp.json`), so "settles up to 26.0 to 26.5" is the honest range. The 11 and 14.5 m/s "never strays over 1 m" figures: VERIFIED (`summarize_hold.py`).
15. CdA = 0 line: CORRECTED. The report said "29 deg lean" at 40 m/s; 29.1 deg is the worst moment, the settled lean is 22.7 deg (calculation: 22.8 deg, 68% thrust). Table text now says both. Tilt reaching 45 deg at 108.6 m/s: VERIFIED. CdA 0.08 and 0.12 (holds 19.5 and 16 m/s; 45 deg at 19.9 and 16.6; mean thrust runs out at 21.9 and 17.8; collective pinned about half the time at 0.12): VERIFIED.
16. Peak-gust table (section 4): VERIFIED against `wind_stats.json` for all eight cells, and one cell (W10 15, 30 m) re-run: identical (mean 20.39, median peak 29.34, 95th 33.09). Crossing estimates (W10 20.3 for the mean at 30 m; 13.6 for the median peak): VERIFIED by interpolation.
17. Gusty-hold tables (section 5, both drag values): VERIFIED against `hold_turb_out.txt` for all 16 cells of each; two cells re-run on the GPU and reproduced exactly (W10 15 at 30 m: lost 0%, 95th stray 8.66 m, 5.0% at cap; W10 18 at 30 m: lost 37.5%, 22.2 m, 40.6% at cap).
18. README agreement with Gazebo: CORRECTED. Report said "within about 1 deg". Steady-wind tilt agrees within 0.15 deg (three cases); velocity-step tilt differs up to 1.5 deg (44.1 vs 45.6 deg). Caveat 3 now says so, and says the body-drag check is not independent.
19. Section 8: "leans less than 40 deg even at gust peaks at 30 m" at W10 12: CORRECTED to the median run's largest lean (38.7 deg), because only the median is stored, and with twice the drag the drone is at the cap 11% of the time. All other section 8 numbers: VERIFIED against items 14 to 17.

### Counts

Of the 19 numbered items: VERIFIED 11 (items 1, 2, 3, 6, 7, 9, 11, 12, 14, 16, 17), CORRECTED 6 (4, 10, 13, 15, 18, 19), REMOVED 2 (5, 8). The report's headline answer (about 26 m/s steady with an assumed CdA 0.04; about W10 15 practical at 30 m) stands.
