# How our results compare with published work (checked 30 Sept 2026)

Every number below was read from the paper itself this session (arXiv full text), except where marked.
Different papers use different tasks and metrics, so rows are comparable in kind, not head to head.

## Wind

| Work | Task | Sim / hardware | Wind | Baseline | Result | Trials |
|---|---|---|---|---|---|---|
| **Ours, flying learner** | landing on a back-yard pad, strict touchdown | MJX with PX4's loops emulated; PX4/Gazebo SITL check | W10 up to 20 m/s, MIL-F-8785C turbulence + 1-cos gusts | the rule on PX4's default PID (with integral) | landed 77.9 -> 99.1% at 15 m/s, 5.6 -> 50.7% at 18; crashes <= 0.5%. Gazebo 15 m/s: 12/19 -> 18/20, 19/20 | 1,024 paired flights x speed x 8 seeds; Gazebo 20 paired yards |
| **Ours, crash-priced decider** | same | same | same | same rule | 87.3% landed at 18 m/s, 3.3% crashed; Gazebo 15/20, 3 crashes | 3 seeds; Gazebo 20 |
| Neural-Fly, O'Connell 2022 (arXiv 2205.06908) | trajectory tracking | hardware, wind tunnel, 2.6 kg PX4 drone | up to 12.1 m/s | nonlinear, INDI, L1 | RMS error at 12.1 m/s: NF 9.4 cm, INDI 10.7, L1 22.7, nonlinear 23.9 | 6 laps per condition |
| Wind-aware RL, Al Tasim & Sun 2026 (2607.01528) | trajectory tracking | simulation only, 2.59 kg | 4-12 m/s (13-15 out of range) | wind-blind PD, no integral action | RMSE -48%, wins 225/225 paired episodes; PD fails at 13-15 | 225 paired |
| GustPilot, Habel 2026 (2603.19966) | gate navigation | hardware, 50 g Crazyflie, fans | up to 3.5 m/s | DRL-PID | success 94.7% vs 55.0% | 80 flights |
| TornadoDrone, Peter 2024 (2406.16164) | landing on moving pad | hardware Crazyflie + fan | fan 4,500/8,500 rpm (m/s not given) | EKF+PID (no-wind cases only) | static pad with wind 100%, moving 91.7/78.6%; 3.96 cm mean miss | 120 test cases |
| WaveLander, Li 2026 (2607.01281) | landing timing on a tilting deck (vertical velocity from RL) | MuJoCo; SITL qualitative; one real test | real test ~10 kt, gusts 18 kt | constant descent | strict touchdown 66/60/39% vs 33/14/8% | 100 paired seeds per case |
| HARL-AC, Nguyen 2026 (2609.12758) | mid-air capture by ship arm | Isaac Lab simulation | sea states 0-10 | domain randomization | up to 97.5% in distribution; 60 vs 45.9% at sea state 10 | 10 runs x 10,000 |

**Reading.** In simulation we test landing at stronger mean winds than any of these (their maximum: 12.1 m/s on hardware,
15 m/s in simulation) and with larger paired samples, and we check in PX4/Gazebo. Our baseline (PX4's PID cascade with
integral) is at least as strong as the PD / PID / constant-descent baselines most RL-wind papers use. We are weaker on two
points: no hardware (Neural-Fly, GustPilot, TornadoDrone fly real drones) and no modern disturbance-rejection baseline
(INDI or L1, which Neural-Fly shows are strong at 12 m/s). The wind session found a fixed push does nothing (rule stays 77/5%).

## Magpie and other fast movers

| Work | Threat | Sim / hardware | Result | Trials |
|---|---|---|---|---|
| **Ours** | magpie attacking from behind, steering at up to 20 m/s^2, commits 0.3 s before contact; 2.6 kg delivery drone | fast simulator; live detector in Gazebo | learner +4.1 points of clean deliveries over the best same-sensor planner; struck on 38.9% of magpie flights (rule, rig) vs 27.7% with perfect knowledge | 4,096 paired flights x 3 seeds |
| Threat-aware dodging, Zhang 2025 (2511.22847) | human-thrown ball, 3-13 m/s, from 3-6 m; 1.35 kg drone | hardware + replay simulation | dodge success 76-100% on hardware (82.4% in the fastest band); 96.7% vs 60% baseline in simulation | 21 per condition; 30 in sim |
| Pursuit-evasion, Sanchez Roncero 2025 (2506.02849) | learned drone pursuer | Isaac Lab + 42 Crazyflie flights | learned evader escaped the manual and FRPN pursuers in every tested hardware trial | 1,024 x 3 seeds per matchup |
| Falanga 2020, Science Robotics (papers/acra2026_refs/falanga2020.pdf) | thrown objects, relative speeds up to 10 m/s, from about 3 m; event cameras, 3.5 ms latency | hardware | outdoors, detected and avoided the thrown objects more than 90% of the time | trial count not stated |
| NavRL, Xu 2025 (2409.15634) | walking people and moving obstacles | Isaac Sim, Gazebo, hardware | collisions per run: dynamic 0.85 vs ViGO 3.15; policy 7 ms on Orin NX | 20 runs per environment |

**Reading.** No published study measures magpie (or any bird) attacks on delivery drones; bird work is detection and
datasets, plus the Lyons 2018 field report. A thrown ball does not steer and a small drone out-accelerates it, so the
76-100% dodge rates are not comparable to our per-flight strike rates. What is new here is the problem and the finding that
sensing sets the strike rate more than deciding does.

## Obstacles and landing search

| Work | Result |
|---|---|
| **Ours** | local 3D planner passed all 10 crane positions (never closer than 2.59 m); survey-only planner crashed 8/10; learning adds nothing specific (privileged planner +0.1 over the best planner) |
| NavRL 2025 | static environment collisions per run: NavRL 0.65 vs EGO-Planner 0.45 (a planner beats the RL there) |
| Marker search in urban AirSim, Yao 2026 (2601.11078, Macquarie) | success: spiral search 60.6%, zigzag 58.7%, end-to-end RL 24.4% over 966 episodes with fog and dust |

**Reading.** Both show the pattern our ladder is built to catch: on static structure a planner or a fixed pattern beats RL.

## Delivery with aircraft in the airspace

Benali & Hamoud 2026 (Vehicles 8(4):85; full text in papers/vehicles-08-00085.pdf and acra2026_refs/benali2026.pdf):
PPO in an AirSim/Unreal model of Bristol. The helicopter is non-reactive on one predetermined straight-line path, and the
drone's initial relative position and heading are fixed in every episode (head-on and oblique intercepts). Reported avoidance
success, read from training curves (no separate test set, no rule or planner baseline): 92% at 15 m/s dry, 85% at 25 m/s dry,
58% at 15 m/s in rain, 52% at 25 m/s in rain; 62 to 89% with 12 to 20 kg payloads. Ours: the ambulance arrives on half the
flights at a random time from the north-west (any direction in the earlier GeoAnchor-nav runs), 4,096 paired test flights,
compared with a rule, two planners and a privileged planner. Different metric (a whole clean delivery, not one encounter),
so no head-to-head number; the difference is the method.
