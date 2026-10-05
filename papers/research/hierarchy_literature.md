# Has someone already built our hierarchical architecture? Literature check

Written for Sai first (plain words, one everyday picture per idea), then the details for the paper.
Date of search: 4 Oct 2026. Sources: arXiv/alphaXiv, web search, and the PDFs already in /home/sai/RL_projects/papers/. I only cite what I opened. Where I could not open a source I say so at the end ("What I could not verify").

---

## (a) The plain answer

No single paper builds our whole stack. But every layer of it, and most pairs of layers, have been built by someone. The nearest whole-system work I found is the NTNU "Unified Autonomy Stack" (Dharmadhikari et al., 2026): a planner, then a swappable local navigator (either a model-based controller or a PPO-trained network), then a last-resort safety filter, then the autopilot (PX4 or ArduPilot), flown in GNSS-denied field tests. The nearest "RL picks a move and a normal controller flies it" papers are VO-Safe RL (Lin et al., ICRA 2024, the one your professor pointed at) and NavRL (Xu et al., RA-L 2025, which adds a velocity-obstacle shield). The nearest wind work is Ishihara et al. 2023 (a learned correction on top of PX4's PID, trained in Gazebo) and WaveLander 2026 (RL chooses only the vertical speed during landing). What I did not find anywhere is the combination we have: a learned decider that chooses among 14 named candidate velocity commands while seeing a planner's 20-second forecast for each one, sitting under hard aviation rules (CASA-style), above PX4, with satellite-image localization feeding EKF2 and a marker landing at the end. So the honest claim for the paper is "a new combination and a new decision interface", not "a new kind of hierarchy". Hierarchy itself (a learned high level over a classical low level) is old, and the professor is right to expect you to cite it and say how you differ.

Caveat: "I did not find it" is not "it does not exist". My search was broad (eight angles) but not exhaustive, and two journal pages (MDPI) refused access.

---

## Words you need (each with a picture)

| Term | Plain meaning | Everyday picture |
|---|---|---|
| Hierarchical control | A "thinker" picks what to do, a "doer" carries it out. | A driver decides "turn left at the lights". The car's steering and engine do the mechanics. |
| High-level / low-level | Thinker / doer. In ours: decider (0.5 s) over PX4 (fast loops). | Same as above. |
| Shield / safety filter | A fixed rule layer that checks the learned policy's action and overrides it only if it is unsafe. | A driving instructor with a second brake pedal. |
| Control barrier function (CBF) | A maths version of a shield: a formula that says "stay on the safe side of this line", solved as a tiny optimisation each step. | A guard rail whose stiffness is computed from how fast you are going. |
| Runtime assurance / Simplex | Run a clever but unproven controller. Switch to a simple, proven backup when a monitor says danger. | Autopilot with a "return to safe mode" button. |
| MPC (model predictive control) | Plan a few moves ahead using a physics model, execute the first, repeat. | A chess player who thinks three moves ahead every turn. |
| DAgger | Imitation where the learner drives, and the teacher labels what it should have done at the places the learner actually ends up. | An apprentice who drives the route while the master sits beside them and says "here you should have braked", instead of only watching the master drive. |
| Privileged expert | A teacher that sees things the learner cannot (true map, true positions). | A coach who watches from a drone above. |
| Residual RL | Keep the classical controller. Learn only a small correction added on top. | Cruise control plus a human who nudges the pedal. |
| Feed-forward | Act before the error appears, from a prediction. | Leaning into the wind before the gust shoves you. |
| Sim-to-real | A policy trained in simulation works on the real robot. | A pilot trained in a simulator, then flying the real plane. |

---

## (b) Comparison of the nearest works

Scores are my judgement. "Whole" = closeness to our whole stack. "Layer" = closeness to the one layer the paper covers. 5 = almost the same, 1 = same field only.
"Link" is the page I opened. All items are in the reference list in section (f).

| # | Work (authors, year, venue) and link | Hierarchy levels | What is learned | What is planned / hand-designed | Safety layer | Sim / real | Whole / Layer | Key difference from ours |
|---|---|---|---|---|---|---|---|---|
| 1 | **VO-Safe RL**. Lin, Wei, Grech, Ji. 2024. ICRA. doi 10.1109/ICRA57147.2024.10611487 | 2: PPO waypoint chooser, then cascaded PID | PPO picks 1 of 8 directions (1 m step) from a semantic image and the VO-estimated goal | Low-level PID (position, then attitude). Semantic classifier. ORB-SLAM2 for pose | None as a hard layer. Safety = a reward penalty for visual-odometry failure. Collision-free world assumed | Flightmare sim, plus one small indoor flight (RealSense, white-paper patch) | 2 / 3 | Same pattern (learned discrete mover over a classical flyer, GNSS-free via camera). But no obstacles, no rules, no planner forecast, no landing, no wind. Not a full stack |
| 2 | **Unified Autonomy Stack**. Dharmadhikari et al. (NTNU ARL). 2026. arXiv 2605.12735 | 4: perception (SLAM), planner (OmniPlanner), local navigation, safety filter, autopilot | Optional PPO "ExRL" policy (Aerial Gym) outputs acceleration + yaw-rate setpoints | Multi-sensor SLAM, graph planner, neural-SDF MPC (the alternative navigator), CBF filter | Composite CBF as a last-resort QP on setpoints; map-based planning as first layer | Many real field tests (mine, forest, ship hold; aerial and legged robots) | 4 / 4 | Closest overall structure (planner + learned navigator + hard filter + PX4). But the learned part is a local navigator, not a chooser among named candidates. No landing, no traffic rules, no birds, no satellite localization, no wind policy. GNSS-denied by LiDAR/radar SLAM |
| 3 | **NavRL**. Xu, Han, Shen, Jin, Shimada. 2025. IEEE RA-L 10(4). arXiv 2409.15634 | 2: PPO velocity policy, then flight controller | PPO outputs velocity commands from static + dynamic obstacle encodings | Perception (voxel map, dynamic boxes). Velocity-obstacle (VO) shield solved as linear programme | VO shield corrects the policy's velocity if a collision is forecast | Trained in Isaac Sim, thousands of drones in parallel; zero-shot real flights | 3 / 4 | Nearest on "thousands of drones in a GPU sim + learned velocity + hand-made shield". Policy is continuous and one-level. No landing, wind, rules, localization, or per-candidate forecast fed to the learner |
| 4 | **Safety-shielded RL for high-speed flight**. Zhang, Lei, Dai, Hoi, Wang, Han, Gao. 2026. IEEE RA-L (arXiv comment: "Published in IEEE Robotics and Automation Letters"). arXiv 2602.08653 | 2: PPO attitude/thrust policy at 100 Hz, then PX4 | PPO from depth image (CNN+GRU) to thrust + attitude | Dijkstra-field reward, CBF reward, HOCBF quadratic programme | HOCBF filter corrects policy output at run time | Sim, indoor and forest real flights to 7.5 m/s | 2 / 3 | Same idea of a hard filter beneath a learned policy. Their policy talks to PX4 at attitude level, ours at velocity level. Static clutter only; no dynamic intruders, rules, landing |
| 5 | **Policy Library CBF**. Kim, Okamoto, Hoxha, Fainekos, Panagou. 2026. arXiv 2605.16588 | Filter beside any nominal controller | Nothing learned in the filter | Library of fallback policies rolled forward in parallel; picks the least invasive safe one | The filter itself | Sim only (double integrator, car, 12-state quadrotor) | 2 / 4 | Closest to our shield-form planner ("fly the rule unless a violation is forecast, otherwise pick the mildest safe candidate"). We add a learned decider on top and a candidate set that includes land-anyway and climb-straight-up |
| 6 | **Learning high-speed flight in the wild**. Loquercio, Kaufmann, Ranftl, Müller, Koltun, Scaramuzza. 2021. Science Robotics 6(59). doi 10.1126/scirobotics.abg5810 | 2: CNN picks among 3 trajectories, then MPC + flight controller | Student CNN that imitates a privileged sampling planner (DAgger), outputs 3 candidate trajectories with collision costs | Privileged expert (Metropolis-Hastings sampler), MPC, BetaFlight | None formal. Lowest predicted cost is executed | Sim-only training, zero-shot real flights to 10 m/s | 2 / 4 | Nearest on the training recipe (privileged planner, DAgger) and on "learner chooses among candidates". Imitates only, no RL fine-tune, no rule layer, no landing, no moving intruders |
| 7 | **Learning high-level policies for MPC**. Song, Scaramuzza. 2020. IROS. arXiv 2007.10284, doi 10.1109/IROS45743.2020.9340823 | 2: network sets MPC decision variables | Network that chooses high-level variables for the MPC | MPC, dynamics model | MPC constraints only | Simulated quadrotor, swinging-gate task. No real-flight experiment (abstract and text checked in verification) | 2 / 3 | The classic "learned high level over a planner". Continuous decision variables for one task, not a discrete menu under rules |
| 8 | **Residual RL on cascaded PID**. Ishihara, Hazama, Suzuki, Yokono, Sabe, Kawamoto. 2023. arXiv 2308.01648 (preprint) | 2: PX4 PID + SAC residual (10 Hz) | SAC outputs a residual of thrust and roll/pitch/yaw torques | PX4 cascaded PID kept intact | None | Trained in Gazebo + PX4 with 15-25 m/s wind. Real hardware: blower test (41 % less mean deviation, 52 % less maximum) and fan at about 13 m/s (35 % less mean, 53 % less maximum) | 1 / 4 | Nearest wind-layer work. Residual is at torque level, position hold only, not landing, no vertical decider, no turbulence spectrum |
| 9 | **WaveLander**. Li, Sit, Siu, Kui, Lin, Wang, Shi. 2026. arXiv 2607.01281 | 2: RL vertical speed, then PID + SE(3) flight controller | GRU PPO outputs one number: vertical velocity in [-0.45, 0.45] m/s (descend, hold, retreat) | AprilTag pose, lateral PID, low-level controller | None formal (reward shaping on tilt) | MuJoCo, Isaac + ArduPilot SITL, one real test with wind | 2 / 4 | Nearest to our "learned decider for the vertical action while landing". Waves, not wind; one scalar, not our 5-option vertical menu (rule's own, down 1.5, down 0.6, hold, up 1 m/s) every 0.04 s plus a feed-forward head; no rules; one real test |
| 10 | **GustPilot**. Habel, Khan, Mehboob, Fortin, Tsetserukou. 2026. arXiv 2603.19966 | 2: DRL velocity reference, then INDI controller | PPO inertial velocity reference for gate traversal under wind | INDI inner loop (incremental, acceleration-feedback disturbance rejection) | None | 50 g Crazyflie real flights, 80 runs, wind up to 3.5 m/s; 94.7 % vs 55.0 % for DRL-PID | 2 / 3 | Same "learn the reference, let the inner loop fight gusts" idea. Tiny drone, gates, not landing. Rejection is in the inner loop, ours is learned |
| 11 | **Sequential DQN landing (SDQN)**. Polvara, Patacchiola, Hanheide, Neumann. 2020. Robotics 9(1):8. doi 10.3390/robotics9010008 | 2: DQN "align" then DQN "descend", triggered | Two DQNs on low-res camera images pick high-level moves | Marker, ROS controller, Gazebo | None | Gazebo + domain randomisation, real Parrot Bebop 2 | 2 / 3 | A marker-landing hierarchy with discrete learned commands. A 2024 survey (Amendola et al.) calls the earlier ICUAS 2018 version of this work (Polvara et al., "Toward end-to-end control for UAV autonomous landing via deep RL") the only hierarchical-RL landing work it found. No obstacles, rules, wind, GNSS-free |
| 12 | **SafeLand**. Gross, Greiner, Matha, Soest, Cremers, Meeß. 2026. arXiv 2603.17430 | Perception, Bayesian map, behavior-tree decider | Segmentation network only | Behavior tree: pause, climb, reroute; Bayesian landing-zone filter | Rules in the behavior tree | 200 sims + 60 field tests (95 % success), no map needed | 2 / 3 | The nearest real-world "delivery-style landing near people" stack, but the decider is hand-written, not learned. No wind, no intruders beyond people, no RL |

Other works that matter but did not make the top twelve (details in section (d) and (f)):
Kulkarni and Alexis, ICRA 2024 (PPO velocity + yaw-rate from a depth "collision encoder", NTNU); Aerial Gym, Kulkarni et al., RA-L 2025 (the GPU simulator NTNU uses, Isaac Gym based); Xing et al., CoRL 2024 (RL teacher, imitation student, then RL fine-tune); Romero et al., T-RO 2026 (differentiable MPC inside actor-critic); Kaufmann et al., Nature 2023 (Swift: RL policy outputs thrust and body rates); Ramezani and Voos 2026 (rule-based advisor on top of an online RL controller, i.e. the reverse hierarchy); Zongo and Wei 2026 (runtime filter for a learned separation policy under GNSS degradation); NGPS, Sharma, IROS 2026 (satellite-image matching fused into ArduPilot's EKF).

### Nearest work per layer (quick view)

| Our layer | Nearest published work | What still differs |
|---|---|---|
| Localization (satellite fixes into EKF2, marker end-game) | NGPS (Sharma, IROS 2026): SuperPoint + LightGlue + RANSAC against satellite images, fused with VIO, feeds ArduPilot EKF3, 2.94 m RMSE at 60-150 m. Polvara 2020 and Schroder et al. 2025 for marker landing | No RL involved there. NGPS is a high-altitude system (flights at 60-150 m AGL) and is not a landing stack |
| Hard safety layer (rules + planner + voxel map) | Zhang 2026 (HOCBF), NavRL (VO shield), UAstack (C-CBF), Policy Library CBF (rollout library), Alshiekh et al. 2018 (shielding idea) | Ours adds named aviation rules (marker identity, gate below 6 m, commit-on-lock, CASA-style) and a 20 s forecast |
| Decision layer (learned chooser of 1 of 14 candidates) | Loquercio 2021 (chooser among 3 candidate trajectories), VO-Safe (1 of 8 directions), Song and Scaramuzza 2020 (learned MPC knobs) | None gives the learner the planner's per-candidate forecast; none has a menu with hand-written manoeuvres like "land anyway" |
| Flying layer in wind | Ishihara 2023 (residual on PX4), GustPilot 2026 (velocity + INDI), WaveLander 2026 (vertical speed) | Ours: acceleration feed-forward head plus a 0.04 s vertical-action decider, MJX twin, turbulence + gusts, flight-by-flight SITL check |
| Training recipe (DAgger from planner, then PPO) | Loquercio 2021 (DAgger from privileged planner), Xing 2024 (IL then RL fine-tune), Ross 2011 (DAgger) | Copy of a shield planner, then PPO inside a GPU sim with intruders |

---

## (c) The single nearest architecture, and why

**Nearest whole system: the Unified Autonomy Stack (NTNU Autonomous Robots Lab, arXiv 2605.12735, 2026).**

Why, in plain words. Picture two buildings with the same floor plan. Ours: hard-rule layer at the bottom of authority, a learned decider in the middle, PX4 doing the actual flying. UAstack: a map-based planner for the first line of safety, a learned or model-based local navigator that produces acceleration setpoints, a composite-CBF last-resort filter that can modify those setpoints, then PX4 or ArduPilot. In both, the learned part is allowed to be clever but never allowed to be the last word on safety. Both aim at GNSS-denied work. UAstack has real field tests, we do not (yet).

What is different, so you can say it to the professor:

| UAstack | Ours |
|---|---|
| Learned part = local navigator producing continuous acceleration setpoints | Learned part = chooser of 1 of 14 named velocity commands every 0.5 s, with the planner's forecast as input |
| Safety = map planner + CBF quadratic programme | Safety = marker identity, 6 m landing gate, commit-on-lock, 3D A* planner (hard 2 m pad around obstacles, soft penalty out to 5 m), bird-dodge reflex, CASA-style rules. The 6 m gate and the A* planner exist in the Gazebo flights only, not in the fast simulator |
| Task = explore, inspect, reach a waypoint | Task = deliver and land on the right pad in a suburb |
| Localization = LiDAR, radar, vision SLAM | Localization = satellite-image fixes into EKF2, then markers and rangefinder |
| No wind policy | Wind policy (feed-forward + vertical decider) |
| Real robots | Simulation only so far |

**Nearest to what the professor pointed at: VO-Safe RL** (Lin et al., ICRA 2024). It is the plain two-level pattern, "RL chooses the next step, a classical controller flies it", in a camera-only GNSS-free setting. It is much smaller than our stack. It has no hard safety layer, no obstacles, no landing.

**Nearest on shield + velocity-RL + GPU-parallel training: NavRL** (Xu et al., RA-L 2025). I would cite it right next to UAstack when describing the "learned policy under a hand-made shield" pattern.

---

## (d) What is genuinely new in ours, and what is not

### Not new (cite, do not claim)

1. A learned high level over a classical low level: Lin 2024 (VO-Safe), Polvara 2020 (SDQN), NavRL 2025, GustPilot 2026, WaveLander 2026, Kulkarni and Alexis 2024. Explicitly stated for UAVs in the VO-Safe paper itself ("platform-agnostic nature allows for easy transfer").
2. A hand-made safety layer under a learned policy: shielding (Alshiekh et al., AAAI 2018), VO shield (NavRL), HOCBF (Zhang 2026), C-CBF (UAstack), rollout-library filter (Policy Library CBF, 2026).
3. Imitating a privileged planner with DAgger (Ross et al. 2011; Loquercio et al. 2021) and imitation followed by RL fine-tuning (Xing et al., CoRL 2024).
4. Residual RL on a classical controller (Johannink et al. 2019; Silver et al. 2018; Ishihara et al. 2023 on PX4).
5. Thousands of drones trained at once on a GPU (Aerial Gym, Isaac Sim in NavRL, our MJX).
6. Wind-aware learned flight (GustPilot, TornadoDrone, Neural-Fly, DATT, Ishihara).
7. Satellite-image matching fused into an autopilot EKF in GNSS-denied flight (NGPS 2026). RL avoidance of crewed aircraft in a Bristol city-centre simulation, with an SQL compliance monitor for low-airspace rules (Benali and Hamoud 2026). Note: this is a simulation study with a literature table, not a survey. I do not know which document your professor called a survey, so check that with him.
8. Rule-based layer working with an RL layer in search-and-rescue-motivated UAV delivery tasks (Ramezani and Voos 2026, but with the roles reversed: rules advise, RL acts).

### Plausibly new as a combination (my reading, "not found in my search")

| Claim | Closest prior | Why I think ours differs | Risk if challenged |
|---|---|---|---|
| The decider chooses among a named menu of 14 candidate velocity commands, including hand-written manoeuvres (land anyway, climb straight up, fly away from the helipad, turn +-45/+-90, reverse) | Loquercio 2021 (3 candidate trajectories), VO-Safe (8 directions), OmniRisk 2026 (risk-ranked primitive lattice, arXiv 2609.18191) | The menu mixes guidance, rule output and emergency actions, and the learner is trained by DAgger then PPO | Reviewer may say "this is just discrete action space". Answer: the menu is the hierarchy's interface, and duplicates are masked |
| The learner also sees the planner's per-candidate 20 s forecast | Anticipatory risk-guided RL (Mei et al., arXiv 2607.23565) gives a risk map to the policy; OmniRisk ranks primitives | Per-candidate rollout against predicted intruders, produced by the same planner the shield uses | Moderate. I only read their abstracts |
| Shield-form planner ("fly the rule unless a violation is forecast") | Policy Library CBF 2026 | Nearly the same idea. Treat as a precedent, not as novelty | High if claimed as new |
| CASA-style rules (120 m ceiling, 30 m from people, give way to helicopters) as a non-overridable layer | Benali and Hamoud 2026 (compliance monitoring in a Bristol simulation study) | Ours enforces them in the loop; theirs monitors | Moderate |
| Whole stack: satellite-image fixes into EKF2, learned decider, hard rules, marker landing, wind policy | UAstack (stack), NGPS (localization), SafeLand (landing), VO-Safe (GNSS-free RL) | I found no single paper with all of these | Low risk to state as "to our knowledge" |
| Wind policy = acceleration feed-forward head (<= 3 m/s^2, high-passed 4 s) plus a 0.04 s vertical-action decider, trained in an MJX twin with MIL-F-8785C turbulence and 1-cosine gusts, checked flight-by-flight in PX4 SITL | Ishihara 2023 (torque residual, PX4, Gazebo), WaveLander 2026 (vertical speed), GustPilot 2026 (velocity + INDI) | Two heads, two time scales, landing task, up to W10 = 20 m/s, validation ladder | Moderate. Each head has precedent. The pairing and the ladder are the novelty |

### Honest weaknesses a reviewer will spot

1. No hardware flights yet. Most nearest works have at least one (VO-Safe, NavRL, Zhang, GustPilot, Ishihara, WaveLander, SafeLand, UAstack). Say clearly: "validated in simulation (MJX, then PX4 SITL in Gazebo)".
2. Zongo and Wei 2026 (arXiv 2607.10014) found that overriding a learned policy's actions with a hand-made CBF gave no measurable safety gain, while correcting its observations cut near mid-air collisions by 90 %. Their setting is different (speed-only separation, 1 s steps), but it is a direct question for any design with a non-overridable layer above a learner. A plain counter: report how often the hard layer overrides the decider.
3. UAstack reports that its filter turns crashes into stagnation. Our "commit-on-lock" rule exists for the same reason. Worth saying.
4. Wind: Amendola et al. 2024 (landing survey) list wind as an untackled challenge for RL landing. That helps the wind paper's motivation.

---

## (e) Suggested related-work sentences for a WIND-ONLY paper

Style: short, active, no semicolons, no contractions. Every cited source was opened (see section (f)). Bracketed names are placeholders for your BibTeX keys.

1. Learned methods for wind rejection in quadrotors fall into three groups. The first learns the aerodynamic effect and adapts a classical controller online, such as Neural-Fly [O'Connell 2022], which learns the wind's aerodynamic effect, and DATT [Huang 2023], which trains a feedforward-feedback policy with RL and adds an L1 disturbance estimator on hardware.
2. The second keeps a classical controller and learns a correction on top of it. Ishihara et al. trained a residual policy with SAC in a Gazebo and PX4 simulation. On real hardware in a fan wind of about 13 m/s it cut the mean position deviation by 35 percent and the maximum deviation by 53 percent [Ishihara 2023]. General residual RL was introduced for manipulation by Johannink et al. and Silver et al. [Johannink 2019, Silver 2018].
3. The third learns the reference that a robust inner loop tracks. GustPilot learns a velocity reference and tracks it with an INDI controller, and reaches a 94.7 percent overall success rate against 55.0 percent for a PID baseline in real flights on a 50 g drone with wind up to 3.5 m/s [Habel 2026].
4. For landing, TornadoDrone trained a DRL agent in gym-pybullet-drones against wind forces on a moving platform and flew it on a Crazyflie [Peter 2024]. WaveLander lets an RL policy choose only the vertical velocity reference during landing on a wave-disturbed platform and keeps lateral tracking and attitude control in a conventional flight controller [Li 2026].
5. A recent survey of RL for drone landing lists wind disturbance among the open challenges [Amendola 2024].
6. Our policy differs in three ways. It combines an acceleration feed-forward term for PX4 with a fast discrete decision on the vertical action. It is trained with PPO on 4,096 drones in a MuJoCo MJX model of the x500 and the PX4 control cascade, using MIL-F-8785C turbulence and 1-cosine gusts. It is checked flight by flight in PX4 software-in-the-loop with Gazebo.
7. As in these works, the learned part sits above a classical flight controller, so the controller keeps its role in attitude and rate stabilisation [Lin 2024, Li 2026].
8. We report simulation results only. Hardware flights remain future work.

Notes for you (not for the paper): sentence 2 uses "about 13 m/s" because the paper's abstract says "greater than 13 m/s" but its text limits the fan to 13 m/s. The headline "about 50 %" in the abstract matches only the maximum deviation, so I give the mean (35 %) and the maximum (53 %) separately. Sentence 6 describes your own work from the brief, check it against the ACRA draft before use.

---

## (f) Sources I opened

| Ref | Source | How I opened it |
|---|---|---|
| Lin 2024 | F. Lin, C. Wei, R. Grech, Z. Ji. VO-Safe Reinforcement Learning for Drone Navigation. ICRA 2024. doi 10.1109/ICRA57147.2024.10611487 | Local PDF /home/sai/RL_projects/papers/acra2026_refs/lin2024vosafe.pdf, read in full |
| Benali 2026 | Z. Benali, A. Hamoud. Simulation-Driven Approach to Evaluate a RL-Based Navigation System for Last-Mile Drone Logistics. Vehicles 8(4):85, 2026. doi 10.3390/vehicles8040085 | Local PDF /home/sai/RL_projects/papers/vehicles-08-00085.pdf, Table 1 and sec. 2.2-2.3 |
| Dharmadhikari 2026 | M. Dharmadhikari et al. The Unified Autonomy Stack. arXiv 2605.12735. https://arxiv.org/abs/2605.12735 | alphaXiv report + abstract |
| Xu 2025 | Z. Xu, X. Han, H. Shen, H. Jin, K. Shimada. NavRL: Learning Safe Flight in Dynamic Environments. IEEE RA-L 10(4), 2025. https://arxiv.org/abs/2409.15634 | Local PDF (xu2025navrl_arxiv.pdf), sections read |
| Zhang 2026 | J. Zhang et al. High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning. IEEE RA-L (per the arXiv comment field). arXiv 2602.08653. https://arxiv.org/abs/2602.08653 | arXiv abstract page + full text (checked in verification) |
| Kim 2026 | T. Kim, H. Okamoto, B. Hoxha, G. Fainekos, D. Panagou. Policy Library CBF. arXiv 2605.16588. https://arxiv.org/abs/2605.16588 | arXiv abstract page |
| Loquercio 2021 | A. Loquercio et al. Learning High-Speed Flight in the Wild. Science Robotics 6(59), 2021. doi 10.1126/scirobotics.abg5810. https://arxiv.org/abs/2110.05113 | arXiv PDF text (DAgger, Metropolis-Hastings expert, three trajectories, MPC and BetaFlight all found in the paper text during verification) |
| Song 2020 | Y. Song, D. Scaramuzza. Learning High-Level Policies for MPC. IROS 2020. https://arxiv.org/abs/2007.10284 | arXiv abstract page |
| Ishihara 2023 | Y. Ishihara et al. Improving Wind Resistance Performance of Cascaded PID Controlled Quadcopters using Residual RL. arXiv 2308.01648. https://arxiv.org/abs/2308.01648 | Full text (PDF) read; venue not checked |
| Li 2026 | C.-K. Li et al. WaveLander. arXiv 2607.01281. https://arxiv.org/abs/2607.01281 | alphaXiv report + local PDF listed |
| Habel 2026 | A. A. Habel et al. GustPilot. arXiv 2603.19966. https://arxiv.org/abs/2603.19966 | Full text read |
| Polvara 2020 | R. Polvara, M. Patacchiola, M. Hanheide, G. Neumann. Sim-to-Real Quadrotor Landing via Sequential Deep Q-Networks and Domain Randomization. Robotics 9(1):8, 2020. doi 10.3390/robotics9010008 | Local PDF polvara2020.pdf |
| Amendola 2024 | J. Amendola, L. R. Cenkeramaddi, A. Jha. Drone Landing and RL: State-of-Art, Challenges and Opportunities. IEEE OJITS 5, 2024. doi 10.1109/OJITS.2024.3444487 | Local PDF amendola2024.pdf |
| Gross 2026 | M. Gross et al. SafeLand. arXiv 2603.17430. https://arxiv.org/abs/2603.17430 | arXiv abstract page |
| Kulkarni 2024 | M. Kulkarni, K. Alexis. RL for Collision-free Flight Exploiting Deep Collision Encoding. ICRA 2024. https://arxiv.org/abs/2402.03947 | alphaXiv report |
| Kulkarni 2025 | M. Kulkarni, W. Rehberg, K. Alexis. Aerial Gym Simulator. RA-L 2025. https://arxiv.org/abs/2503.01471 | arXiv abstract page |
| Xing 2024 | J. Xing, A. Romero, L. Bauersfeld, D. Scaramuzza. Bootstrapping RL with Imitation for Vision-Based Agile Flight. CoRL 2024. https://arxiv.org/abs/2403.12203 | arXiv abstract page |
| Romero 2025 | A. Romero, E. Aljalbout, Y. Song, D. Scaramuzza. Actor-Critic MPC. IEEE T-RO vol. 42, 2026 (online Dec 2025), doi 10.1109/TRO.2025.3644945. https://arxiv.org/abs/2306.09852 | arXiv abstract page; volume/year from Crossref |
| Kaufmann 2023 | E. Kaufmann et al. Champion-level drone racing using deep RL. Nature, Aug 2023. doi 10.1038/s41586-023-06419-4 | Full text via Europe PMC (PMC10468397) in verification: "desired collective thrust and body rates". Nature 620:982-987 |
| O'Connell 2022 | M. O'Connell et al. Neural-Fly. Science Robotics 7(66), 2022. https://arxiv.org/abs/2205.06908 | arXiv abstract page |
| Huang 2023 | K. Huang, R. Rana, A. Spitzer, G. Shi, B. Boots. DATT. arXiv 2310.09053. https://arxiv.org/abs/2310.09053 | arXiv abstract page; CoRL 2023 confirmed from the paper's own footer ("7th Conference on Robot Learning (CoRL 2023)") |
| Peter 2024 | R. Peter et al. TornadoDrone. arXiv 2406.16164. https://arxiv.org/abs/2406.16164 | Local PDF peter2024.pdf |
| Johannink 2019 | T. Johannink et al. Residual RL for Robot Control. arXiv 1812.03201 (ICRA 2019) | Local PDF |
| Silver 2018 | T. Silver et al. Residual Policy Learning. arXiv 1812.06298 | Local PDF |
| Alshiekh 2018 | M. Alshiekh et al. Safe RL via Shielding. AAAI-18 | Local PDF alshiekh2018.pdf |
| Ross 2011 | S. Ross, G. Gordon, D. Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011 | Local PDF ross2011.pdf |
| Zongo 2026 | A. Zongo, P. Wei. Runtime Safety Filtering for Learned Small UAS Separation Policies under GNSS Degradation. arXiv 2607.10014. https://arxiv.org/abs/2607.10014 | Full text read |
| Ramezani 2026 | M. Ramezani, H. Voos. Rule-based High-Level Coaching for Goal-Conditioned RL in SAR UAV Missions. arXiv 2604.26833. https://arxiv.org/abs/2604.26833 | Full text read |
| Sharma 2026 | S. Sharma. NGPS. IROS 2026. https://arxiv.org/abs/2607.18936 | Full text read |
| Mei 2026 | Y. Mei et al. Anticipatory Risk-Guided RL for Safe Flight Through Dynamic Clutter. arXiv 2607.23565. https://arxiv.org/abs/2607.23565 | arXiv abstract page only |
| He 2026 | Y. He et al. OmniRisk. arXiv 2609.18191. https://arxiv.org/abs/2609.18191 | arXiv abstract page only |
| Kochenderfer 2012 | M. J. Kochenderfer, J. E. Holland, J. P. Chryssanthacopoulos. Next-Generation Airborne Collision Avoidance System. Lincoln Laboratory Journal 19(1), 2012 | PDF text read: ACAS X logic = optimised numeric lookup table from a probabilistic model and dynamic programming |
| MIT LL news | "Award-winning technology prevents aircraft collisions". https://www.ll.mit.edu/news/award-winning-technology-prevents-aircraft-collisions | Page summary: ACAS sXu is the small-drone member of ACAS X, built by MIT Lincoln Laboratory with FAA, MITRE and JHU APL |
| Cone 2022 | C. Cone, M. Owen, L. Alvarez, M. Brittain. Reward Function Optimization of a DRL Collision Avoidance System. arXiv 2212.00855 | arXiv abstract page |
| Li 2019 | S. Li, M. Egorov, M. Kochenderfer. Optimizing Collision Avoidance in Dense Airspace using DRL. ATM Seminar 2019. arXiv 1912.10146 | arXiv abstract page |
| Lyons 2018 | M. Lyons et al. Bird interactions with drones, from individuals to large colonies. Australian Field Ornithology 35:51-56, 2018, doi 10.20938/afo35051056 (bioRxiv preprint 10.1101/109926) | Local PDF lyons2018.pdf (first page only); venue from the PDF and Crossref |

### Angle-by-angle coverage (what I found, what I did not)

1. Hierarchical RL waypoints/velocities over a classical controller: VO-Safe, NavRL, Kulkarni and Alexis, GustPilot, WaveLander, SDQN, Zhang 2026, UAstack. Done.
2. RL over planner candidates, learned high level over MPC: Loquercio 2021, Song and Scaramuzza 2020, Romero 2026, OmniRisk, Policy Library CBF. Done.
3. Shields, CBFs, runtime assurance: Alshiekh 2018, NavRL, Zhang 2026, UAstack, Policy Library CBF, Zongo and Wei 2026. For Simplex-style runtime assurance I only saw search hits (for example the Black-Box Simplex paper, arXiv 2102.12981), not opened.
4. Residual RL / feed-forward in wind: Ishihara 2023, Neural-Fly, DATT, GustPilot, TornadoDrone, Johannink, Silver. Done.
5. Delivery landing pipelines with markers and GNSS-denied localization: SafeLand, SDQN, NGPS, Schroder et al. 2025 (arXiv 2505.12176, abstract only), Yao et al. 2026 (arXiv 2601.11078, marker search in AirSim, abstract only). No RL + rules + satellite-localization + marker-landing system found.
6. DAA vs crewed aircraft and birds: ACAS X/sXu (Kochenderfer 2012, MIT LL news), RL collision avoidance for UAS (Cone 2022, Li 2019), Benali and Hamoud 2026. For birds I found the biology (Lyons 2018) but NO learning-based bird-dodge paper in two searches, so the magpie-dodge reflex has no direct prior in what I opened.
7. NTNU ARL: Kulkarni and Alexis 2024, Aerial Gym 2025, UAstack 2026. Done.
8. Imitation then RL for drones: Loquercio 2021, Xing 2024, Ross 2011. Done. No paper found that copies a shield planner and then fine-tunes by PPO for a discrete decision layer.

## What I could not verify (do not cite these on my word)

1. (Verification added Crossref metadata only: Wei, Zhao, Shu et al., Drones 10(6):448, 2026, "A Review of Reinforcement Learning for Multirotor UAVs from a Hierarchical Control Perspective: Biomimetic Architecture and Sim-to-Real", doi 10.3390/drones10060448, and Huang, Liu, Wu et al., Drones 10(1):2, doi 10.3390/drones10010002. Their content is still unread.) MDPI pages returned "Access Denied" (also blocked WebFetch): the Drones 10(6):448 review "RL for Multirotor UAVs from a Hierarchical Control Perspective" and Huang et al., Drones 10(1):2 "Safe UAV control against wind disturbances via demonstration-guided RL". I saw them only as search hits (and the second as a line in GustPilot's related-work section). The first looks very relevant to your question and is worth opening in a browser.
2. (Updated in verification.) Kaufmann 2023 was checked in full text via Europe PMC. DATT's venue is confirmed. The ACAS sXu claim rests on the MIT LL news page, which names MIT LL, FAA, MITRE and JHU APL and says sXu is for small drones. Still abstract level only: Neural-Fly, Xing 2024, Mei 2026, OmniRisk (all four re-read at abstract level in verification, claims match).
3. Song and Scaramuzza 2020: I did not check whether the experiment was flown on hardware.
4. (Resolved in verification: the Loquercio DAgger, privileged-expert and three-trajectory statements are in the paper text.)
5. The closeness scores are my judgement.
6. No GPU or simulation was used for this task (pure desk study, repo untouched).

---

## Verification

Checked 4 Oct 2026 by a second agent. Method: every arXiv item was opened on arxiv.org (title, authors, date, comment/journal fields); journal items were checked against Crossref records; full texts were opened as PDFs for the papers whose content the report leans on; claims about our code were re-derived from the repo (read-only). Totals, over 41 cited sources (36 in the reference table plus Schroder, Yao, Black-Box Simplex and the two MDPI papers): **sources VERIFIED 29 (two of them, the MDPI papers, on metadata only), CORRECTED 10, REMOVED 0 sources**. Claims about our code and wording: 13 corrections listed below (10 are the sources just counted, the other 3 are the 5-option menu, the Gazebo-only note and the Neural-Fly/DATT wording), and 1 removed figure (NGPS 'above 40 m'). Code numbers re-derived: 12 verified, 1 corrected.

### Corrected (what was wrong, what it now says)

| # | Item | Problem | Source I opened |
|---|---|---|---|
| 1 | Ishihara 2023, "about 50 % less deviation in real winds of about 13 m/s" (table row 8 and related-work sentence 2) | On the 13 m/s fan the paper reports 35 % less mean and 53 % less maximum deviation. The 50 % is only the headline. Blower test: 41 % mean, 52 % maximum. Now stated as such | arxiv.org/abs/2308.01648, PDF, sec. V |
| 2 | Benali and Hamoud 2026 called "the survey your professor cited" | It is a simulation study (PPO, Bristol city centre, SQL compliance log) with a literature table, not a survey. Wording fixed. Which document your professor meant is unknown | vehicles-08-00085.pdf, abstract and sec. 1-2; Crossref doi 10.3390/vehicles8040085 (Vehicles 8(4), 85, 2026) |
| 3 | Our vertical decider "4-action menu" (WaveLander row) | The code has 5 options: the rule's own, down 1.5, down 0.6, hold, up 1 m/s | conditions/wind/land_env.py line 40, conditions/wind/README.md sec. 4 |
| 4 | Zhang 2026 "venue not stated, preprint"; author list | Venue is IEEE RA-L (arXiv comment). Seven authors, Hoi was missing | arxiv.org/abs/2602.08653 |
| 5 | Song and Scaramuzza 2020 "real-flight status not checked" | Simulated quadrotor only, no real flight in abstract or text | arxiv.org/abs/2007.10284 and PDF (searched for real-world, hardware) |
| 6 | NGPS "flies above 40 m" | Flights are at 60-150 m AGL | arxiv.org/abs/2607.18936 |
| 7 | Romero 2025, T-RO | Published in IEEE T-RO vol. 42, 2026 (online Dec 2025), doi 10.1109/TRO.2025.3644945 | Crossref |
| 8 | SDQN: survey "calls it the only hierarchical-RL landing work" | The survey cites the 2018 ICUAS paper by Polvara et al., the earlier version, not the 2020 Robotics paper | amendola2024 PDF, sec. on static platforms and ref. [13] |
| 9 | Lyons 2018 cited as bioRxiv only | Published in Australian Field Ornithology 35:51-56 (2018) | local PDF header; Crossref doi 10.20938/afo35051056 |
| 10 | DATT venue "secondary" | CoRL 2023 confirmed from the paper's own footer | arxiv.org/pdf/2310.09053 |
| 11 | Neural-Fly/DATT lumped as "learn the aerodynamic effect" | DATT trains a feedforward-feedback policy with RL plus an L1 disturbance estimator. Sentence reworded | arxiv.org/abs/2310.09053 abstract |
| 12 | Ramezani and Voos, "UAV delivery missions" | Motivated by search and rescue, tasks are delivery. Wording fixed | arxiv.org/abs/2604.26833 |
| 13 | "6 m gate" and "3D planner, 2 m clearance" given as part of the stack | True in code but only in the Gazebo flights (shared/px4_gazebo/gz_learner.py line 939; shared/fast_sim/local_planner.py: A*, hard pad 2 m, soft penalty to 5 m). The fast simulator has neither. Note added | repo, read-only |

### Removed
- The two "from an AI-generated report" caveats on Loquercio are gone because the paper text now confirms them (DAgger, Metropolis-Hastings expert, three trajectories, MPC, BetaFlight). Nothing that the report asserted as a result had to be deleted for lack of a source. The only removal is the incorrect "above 40 m" NGPS figure (item 6), which had no support anywhere in the paper.

### Verified (title, first author, year, venue and the claim made; source opened)
- **arXiv pages opened (title, authors, date, comment all match)**: Dharmadhikari 2605.12735; Xu 2409.15634 (RA-L 10(4), 2025, Crossref doi 10.1109/LRA.2025.3546069); Kim 2605.16588; Loquercio 2110.05113 (Sci. Robotics 6(59)); Li WaveLander 2607.01281; Habel GustPilot 2603.19966; Gross SafeLand 2603.17430; Kulkarni and Alexis 2402.03947 (ICRA 2024); Kulkarni Aerial Gym 2503.01471 (RA-L 10(4), 2025); Xing 2403.12203 (CoRL 2024); O'Connell 2205.06908 (Sci. Robotics 7(66)); Peter 2406.16164; Johannink 1812.03201 (ICRA 2019); Silver 1812.06298; Zongo 2607.10014 (DASC 2026); Ramezani 2604.26833; Sharma 2607.18936 (IROS 2026); Mei 2607.23565 (IROS 2026); He OmniRisk 2609.18191; Cone 2212.00855; Li 1912.10146; Schroder 2505.12176; Yao 2601.11078; Mehmood Black-Box Simplex 2102.12981.
- **Content claims checked in full text**: UAstack (SDF-NMPC and PPO ExRL swappable, acceleration and yaw-rate setpoints, Aerial Gym training, C-CBF last-resort filter, ship cargo hold and legged tests, crashes turned into stagnation, no wind, no satellite localization, no regulation, no birds); NavRL (PPO, Isaac Sim thousands of drones, VO shield by linear programming, zero-shot); Zhang (PPO attitude and thrust at 100 Hz, PX4, CNN+GRU, Dijkstra reward, HOCBF, 7.5 m/s); WaveLander (scalar vertical velocity clipped to +-0.45 m/s, GRU actor-critic, Isaac Lab training, MuJoCo, Isaac Sim + ArduPilot SITL, one real test in wind); GustPilot (PPO velocity, INDI, Crazyflie 50 g, 94.7 vs 55.0 %, 3.5 m/s); Policy Library CBF (library of fallback policies, parallel rollouts, 4/8/12-state sims); SafeLand (behavior tree, 200 sims, 60 field tests, 95 %); Zongo (action filtering negligible, observation filtering cuts near mid-air collisions 90 %, 1 s step, speed-only action); Ramezani (rule-based advisor over online RL); NGPS (SuperPoint + LightGlue + RANSAC, ArduPilot EKF3, 2.94 m RMSE); VO-Safe (8 directions, 1 m, PPO, PID cascade, Flightmare, ORB-SLAM2, RealSense and white-paper real test, "platform-agnostic" sentence); Ishihara (SAC, 10 Hz, torque residual, Gazebo + PX4, 15-25 m/s training wind); Polvara 2020 (sequential DQN, Gazebo, Bebop 2); TornadoDrone (gym-pybullet-drones, Crazyflie 2.1); Amendola (wind named as open challenge, 2024, OJITS 5, doi 10.1109/OJITS.2024.3444487); Kaufmann 2023 (Nature 620, thrust and body rates, via Europe PMC full text).
- **Crossref-confirmed journal records**: Polvara (Robotics 9(1):8), Amendola, Lin (ICRA 2024 doi), Kaufmann, Loquercio, Neural-Fly, Alshiekh (AAAI-18 vol. 32), Johannink (ICRA 2019).
- **Others**: Kochenderfer, Holland, Chryssanthacopoulos, Lincoln Laboratory Journal 19(1), 55-71, 2012 (LL publication listing); MIT LL ACAS page (sXu for small drones; MIT LL with FAA, MITRE, JHU APL named). Ross 2011, Alshiekh 2018 and Silver 2018 title pages read in the local PDFs.

### Numbers about our code (re-derived from the repo, read-only)
- 14 candidates, 0.5 s decisions: shared/fast_sim/urban_rl.py `N_ACT = 14`, docstring "Every 0.5 s". The 14 are: the rule's own, land-anyway, nominal, turn +-45, +-90, 180 (reverse), half speed, hover, straight up, flee the helipad, hold altitude, climb (urban_agents.py `candidates`). Report's list matches.
- 20 s forecast per candidate: urban_agents.py planner docstring. Duplicates masked: urban_rl.py docstring.
- CASA rules: 120 m ceiling, 30 m from people, move away from helicopters near a helicopter landing site: shared/fast_sim/urban_env.py header.
- Wind: 4,096 drones (README line 34, queue script `--envs 4096`), decision 0.04 s (`DT_C`), feed-forward 3 m/s^2 (`A_FF = 3.0`), high-pass 4 s (`--hp-tau 4`), W10 ladder to 20 m/s, MIL-F-8785C and 1-cosine gusts (README sec. 2). All match sentence 6 of section (e).
- **Re-run (two computed numbers)**: the rule on 1,024 paired flights, seed 10,000 (conditions/wind/land_env.py evaluate, fly=False, decide=False, GPU with 5.5 GB free, nice 10, no prealloc, no repo writes). W10 = 15: 78.1 % landed (stored 77.9 %, README 78 %). W10 = 18: 5.3 % landed (stored 5.6 %, README and brief 5.5 %). Another seed at 15 m/s gave 75.0 %, so the seed spread is a few points. The report itself quotes none of these, so no edit was needed there, but quote "about 5 %" at 18 m/s rather than 5.5 % if you want a number that survives re-runs.

### Not verified (left as the report said)
- MDPI Drones 10(6):448 and 10(1):2: existence and metadata confirmed via Crossref, content not read (page blocks access).
- The closeness scores in section (b) are the first agent's judgement.
- "No paper combines all of ours" is a statement about the search, not a fact I can confirm. I opened no source that contradicts it.
- ACAS X, Neural-Fly, Xing, Mei, OmniRisk: abstract level.
