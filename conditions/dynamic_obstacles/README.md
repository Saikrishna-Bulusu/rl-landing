# Dynamic obstacles: things that move in the last kilometre

The last ~1 km of a delivery in Logan, QLD, with GPS-grade navigation, flown 4,096 deliveries at a time on the GPU
(`shared/fast_sim/urban_env.py`). This is the "clear" world: no wind, no fog, no bird, nothing the survey misses.
Only movers are in it:

| mover | how it behaves | rule it breaks |
|---|---|---|
| air ambulance | arrives in half the flights (`P_HELI` 0.5) to a real OSM helipad; always from the north-west in the corridor world, any bearing in the random world | CASA: within 1.4 km of a helicopter landing site, move away from one that is near or landing; ASTM F3442 near mid-air 152 m / 30 m |
| other delivery drones | 0-6 of them, radials to and from the hub (corridor) or between random points (random) | 30 m separation (assumed: CASA gives no number) |
| people | up to 2 near the pad (half the flights, `P_PEOPLE`), they leave after 30-240 s (assumed) | CASA: stay 30 m from people, never over one; the recipient counts |
| cars | 6 on the streets | 30 m, same rule |

Every episode is an encounter set, not a day in Logan: the helicopter turns up far more often than at a real hospital.
Report results "given an encounter" and weight by real frequency afterwards.

**Gazebo has none of this.** `decider_server.py` reports helicopters, drones, people and cars absent when it decides for a
PX4 flight, so this condition lives only in the fast simulator. (Moving things that Gazebo does have: the magpie, `../magpie/`.)

## Before you start

* Run everything below **from `shared/fast_sim/`** (maps, checkpoints and jsons are read by bare name).
* `P=/home/sai/miniconda3/envs/mujoco_drones/bin/python`
* The map `logan_map_lidar.npz` must exist. If it does not: `shared/fast_sim/lidar_map.py`, then `logan_map_lidar.py` (root README section 2).
* Check the GPU first (`nvidia-smi`): a ladder uses about 3.3 GB, a training about 2 GB, never both together with a Gazebo flight.

```bash
cd shared/fast_sim
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

---

## 1. How much room is there? (the headroom ladder, run before any learning)

The same 4,096 episodes go to six deciders, so every comparison is paired:

| step on the ladder | what it tells you |
|---|---|
| rule | what a careful practitioner writes |
| best same-sensor planner | better decisions from the same information |
| all-knowing planner | knowing this flight's future (an oracle) |
| 100 - oracle | failures no decider avoids with these sensors |

A guard stops the run if the oracle scores below the rule (an upper bound that loses to its baseline is broken, not informative).

1. The clear world, GPS navigation, LiDAR map. A few minutes to under an hour (old notes say 30-60 min; the shorter sensitivity runs take about a minute).

```bash
$P urban_headroom.py --map logan_map_lidar.npz --nav gps
```

   Writes `headroom_lidar_gps.json`. Expect "guard: ... the bound is valid", then the ladder:

| deciders on the ladder | clean delivery (corridor) |
|---|---|
| rule | 93.7% |
| best same-sensor planner | 94.5% |
| all-knowing planner | 95.4% |

   Total room over the rule is +1.8 pp (decisions +0.9, knowing the future +0.9). Clean = delivered with no violation all flight.
   The first version (GeoAnchor navigation, old map) was `$P urban_headroom.py`: rule 93.2, planner 95.3, oracle 96.5, `headroom.json`.
   GPS against GeoAnchor changes every decider by 0.5 pp or less: the camera sees the pad from ~44 m out, which absorbs the nav error.

2. Does the ladder survive the people assumption? It reruns the ladder with people leaving in 0-60 s, 30-240 s (now), and 60-300 s.

```bash
$P ../../conditions/dynamic_obstacles/sweep_people.py
```

   Prints three lines. If only the "unavoidable" figure moves, the ladder is a property of the deciders; if its shape moves, it was the assumption.

3. The chart of where the failures are (reads `headroom.json`).

```bash
$P ../../conditions/dynamic_obstacles/plot_headroom.py
```

   Writes `outputs/headroom.png` in the repo root.

4. Sensitivity of the ladder and the trained learners to other assumptions, no retraining. About 1-2 min each.

```bash
SENS_LEAVE=10,120 $P sens.py clear leave10_120
```

   Writes `sens/clear_leave10_120.json`. The other knobs are `SENS_T_MAX`, `SENS_DRONE_SEP`. The runs on disk: the learner's gain over the planner stays +1.0 to +2.1 pp
   for leave times and drone separation, falls to +0.7 to +1.0 with a 400 s limit, and turns negative (-0.2 to -0.6) with a 200 s limit.
   So the learner's gains lean on the time limit (300 s) and the people's leave time: say so wherever they are quoted.

## 2. The copy check (before blaming learning)

When a trained policy loses to a hand-built planner, first copy the planner into the policy's own network (same inputs, same 14 manoeuvres, same 0.5 s
decisions) and fly the copy. It splits the failure in two:

* copy much better than PPO: PPO did not find a policy the network can hold (search failure);
* copy much worse than the planner: the inputs cannot carry what the planner computes (representation gap).

1. About 1 min of GPU.

```bash
$P clone_check.py
```

   Writes `clone_check.json`, `clone_corridor.pt`, `clone_random.pt`. Expect for each world a line "CLONE greedy ... clean 94.1%".

| decider | clean (corridor) |
|---|---|
| PPO from scratch (410k episodes) | 91.8% |
| the copy of the planner | 94.1% |
| the planner itself | 95.3% |

   Both failures were present: 2.3 pp of search failure, 1.1-1.5 pp of representation gap (the copy reproduced only 48% of the planner's overrides of the rule).
   The fix is `--eyes`: hand the policy the planner's 20 s forecast for each manoeuvre, then let PPO look for what that window misses.

## 3. Train the decision layer

Every 0.5 s the policy picks one of 14 manoeuvres the rule, the planners and the oracle also choose from (the rule's own action, land anyway, sidesteps,
half speed, hover, climb, flee the helipad...). Same candidates everywhere, so any difference is the decision and nothing else.

Reward is priced against the goal: +1 clean delivery, -1 crash, -1 per near mid-air, -0.25 per violation-second, -0.001 per decision (300 s of waiting
costs 0.6, less than a clean delivery is worth). Candidates that give the identical command are masked, because PPO's choice among them earns nothing and drifts.

1. Three seeds, with the planner's eyes. About 2-3 h per seed on this GPU.

```bash
$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --seeds 0 1 2
```

   What to watch while it runs: the log line per round (`clean`, `violation`, `rule_share`, `planner_share`). Early on `planner_share` is ~96% (it starts as the copy);
   late rounds drift down, so the trainer keeps the best round per read-out on 2,048 validation flights (seed 3030).

2. Writes (in `shared/fast_sim/`):

| file | what |
|---|---|
| `rl_corridor_eyes_lidar_gps_s<k>_{mean,sampled,greedy}.pt` | best round per read-out (average command, dice, top choice) |
| `rl_corridor_eyes_lidar_gps_s<k>_last.pt`, `..._s<k>.pt` | the last round, and the final network |
| `rl_eyes_lidar_gps_corridor_0_1_2.json` | the log, and each read-out's score on the ladder's 4,096 test flights, paired against rule and planner |

3. ClearML logs live while it trains: project `rl-landing/fast_sim`, one task per seed (`rl_eyes_lidar_gps_corridor_s<k>`). Past logs are in `rl-landing/fast_sim/history`.
   Add `--no-clearml` to turn it off. Check the import has nothing left to do:

```bash
$P clearml_import.py --dry-run
```

   Expect "would import: 0 to rl-landing/wind/history, 0 to rl-landing/fast_sim/history".

4. Other flags: `--world random` (arrival bearing uniform), `--kl X` (price of straying from the copy, an anchor), `--gens N`, `--n-env N` (episodes per generation, default 2048),
   `--tag NAME`. Without `--eyes` it is PPO from scratch.

5. Result (the learner at its `mean` read-out, corridor, GPS, LiDAR map, 4,096 test flights):

| seed | clean | over the rule | over the planner |
|---|---|---|---|
| 0 | 96.6% | +2.9 pp | +2.1 pp |
| 1 | 96.4% | +2.8 pp | +1.9 pp |
| 2 | 96.6% | +3.0 pp | +2.1 pp |

   All paired intervals exclude zero. Where it wins: time. It cuts helicopter give-way timeouts and people-at-pad timeouts, and rule breaks rise slightly.
   The corridor does not help it more than the random world does, so the "learning pays where there is a pattern" idea is not supported.
   The all-knowing planner is not a ceiling: the learner beats it.

6. The same analysis in numbers and pictures.

```bash
$P why_cond.py clear
$P plot_conditions.py
$P paper_figs.py world headroom learning
```

   `why_cond.py` writes `why_cond.json` (failures by first reason, planner against learner). `plot_conditions.py` writes `conditions.png` here (success against training episodes, all conditions).
   `paper_figs.py` writes `../../outputs/paper/figs/*.pdf` and a png preview.

## 4. Where the numbers live

| file (in `shared/fast_sim/`) | holds |
|---|---|
| `headroom_lidar_gps.json`, `headroom.json` | the ladders (clear world; GPS+LiDAR, and the first GeoAnchor version) |
| `rl_eyes_lidar_gps_corridor_0_1_2.json`, `rl_eyes_lidar_corridor_0_1_2.json` | the learners (GPS, GeoAnchor navigation) |
| `rl_corridor_0.json`, `rl_random_0.json` | the first PPO from scratch (91.8%) |
| `clone_check.json` | the copy check |
| `sens/clear_*.json` | the sensitivity runs |
| `why_cond.json` | why each condition's learner wins or loses |

## 5. What this condition taught us

* The room above a careful rule is about 2 pp on a clear day; most of what is left is people lingering at the pad, driven by an assumption, not a measurement.
* Learning pays where information is partial (fog, the magpie through a camera rig: see `../weather/`, `../magpie/`), not where the world is fully seen.
* Before adding training time or seeds to a losing run: copy the planner first, then check how often candidates coincide.
* A headroom number that comes from a made-up scenario density is a made-up number: take densities from real references.
