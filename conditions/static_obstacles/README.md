# Static obstacles: things that stand still

Two kinds, handled in two different ways (Sai, 25 Sept: plan for what the map shows, learn for what moves).

| kind | example | what handles it |
|---|---|---|
| in the 2023 survey | houses, trees, poles | the survey map (an absolute map: it inherits the drone's navigation error) |
| not in the survey | a 58 m tower crane on the approach, street lights, a traffic light, a new tree, a shed | the voxel map built from the depth camera (relative to the drone: exact whatever the navigation error) |
| trees round the pad | 10-13 trees, 15-25 m, canopy overhanging | a bigger marker, so the pad is seen from above the canopy |

* **A local 3D planner** (weighted A*) turns the rule's straight line into a path that keeps clear of everything on the map. Static safety rests on search, not on anything learned.
  The learner is not trained for static obstacles: in the fast simulator its unmapped crashes (0.40%) match the planner's (0.37%) and the all-knowing planner's (0.39%).
* The hazard is **tall temporary things on the approach** (cranes, boom lifts, 25-60 m). Sheds and sails near the pad hardly matter: the drone descends straight onto a clear circle.
* Obstacles are largely a **marker-size problem**: a bigger pad, from above the trees, beat a stereo rig, a new simulator and RL.

Setup:

```bash
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

* Fast-sim steps run **from `shared/fast_sim/`**. Gazebo and PX4 steps run **from the repo root**, **one Gazebo at a time** (root README, rule 2).
* Esri imagery never goes into git; the voxel films here use the plain world, but keep it that way.

---

## 1. The local 3D planner (no simulator, 1 second)

A* through a 61 x 61 m window (1 m columns, 2 m tall cells, up to 90 m) around the drone on the survey plus the voxel map. Blocked below a column's top + 2 m and within 2 m sideways.
Cost is length, climbing and descending 2x (going round beats going over) and a clearance penalty inside 5 m. Weighted A* (paths within 1.5x of the best) took a plan from 75 m out from 2 s down to 17 ms:
at 2 s the drone met the crane before the answer arrived. At 8 m/s the depth camera's 8-12 m gives about a second, so the planner slows to 6 m/s (`V_SENSE`) once the map matters.

1. Self-check.

```bash
$P shared/fast_sim/local_planner.py
```

   Expect: `self-checks passed: mast passed at 2.3 m via 15 waypoints; free line straight; flown: never backwards, mast seen late passed at 1.2 m, 75 m covered in 15 s`.

## 2. Unmapped obstacles in the fast simulator

Flag `--unmapped` adds 1-5 boxes near the pad (sheds, sails; 2-6 m wide, 2.5-10 m tall) and, on half the flights, something tall and temporary 15-60 m short of the pad
(25-60 m high, within 15 degrees of the approach). The drone learns of them through a forward depth camera (73 x 58 degrees, 8 m) and a downward rangefinder (40 m).
Sizes are assumed (the survey is 2023, the flights 2026). The fast simulator treats the drone as a point: its crash rates are optimistic by the x500's body radius.

```bash
cd shared/fast_sim
```

1. The ladder with unmapped obstacles. A few minutes to under an hour (old notes say 30-60 min).

```bash
$P urban_headroom.py --map logan_map_lidar.npz --nav gps --unmapped
```

   Writes `headroom_lidar_gps_unmapped.json`. Clean delivery: rule 93.3%, planner 94.3%, all-knowing 94.4%: total room +1.0 pp.
   Crashes into unmapped things (% of flights): rule 0.61, best planner 0.37, all-knowing 0.39. The earlier 2,048-flight study gave survey-only rule 2.69, voxel-map rule 1.03, planner 0.44, oracle 0.29.

2. Train a learner on it (only to show there is nothing to learn here). About 2-3 h per seed.

```bash
$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --unmapped --seeds 0 1 2
```

   Writes `rl_corridor_eyes_lidar_gps_unmapped_s<k>_*.pt` and `rl_eyes_lidar_gps_unmapped_corridor_0_1_2.json`. ClearML project `rl-landing/fast_sim` (`--no-clearml` turns it off).
   Result: 96.1 / 96.2 / 96.1% clean, +2.1 pp over the planner, all of it from fewer timeouts, not fewer crashes.

3. A film of the same deliveries flown trusting the survey alone and by the learner that flies on the voxel map. CPU only. Top: from above; bottom: from the side.

```bash
CUDA_VISIBLE_DEVICES="" nice -n 15 $P ../../conditions/static_obstacles/video_voxel.py --seed 7 --n 512
```

   Writes `voxel_video_h264.mp4` here: move it to `outputs/films/` (films are never committed).

## 3. The crane in Gazebo with PX4 and a real depth camera

The world is Logan from the survey, plus: a 58 m tower crane in the open field on the depot-to-pad line (above the 50 m cruise; its place moves per seed in the field x 42-64, |y| <= 1.5),
street lights, a traffic light whose arm reaches toward the pad, a new tree and a shed. The drone is the x500 with the OAK-D Lite depth camera (640 x 480, 73 degrees, 0.2-19.1 m).
The voxel map has 0.5 m voxels from depth up to 12 m, with stereo noise; a voxel counts after 2 frames; ground and anything within 1 m of the drone itself is dropped
(the front propellers' tips had put voxels at the drone's own spot).

1. Build the models and worlds, once.

```bash
/usr/bin/python3 shared/px4_gazebo/gz_voxel.py --setup
```

2. Bench check: the depth camera is flown along the approach by `set_pose` in its own Gazebo partition (`vox`), the images become a voxel map and the map is checked
   against the objects' true geometry. Run it after any change to the objects. It fails loudly if the frames disagree.

```bash
/usr/bin/python3 shared/px4_gazebo/gz_voxel.py --bench
```

   Writes `shared/px4_gazebo/gz_logan/voxbench/voxels.npz` and `bench.png`. Expect 100% of voxels explained, the mast's face within about 0.7 m of the true mast.

3. The flights. `voxfly.sh` starts PX4 and Gazebo for each run and shuts them down after. Each `RUNS` item is `decider:map`: `survey` trusts the 2023 survey only,
   `voxel` also the voxel map's look-ahead, `local3d` the voxel map and the local 3D planner. Whole flights are slow (0.1x real time by default, the six-camera speed); the fast version below is about 10 minutes for 30 flights on 5 instances.

```bash
OUT=first SEEDS="0 1 2" RUNS="planner:survey rule:voxel rule:local3d" ./conditions/static_obstacles/voxfly.sh
```

   Writes `outputs/gazebo_voxel/first/`: `flights.jsonl` (one line per flight, `outcome` is `passed` / `crashed`, closest pass in metres), a film and a `.csv` per flight, `failsafes.txt`, `px4.log`.

4. The bulk version, as the ten-crane tables were flown: no video, 1x speed, quick climb, flights end 15 m past the crane, five instances side by side, starts 15 s apart.

```bash
RIG=v2 FAST=1 SHORT=1 SPEED=1.0 INSTANCE=4 OUT=set10c/i4 SEEDS="1 2" RUNS="rule:local3d planner:survey planner:voxel" ./conditions/static_obstacles/voxfly.sh
```

   Knobs: `FAST=1` (no video cameras, quick climb), `SHORT=1` (short flights, with video when `FAST` is off, for films), `FULL=1` with `FAST=1` (the whole delivery, landing on the pad),
   `SPEED` (real-time factor), `INSTANCE=N` (its own PX4 instance and Gazebo partition: ports 14540+N), `RIG=v2|v2f|v2t|v2u|v2l` (the camera rig's gimbal and cameras: `../magpie/`).
   Never start two runs with the same `INSTANCE`.

Ten crane positions (set10b and set10c, judged on Gazebo's truth, `outputs/gazebo_voxel/set10b/` and `set10c/`):

| decider | crashed into the crane |
|---|---|
| survey only | 6 of 10 (set10b), 8 of 10 (set10c, rig v2) |
| voxel map look-ahead | 0 of 10, closest passes 1.2-5.4 m |
| rule + local 3D planner | 0 of 10, closest passes 4.1-6.2 m |

Three things went wrong before these were right. The first flights flew tail first (PX4 held its spawn heading, so the depth camera looked backwards): the nose now follows the path.
PX4's estimate ran 0.11 s ahead of the drone because it assumes a receiver delay Gazebo lacks: `EKF2_GPS_DELAY=0` is set in `voxfly.sh`. And the film recorder stalled the flight loop
(29 to 2 Hz): Gazebo's blocking `set_pose` now runs in a helper process (`Mover(asyn=True)` in `shared/px4_gazebo/gz_learner.py`).

## 4. The marker and the trees

The claim: detection altitude is linear in marker size. A 1.20 m marker is reliable from about 19 m (more with the measured ramp: 50% at 27 m), so with 15-25 m trees the drone must hunt for the pad
from inside the canopy layer, the only dangerous place to fly. A 1.80 m marker is acquired at cruise height (~52 m, field of view limited) and never enters the canopy.

1. Fast study with the point-mass simulator: delivered and crash rate against marker size, with the old cliff detector and the measured ramp (`shared/px4_gazebo/detect_curve_60m.json`). CPU, a few minutes.

```bash
$P conditions/static_obstacles/bigger_marker.py
```

   Expect two tables (pad 1.20, 1.80, 2.40 m) and the line "the cliff credits the bigger marker with ... pp of crash reduction; the measured detector credits it with ... pp".
   It reads the part-1 simulator in `archive/sigma/` and the pool `shared/fast_sim/sigma_pool.npz`.

BIGGER_MARKER_OUTPUT

2. The pad itself. `make_pad.py` builds the generated pad (markers 2, 1, 0 on a 5.0 m mat) and `ab_pad.sh` installs one of two pads into **PX4's model tree**
   (`/home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/models/arucotag`, outside this repo). The old pad is restored from its `.bak-1.2m` copy there. Gazebo loads the model once: restart SITL after switching.

```bash
./conditions/static_obstacles/ab_pad.sh old
./conditions/static_obstacles/ab_pad.sh new
```

   Expect "installed the 1.20 m pad" and "installed the generated pad". Nothing else changes between the two (flight height included).

3. The same five missions on real PX4, per pad (start SITL with `shared/px4_gazebo/launch_px4.sh` first, one terminal). About 5 min a mission.

```bash
/usr/bin/python3 shared/px4_gazebo/px4_mission.py --runs 5
```

   Always set `MISSION_JSON=conditions/static_obstacles/results/<name>.json`, for example `MISSION_JSON=conditions/static_obstacles/results/missions_pad18.json /usr/bin/python3 shared/px4_gazebo/px4_mission.py --runs 5`.
   Hazard: the default name is `px4_response.json` in the folder you run from, and `shared/px4_gazebo/px4_response.json` is the measured velocity lag (`measure_tau.py`): a run from that folder overwrites it.
   On disk: `missions_pad12.json`, `missions_pad18.json`, `missions_pad24.json`, `missions_pad24_alt18.json`, `trees_pad12.json`, `trees_pad18.json`.

| coarse marker | mat | landed (5 seeds) | first read | mean time |
|---|---|---|---|---|
| 1.20 m | 3.0 m | 5/5 | 46 m | 134 s |
| 1.80 m | 5.0 m | 5/5 | 53 m | 90 s |
| 2.40 m | 6.6 m | 5/5 | 53 m | 82 s (search altitude 18 m) |

   Accuracy is the same (0.3-0.4 m from centre). 1.80 m takes about 85% of the gain: the bigger marker buys time, not accuracy.

4. Twenty different forests, both pads. It replants the trees through Gazebo's create and remove services (about 1 s, against 45 s for a SITL restart), flies one mission per process, and switches the pad itself.
   Trial i flies forest seed 300+i and sigma seed i in both arms, so a pair differs only in the marker. About 3 h.

```bash
./conditions/static_obstacles/run_forest20.sh
```

   Warning: it starts and stops PX4 and Gazebo itself with `pkill`, and **edits `shared/px4_gazebo/px4_mission.py`** (`TAG_SIZES`, `PAD_IDS`) for each arm, then puts it back. Run it only on a quiet machine, and check
   `git diff shared/px4_gazebo/px4_mission.py` afterwards. `run_n20.sh` is the earlier two-arm version that waited for a running arm A.

   Logs: `outputs/logs/forest_pad18_n20.log`, `forest_pad12_n20.log`. The per-trial json files were not written for the 1.80 m arm, so read the logs:

```bash
/usr/bin/python3 conditions/static_obstacles/parse_forest.py outputs/logs/forest_pad18_n20.log outputs/logs/forest_pad12_n20.log
```

   Expect: 1.80 m "landed 20 abort 0 crashed 0", 1.20 m "landed 5 abort 11 crashed 4", clear column 3.36-7.70 m, trees 10-13. (`results/forest12_<i>.json` hold the 1.20 m arm trial by trial.)

| coarse marker | delivered | aborted | crashed | marker ever seen |
|---|---|---|---|---|
| 1.20 m | 5 / 20 | 11 / 20 | 4 / 20 | 11 / 20 |
| 1.80 m | 20 / 20 | 0 / 20 | 0 / 20 | 20 / 20 |

   Delivery: Fisher exact p = 7.7e-7. All four crashes happened where the clear column of sky above the pad was under 4.2 m wide (4 of 7 trials; 0 of 13 above 4.2 m).
   One fixed forest gave 0 crashes in both arms: vary the forest per trial.

The tree geometry comes from one place, `shared/px4_gazebo/forest_spec.py` (no dependencies), so the point-mass simulator, the world file (`make_trees.py`) and the runtime spawner (`forest.py`) cannot drift apart.
Trees in Gazebo have no collision: a crash is judged geometrically by the same test as the simulator, so a brush does not tumble the airframe and break the seed pairing.
What Gazebo adds is real occlusion: a marker behind a canopy is genuinely not decoded.

## 5. Chains that already ran (`queue/`)

Each header says what it waits for. They use PIDs and markers from the night they ran: read them, do not re-run them blindly.

| file | what it did |
|---|---|
| `vox_yawfix.sh`, `vox_seeds.sh`, `vox_seeds2.sh`, `vox_seeds3.sh` | the first crane flights, then seeds 1-5 with each fix, the rule flying the local planner's path |
| `vox_rerun.sh`, `after_rerun.sh` | the film and ten positions again after the GPS delay and truth-judging fixes |
| `set10b_missing.sh`, `set10c.sh` | the two stalled seeds; the ten positions with today's drone (rig v2) |
| `film4.sh`, `film5.sh`, `film6.sh`, `film7.sh` | the crane film's flights again (async mover; film7 with rig v2) |
| `land_fix.sh`, `land_fix2.sh`, `land_fix3.sh`, `land_gate2.sh`, `land_gate3.sh` | the landing fix: heading locked at the marker read, lag-aware centring; rig v2 (gimbal) against v2f (fixed camera), seeds 1-6 |
| `rigv2_land2.sh`, `rigv2_land3.sh` | rig v2 full deliveries after the self-return fix |
| `vox_after_yolo1.sh` | squeezed the voxel flights between the magpie YOLO flights |

## 6. Where results live

| what | where |
|---|---|
| marker and forest runs | `conditions/static_obstacles/results/` (`forest12_*`, `missions_pad*`, `trees_pad*`) |
| crane flights | `outputs/gazebo_voxel/<OUT>/` (never in git) |
| unmapped ladder and learner | `shared/fast_sim/headroom_lidar_gps_unmapped.json`, `rl_eyes_lidar_gps_unmapped_corridor_0_1_2.json` |
| measured constants (detect curve, camera, lag) | `shared/px4_gazebo/CALIBRATION.md` |

## 7. Things that bit us

* A fixed forest is silent about a whole failure mode: always vary the environment seed across trials.
* A hard minimum-pixels cliff overstated the bigger marker's benefit twofold: Gazebo decodes the 1.20 m marker at 14 px. Use the measured curve.
* The survey is absolute and the sensed obstacles relative: store sensed boxes in the drone's frame, never the believed one (a 2 m column ended 1.4 m from where the drone steered).
* Look-ahead must use the lag model and ignore bare ground (treating the ground as an obstacle delivered 0%).
* Offline checks need the planner's real latency: a synchronous replay passed the crane the live planner (2 s a plan) hit.
* Judge crashes on Gazebo's truth, not PX4's estimate.
