# rl-landing

Landing a delivery drone on the right pad in a Logan (QLD) suburb, with GPS denied. PX4 flies in Gazebo; GeoAnchor
(a separate project) gives it an absolute position from a downward camera; the pad's markers take over for the last
30 m; a rule, a local 3D planner and reinforcement-learned policies decide the landing past cranes, wind and a magpie.

## Where things are

Each folder has a README that is the runbook for that part: how to run it alone, what it produces, how long it takes.

| folder | what is in it |
|---|---|
| [`conditions/wind/`](conditions/wind/README.md) | landing in gusty wind: the MuJoCo MJX twin of the x500 + PX4, the PPO learners, rule-vs-learner videos, the Gazebo wind plugin |
| [`conditions/magpie/`](conditions/magpie/README.md) | the territorial magpie: the camera rig, the YOLO detector, magpie flights |
| [`conditions/dynamic_obstacles/`](conditions/dynamic_obstacles/README.md) | the air ambulance, other delivery drones, people and cars (fast simulator) |
| [`conditions/static_obstacles/`](conditions/static_obstacles/README.md) | the unmapped crane, the voxel map, the local 3D planner (weighted A*), trees, marker size |
| [`conditions/weather/`](conditions/weather/README.md) | fog and mist |
| [`all_conditions/`](all_conditions/README.md) | everything in one flight: the full-stack stress test |
| [`mix_and_match/`](mix_and_match/README.md) | one launcher: any combination of wind, crane, magpie, sky, GNSS and decider for a Gazebo flight (`--dry-run` first) |
| [`gnss_denied/`](gnss_denied/README.md) | GeoAnchor's visual fixes in PX4's loop: whole deliveries with GPS denied |
| [`shared/fast_sim/`](shared/fast_sim/README.md) | the GPU fast simulator of the Logan suburb that the obstacle, magpie and fog conditions train in |
| [`shared/px4_gazebo/`](shared/px4_gazebo/README.md) | PX4 SITL launch, missions, perception, the pad, the Gazebo flight core and the Logan worlds |
| `dashboard/` | flight logs and the mission planner (section 4) |
| [`papers/`](papers/README.md) | the frozen ACRA draft, four fact-checked research reports (`papers/research/`), the deck's source |
| [`archive/`](archive/README.md) | earlier studies that still run: sigma (part 1), forest and marker (part 2), the Tello cage |
| `outputs/` | everything a run writes: logs, films, run folders (never in git) |

The study grew in five parts (sigma, obstacles, the last mile, Gazebo, wind + GNSS-denied); `archive/README.md` and
`papers/acra2026/` keep that history.

The slide deck (all five parts, with the films) is a private Claude artifact:
<https://claude.ai/artifact/QaYeAMh7cZThMmffKxjjNA>. Its source is in `papers/slides/` (one HTML file per slide, order in
`papers/slides/deck.json`); the charts are in `papers/docs/figures/`. The films are **not** in git (see rule 1).

**How to use this README.** Sections 0-2 set up a machine once. Section 3 proves the stack is alive and says where to
go next. Times are for the Legion (RTX 5070 Laptop 8 GB, 32 threads, 23 GB RAM).

---

## 0. Three rules before anything else

1. **Esri Wayback imagery is for measurement only.** The Gazebo floor (2023 capture) and GeoAnchor's reference map
   (2021 capture) come from Esri's Wayback archive. Nothing showing those pixels -- the films, their stills, the
   texture tiles, screenshots of the dashboard's map -- goes into a paper, a public deck, or git. `.gitignore`
   keeps all of it out; keep it that way. Never texture the floor with the same image GeoAnchor matches against.
2. **One Gazebo simulation at a time.** Two PX4 + Gazebo instances on this laptop dragged the sim to 0.23x real
   time and broke PX4's position estimate (27 Sept). Scripts with lanes (`N=`) take `N=1`.
3. **Check the machine before a long run**: `free -g`, `nvidia-smi`, `df -h`, and AC power
   (`cat /sys/class/power_supply/A*/online` = 1). The 8 GB GPU fits one RL training (~2 GB) or one headroom ladder
   (~3.3 GB) next to a Gazebo flight, never more.

---

## 1. Install (once)

This repo drives three other things by absolute path. If you clone it anywhere other than
`/home/sai/RL_projects/rl-landing`, change `R=` in `gnss_denied/geofly.sh`, `gnss_denied/fly_mission.sh`, `shared/px4_gazebo/launch_px4.sh`,
`shared/px4_gazebo/sitl.sh` and the queue scripts, and the PX4 path (`P=`/`PX4=`) in `shared/px4_gazebo/launch_px4.sh`, `shared/px4_gazebo/gz_rig.py`, `shared/px4_gazebo/gz_unified.py`.

| what | where it is expected | used for |
|---|---|---|
| Ubuntu 24.04, NVIDIA GPU | | everything |
| ROS 2 Jazzy (`/opt/ros/jazzy`) and Gazebo Harmonic (gz-sim 8) | system | Gazebo, the `gz` CLI, ROS 2 for the AGP bridge |
| PX4-Autopilot **v1.16.2**, built with `make px4_sitl` | `/home/sai/thesis_2.0/PX4-Autopilot` | the flight software (SITL) |
| Micro XRCE-DDS Agent (`MicroXRCEAgent` on the PATH) | system | PX4 <-> ROS 2 (GeoAnchor's fixes go in as AGP) |
| GeoAnchor (`geoanchor-rt`, with its `.venv`) | `/home/sai/GeoAnchor/geoanchor-rt` | the visual position fix; the Wayback fetcher |
| `px4_msgs` built in a ROS 2 workspace | `/home/sai/GeoAnchor/ros2_ws` | `gnss_denied/agp_sim_bridge.py` |
| system Python (`/usr/bin/python3`): `pymavlink`, `opencv-python`, `numpy`, `python3-gz-transport13`, GDAL, `pyulog`, `playwright` | system / `pip install --user` | flights, films, maps, the dashboard server |
| conda env `mujoco_drones` (Python 3.11: `torch`, `jax[cuda]`, `mujoco`, `mujoco-mjx`, `scipy`, `stable-baselines3`, `gymnasium`) | `/home/sai/miniconda3/envs/mujoco_drones` | deciders, all RL training, the MJX wind twin |
| Ultralytics YOLO and torchvision, `pip install --no-deps --target .yolo_pkgs ultralytics torchvision polars` | this folder | the magpie detector (`PYTHONPATH=.yolo_pkgs YOLO_AUTOINSTALL=false`) |
| Node 18+ | system | building the dashboard |

In the commands below `P` is the conda python: `P=/home/sai/miniconda3/envs/mujoco_drones/bin/python`, and
`GA=/home/sai/GeoAnchor/geoanchor-rt/.venv/bin/python` is GeoAnchor's.

```bash
cmake -S conditions/wind/gz_winddrag -B conditions/wind/gz_winddrag/build && cmake --build conditions/wind/gz_winddrag/build   # the wind plugin
cd dashboard && npm ci && cd ..                                                                  # dashboard packages
```

## 2. Data and worlds (once, about an hour)

| data | how | ends up in |
|---|---|---|
| Logan 2023 LiDAR survey (QLD Government, ELVIS; 25 tiles, `shared/fast_sim/DATA_2462013.zip`, 15.6 GB) | order the Logan tiles from <https://elevation.fsdf.org.au>, save the zip in this folder | read in place by `shared/fast_sim/lidar_map.py` |
| OpenStreetMap roads/land use and Microsoft building footprints | already here: `shared/fast_sim/logan_osm.json`, `shared/fast_sim/ms_logan.geojson` | |
| Esri Wayback captures of the delivery area (2023-06-29 floor, 2021-07-21 map) | GeoAnchor's `scripts/fetch_wayback_tile.py` (its docstring; measurement use only) | `shared/px4_gazebo/gz_logan/imagery/ref_tile_2023-06-29.tif`, `ref_tile_2021-07-21.tif` |
| GeoAnchor's calibration results (part 1 only) | GeoAnchor's own benchmark | `~/GeoAnchor/results/s8_train` |

```bash
cd shared/fast_sim
/usr/bin/python3 lidar_map.py              # the survey on the Logan map's 1 m grid -> lidar_logan.npz (~20 min)
/usr/bin/python3 logan_map_lidar.py        # the map the fast simulator flies: houses, trees, pads -> logan_map_lidar.npz
/usr/bin/python3 gz_logan.py               # the Logan world for Gazebo (houses, trees, roads, 16 pads) -> ../px4_gazebo/gz_logan/
cd ../..
/usr/bin/python3 shared/px4_gazebo/make_pad.py               # our pad: three markers (ids 2, 1, 0) on a 5 m mat, into PX4's model tree
/usr/bin/python3 shared/px4_gazebo/gz_voxel.py --setup       # stage 3's crane, lights, traffic light, trees, shed + the depth drone
/usr/bin/python3 shared/px4_gazebo/gz_rig.py --setup         # the camera-rig drone and the magpie bench
/usr/bin/python3 shared/px4_gazebo/gz_unified.py --setup     # rig v2 and its variants (v2l = + the up camera) + their worlds
/usr/bin/python3 shared/px4_gazebo/gz_learner.py --setup     # the magpie world (shared/px4_gazebo/gz_logan/birds)
$GA gnss_denied/geo_world.py                       # the photo-floor worlds (part 5): geo_floor, geo_clean, geo_magpie
$GA gnss_denied/geo_world.py --variant v2l         # geo_magpie flown by the rig with the up camera
/usr/bin/python3 shared/px4_gazebo/calib_cam.py              # the landing camera's model -> shared/px4_gazebo/cam_B.npy (already in git)
```

Small checkpoints the films use are in git: `conditions/wind/runs/fly_hp_s0_best.npz` (flying learner),
`conditions/wind/runs/stage2_c100_s0_best.npz` (crash-priced decider), `shared/fast_sim/rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt`
(learner trained with birds), `shared/fast_sim/clone_corridor.pt`, and the magpie detectors `shared/px4_gazebo/gz_logan/rig_v2/yolo/run/weights/best.pt`
(plain ground) and `shared/px4_gazebo/gz_logan/rig_v2/yolo/run_geo/weights/best.pt` (retrained on photo ground: use this one).
Everything else section 3 regenerates.

---
## 3. Start here

### Prove the stack is alive (10 min)


```bash
python3 gnss_denied/geo_cruise.py --check                                          # the route follower (no sim)
$P shared/fast_sim/local_planner.py                                                  # the 3D planner's self-checks
$P conditions/wind/x500_mjx.py --check                                                # the MJX drone vs PX4's step responses
OUT=smoke WORLD=geo_clean INSTANCE=5 gnss_denied/geofly.sh /usr/bin/python3 -u gnss_denied/geo_cruise.py   # one GPS-denied cruise
tail -3 outputs/gazebo_geo/smoke/run.log                                           # expect "GNSS denied", a geo {...} line
```

### Run one condition, all of them, or a mix

Pick a folder from the table at the top; its README is the runbook. `all_conditions/` flies everything at once,
`mix_and_match/fly.py` any combination you choose (it refuses combinations the code cannot do, and `--dry-run` prints
the commands without starting anything).

### Watch training in ClearML

Both trainers log live to ClearML (hosted app.clear.ml; credentials in `~/clearml.conf`, never in git):

| project | what |
|---|---|
| `rl-landing/wind` | every `conditions/wind/ppo.py` training run, one task per run: outcomes, return, curriculum, PPO losses, validation landed per wind, speed; the best checkpoint as an artifact; the rule-vs-learner videos |
| `rl-landing/fast_sim` | every `shared/fast_sim/urban_rl.py` seed: DAgger rounds, clean deliveries, rewards, PPO losses, validation |
| `rl-landing/wind/history`, `rl-landing/fast_sim/history` | the training logs from before ClearML, imported by `shared/fast_sim/clearml_import.py` |

Open a run, then the SCALARS tab; tick several runs and press Compare to overlay them. `--no-clearml` trains without
it. `python shared/fast_sim/clearml_import.py --dry-run` must report 0 to import once the history is in.

### Research reports

`papers/research/` holds four reports, each checked source by source: the nearest published architectures to ours,
Aerial Gym against our simulators, how much wind the x500 can take, and the algorithm options for the wind learner.

---

## 4. The dashboard: flight logs and missions


`dashboard/` is a copy of GeoAnchor's dashboard (the original in `geoanchor-rt/dashboard` is untouched) with two
new pages and its own small server:

* **Flight log** -- any flight in `outputs/gazebo_geo/`, QGC-style: roll and pitch with their setpoints, heading, height
  (PX4 against Gazebo's truth), cruise and descent speed, PX4's position error, tilt, and a strip of what EKF2 was
  fusing (GNSS, GeoAnchor/AGP, rangefinder, dead reckoning), phases shaded; the track on the map with every fix.
  Hover to move one cursor across all graphs and the map; drag to zoom. **Live** follows the flight in the air.
* **Mission planner** -- the sixteen pads with their marker ids on the map; click to add waypoints, drag to move,
  right-click to delete; pick the pad (any of the sixteen), cruise height, speed, lander, wind. **Save** writes
  `outputs/gnss_denied/missions/<name>.json`; **Fly it in Gazebo** starts `gnss_denied/fly_mission.sh` (refused while Gazebo is up).

```bash
cd dashboard && NEXT_TELEMETRY_DISABLED=1 nice -n 19 npx next build && cd ..   # after any change to the pages
python3 dashboard/flight_server.py        # -> http://localhost:8010  (GeoAnchor | Flight log | Mission planner)
gnss_denied/fly_mission.sh outputs/gnss_denied/missions/example_pad4.json mission_example_pad4        # a mission by hand
```

`GEO_MISSION=<file>` is all the simulator needs: `gnss_denied/geo_cruise.py` takes the route, height and speed from it,
`shared/px4_gazebo/px4_mission.py` the pad's place and marker ids (the landing camera then refuses our own pad's ids like any
neighbour's), `conditions/wind/gz_windland.py` the pad. `geofly.sh` copies PX4's `.ulg` into every flight folder.

---

## 5. Reading a result

Each GeoAnchor flight leaves in `outputs/gazebo_geo/<OUT>/`: `run.log` (`GNSS denied at ...`, a `geo {...}` report, the
landing line), the film (`.mp4`, 1600x900), PX4's own log (`*.ulg`), the record (`*_geo.json` / `geo_*.json`: every
step's PX4 estimate against Gazebo's truth, and every fix), `agp.jsonl` (what EKF2 fused, when), `live.jsonl`.

The `geo` line: `denied_s` (seconds with GNSS off), `fix_err_m` (GeoAnchor's fixes against the truth), `est_err_m`
(PX4's estimate against the truth after the denial), `gnss_restored_at` (null = GeoAnchor held all the way),
`rtf` (sim speed / real time). Report medians and tails (p90, p99, % within 5/10/20 m) -- never the mean or RMSE.

The 27 Sept flights (`gnss_denied/results/geo_flights_2026-09-27.json`; one flight each: a demonstration, not a rate):

| flight | lander | touchdown from centre | GNSS off | PX4 error (median, GNSS off) | sim speed |
|---|---|---|---|---|---|
| calm | rule | 0.06 m | 110 s | 2.3 m | 0.84x |
| wind 9 m/s | flying learner | 0.19 m | 117 s | 1.7 m | 0.76x |
| obstacles (1st try) | rule + planner | crashed into the crane (planner too slow; fixed) | | | 0.79x |
| obstacles | rule + planner | 0.04 m, crane passed at 6.7 m | 117 s | 1.9 m | 0.76x |
| magpie, rig v2 (old detector) | learner with birds | 0.10 m, 3 swoops, closest 1.9 m | 116 s | 1.9 m | 0.25x |
| magpie, rig v2 (retrained detector) | learner with birds | 0.04 m, 3 swoops, closest 2.0 m | 116 s | 1.7 m | 0.26x |
| magpie, rig v2l (+ up camera) | learner with birds | 0.08 m, 3 swoops, closest 1.9 m | 116 s | 1.4 m | 0.27x |
| mission to pad 4 (dashboard) | rule | 0.05 m from pad 4's centre | 107 s | | 0.89x |

The magpie detector retrained on photo ground: held-out pictures, threshold 0.5 -- found 75% -> 81%, false boxes
0.57 -> 0.002 per picture; plain ground unchanged (89%); the up camera's pictures 90% found. In flight the tracker's
phantom alarms fell from 54% to 18% (11% with the up camera), and the up camera doubled how often a magpie above
the drone was truly seen (20% -> 40%).

## 6. Self-checks (each fails loudly if its logic breaks)

```bash
python3 gnss_denied/geo_cruise.py --check                 # the route follower: corners flown through, turned the short way
$P shared/fast_sim/local_planner.py                         # the 3D planner passes a mast, flies a free line straight
python3 dashboard/flight_server.py --check        # PX4's log and the GeoAnchor record line up on one clock
$P conditions/wind/x500_mjx.py --check                       # the MJX drone against PX4's step responses
```

## 7. Things that bit us (read before changing the flight code)

* The world's GPS origin must be the Logan depot, or PX4 rejects every GeoAnchor fix.
* gz-sim 8's magnetometer ignores the world's field once spherical coordinates are set: PX4's heading is 10-20 deg
  off on the ground. So GNSS is denied only after flying on it (PX4 fixes its heading from motion).
* Without GNSS the barometer drifts 2-5 m in a descent: the drone carries a rangefinder (`x500_unified*_rng`).
* Keep the marker's fixes going to touchdown, with the jump check in `gnss_denied/geo_cruise.py` (`MarkerFix`).
* Ask the local planner without waiting (`gz_learner.Decider`), and keep it fast: at 2 s per plan the drone met
  the crane before the answer (27 Sept); weighted A* made it 0.1 s. Offline checks must include that delay.
* Never zero the velocity command mid-cruise: the old GNSS switch-off did, and PX4 braked at 30 deg of tilt.
* A detector trained on plain Gazebo ground fires on tree crowns over the photo floor: train on the ground you fly.
* Frames: measure them (`gnss_denied/frame_check.py`), never derive them.
* Stop simulators by PID or by their world file, never `pkill -f "gz sim"` (it kills the other partition's bench).
* A stage-2 wind checkpoint names its frozen flyer by absolute path; after the 5 Oct reorganization the loaders fall
  back to `conditions/wind/runs/` (re-scored afterwards: identical ladder).
* ClearML drops scalars sent to a task created with `Task.create` until it is marked started (the first history import
  came out empty). Check a run's graphs after any new logging code.

## Layout

```
conditions/        one folder per condition: wind, magpie, dynamic_obstacles, static_obstacles, weather (scripts, queue/ chains, results/)
all_conditions/    the stress test and the chains that mix conditions
mix_and_match/     fly.py: any combination of conditions for one Gazebo flight
gnss_denied/       GeoAnchor's loop: worlds, cruise, films, missions
shared/fast_sim/   the last-mile fast simulator, maps, deciders, RL, the local 3D planner (run its scripts from this folder)
shared/px4_gazebo/ PX4 launch, missions, perception, the Gazebo landers and the rig; gz_logan/ = generated worlds and models
dashboard/         the forked dashboard + flight_server.py
papers/            acra2026 (frozen), research/ (four checked reports), slides/ (the deck's source), docs/, report/, paper/, deck/
archive/           sigma/ (part 1), part2/, tello/ (the Tello cage)
outputs/           everything a run writes: logs, films, run folders (never in git)
```
