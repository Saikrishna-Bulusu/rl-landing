# rl-landing

Landing a delivery drone on the right pad in a Logan (QLD) suburb, with GPS denied. PX4 flies in Gazebo; GeoAnchor
(a separate project) gives it an absolute position from a downward camera; the pad's markers take over for the last
30 m; a rule, a local 3D planner and reinforcement-learned policies decide the landing past cranes, wind and a magpie.

The study grew in five parts. Each part's code is still here and still runs:

| part | question | where |
|---|---|---|
| 1. sigma | does GeoAnchor's own uncertainty help a landing policy? (it does not; staleness hurts) | `approach_env*.py`, `train*.py`, `baselines.py`, `calibrate.py` |
| 2. obstacles | trees, a bigger marker, the forest study on real PX4 | `px4_mission.py`, `make_pad.py`, `forest*.py`, `bigger_marker.py` |
| 3. last mile | Logan under CASA rules: headroom, a learner that beats the planner, birds | `urban/` |
| 4. Gazebo | PX4 with real pictures: a crane, the voxel map, a 360 camera rig, the magpie | `gz_learner.py`, `gz_voxel.py`, `gz_rig.py`, `gz_unified.py` |
| 5. wind + GeoAnchor | landing in gusts (MuJoCo MJX twin of PX4); whole deliveries with GPS denied | `wind/`, `geo/` |

The slide deck (all five parts, with the films) is a private Claude artifact:
<https://claude.ai/artifact/QaYeAMh7cZThMmffKxjjNA>. Its source is in `slides/` (one HTML file per slide, order in
`slides/deck.json`); the charts are in `docs/figures/`. The films are **not** in git (see rule 1).

**How to use this README.** Sections 1-2 set up a machine once. Section 3 is the whole study as a runbook, part by
part, in the order it was run; each step says what it produces and roughly how long it takes on the Legion (RTX 5070
Laptop 8 GB, 32 threads, 23 GB RAM). Each part needs only sections 1-2, not the parts before it, unless a step says so.

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
`/home/sai/RL_projects/rl-landing`, change `R=` in `geo/geofly.sh`, `geo/fly_mission.sh`, `launch_px4.sh`,
`wind/sitl.sh` and the queue scripts, and the PX4 path (`P=`/`PX4=`) in `launch_px4.sh`, `gz_rig.py`, `gz_unified.py`.

| what | where it is expected | used for |
|---|---|---|
| Ubuntu 24.04, NVIDIA GPU | | everything |
| ROS 2 Jazzy (`/opt/ros/jazzy`) and Gazebo Harmonic (gz-sim 8) | system | Gazebo, the `gz` CLI, ROS 2 for the AGP bridge |
| PX4-Autopilot **v1.16.2**, built with `make px4_sitl` | `/home/sai/thesis_2.0/PX4-Autopilot` | the flight software (SITL) |
| Micro XRCE-DDS Agent (`MicroXRCEAgent` on the PATH) | system | PX4 <-> ROS 2 (GeoAnchor's fixes go in as AGP) |
| GeoAnchor (`geoanchor-rt`, with its `.venv`) | `/home/sai/GeoAnchor/geoanchor-rt` | the visual position fix; the Wayback fetcher |
| `px4_msgs` built in a ROS 2 workspace | `/home/sai/GeoAnchor/ros2_ws` | `geo/agp_sim_bridge.py` |
| system Python (`/usr/bin/python3`): `pymavlink`, `opencv-python`, `numpy`, `python3-gz-transport13`, GDAL, `pyulog`, `playwright` | system / `pip install --user` | flights, films, maps, the dashboard server |
| conda env `mujoco_drones` (Python 3.11: `torch`, `jax[cuda]`, `mujoco`, `mujoco-mjx`, `scipy`, `stable-baselines3`, `gymnasium`) | `/home/sai/miniconda3/envs/mujoco_drones` | deciders, all RL training, the MJX wind twin |
| Ultralytics YOLO and torchvision, `pip install --no-deps --target .yolo_pkgs ultralytics torchvision polars` | this folder | the magpie detector (`PYTHONPATH=.yolo_pkgs YOLO_AUTOINSTALL=false`) |
| Node 18+ | system | building the dashboard |

In the commands below `P` is the conda python: `P=/home/sai/miniconda3/envs/mujoco_drones/bin/python`, and
`GA=/home/sai/GeoAnchor/geoanchor-rt/.venv/bin/python` is GeoAnchor's.

```bash
cmake -S wind/gz_winddrag -B wind/gz_winddrag/build && cmake --build wind/gz_winddrag/build   # the wind plugin
cd dashboard && npm ci && cd ..                                                                  # dashboard packages
```

## 2. Data and worlds (once, about an hour)

| data | how | ends up in |
|---|---|---|
| Logan 2023 LiDAR survey (QLD Government, ELVIS; 25 tiles, `DATA_2462013.zip`, 15.6 GB) | order the Logan tiles from <https://elevation.fsdf.org.au>, save the zip in this folder | read in place by `urban/lidar_map.py` |
| OpenStreetMap roads/land use and Microsoft building footprints | already here: `urban/logan_osm.json`, `urban/ms_logan.geojson` | |
| Esri Wayback captures of the delivery area (2023-06-29 floor, 2021-07-21 map) | GeoAnchor's `scripts/fetch_wayback_tile.py` (its docstring; measurement use only) | `gz_logan/imagery/ref_tile_2023-06-29.tif`, `ref_tile_2021-07-21.tif` |
| GeoAnchor's calibration results (part 1 only) | GeoAnchor's own benchmark | `~/GeoAnchor/results/s8_train` |

```bash
cd urban
/usr/bin/python3 lidar_map.py              # the survey on the Logan map's 1 m grid -> lidar_logan.npz (~20 min)
/usr/bin/python3 logan_map_lidar.py        # the map the fast simulator flies: houses, trees, pads -> logan_map_lidar.npz
/usr/bin/python3 gz_logan.py               # the Logan world for Gazebo (houses, trees, roads, 16 pads) -> ../gz_logan/
cd ..
/usr/bin/python3 make_pad.py               # our pad: three markers (ids 2, 1, 0) on a 5 m mat, into PX4's model tree
/usr/bin/python3 gz_voxel.py --setup       # stage 3's crane, lights, traffic light, trees, shed + the depth drone
/usr/bin/python3 gz_rig.py --setup         # the camera-rig drone and the magpie bench
/usr/bin/python3 gz_unified.py --setup     # rig v2 and its variants (v2l = + the up camera) + their worlds
/usr/bin/python3 gz_learner.py --setup     # the magpie world (gz_logan/birds)
$GA geo/geo_world.py                       # the photo-floor worlds (part 5): geo_floor, geo_clean, geo_magpie
$GA geo/geo_world.py --variant v2l         # geo_magpie flown by the rig with the up camera
/usr/bin/python3 calib_cam.py              # the landing camera's model -> cam_B.npy (already in git)
```

Small checkpoints the films use are in git: `wind/runs/fly_hp_s0_best.npz` (flying learner),
`wind/runs/stage2_c100_s0_best.npz` (crash-priced decider), `urban/rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt`
(learner trained with birds), `urban/clone_corridor.pt`, and the magpie detectors `gz_logan/rig_v2/yolo/run/weights/best.pt`
(plain ground) and `gz_logan/rig_v2/yolo/run_geo/weights/best.pt` (retrained on photo ground: use this one).
Everything else section 3 regenerates.

---

## 3. Run the whole study, step by step

### Before any part: prove the stack is alive (10 min)

```bash
python3 geo/geo_cruise.py --check                                          # the route follower (no sim)
$P urban/local_planner.py                                                  # the 3D planner's self-checks
$P wind/x500_mjx.py --check                                                # the MJX drone vs PX4's step responses
OUT=smoke WORLD=geo_clean INSTANCE=5 geo/geofly.sh /usr/bin/python3 -u geo/geo_cruise.py   # one GPS-denied cruise
tail -3 gazebo_geo/smoke/run.log                                           # expect "GNSS denied", a geo {...} line
```

### Part 1 -- sigma (the fast point-mass study; `HANDOFF.md` has the full story and its run log)

| step | command | produces, time |
|---|---|---|
| 1. calibrate sigma on GeoAnchor's results | `python3 calibrate.py ~/GeoAnchor/results/s8_train --exclude-scene Scene_21` | `sigma_pool.npz`, ENCE, conformal coverage; 1 min |
| 2. the scripted baselines (the decision gate) | `python3 baselines.py --pool sigma_pool.npz --tune --episodes 400` | `baselines.json`; 10 min |
| 3. train the three arms (none / staleness / sigma) | `$P train.py --pool sigma_pool.npz --all --steps 2000000 --n-envs 8` | `runs/ablation.json`; 30-60 min per arm |
| 4. the powered ablation (trees, 8+ seeds, resumable) | `bash run_overnight.sh` (runs `train_v5.py` per arm and seed; `watchdog.sh` guards memory) | `<run>/*/ablation.json`; overnight |
| 5. the same missions on real PX4 | `./launch_px4.sh` in one terminal, then `python3 px4_mission.py --runs 5` | landings per sigma; ~5 min per mission |

Compare `runs/ablation.json` with `baselines.json`; the claim lives in the high-sigma column.

### Part 2 -- the marker and the forest, on real PX4

| step | command | produces, time |
|---|---|---|
| 1. how big must the marker be (fast study) | `python3 bigger_marker.py` | delivery vs marker size; 5 min |
| 2. old pad vs new pad, same five missions | `./ab_pad.sh old`, then `./ab_pad.sh new` | the A/B table; 30 min per arm |
| 3. twenty different forests, both pads | `./run_forest20.sh` (switches the pad itself between arms) | `forest12_*.json`, delivered/crashed per pad; 3 h |
| 4. does a better matcher deliver more? | `./gazebo_matchers.sh` | matched-seed landings per matcher; 3 h |

### Part 3 -- the last mile in Logan (fast simulator; run from `urban/`)

| step | command | produces, time |
|---|---|---|
| 1. headroom: how far is the rule from a perfect decider | `$P urban_headroom.py --map logan_map_lidar.npz --nav gps` (add `--birds --hunter` for the magpie; `--rig --tag _v2l` with `RIG_CALIB=rig_calib_v2l.json` for the camera rig) | `headroom_*.json`; 30-60 min per ladder |
| 2. train the learner (copies the planner, then PPO) | `$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --seeds 0 1 2` (same extra flags as step 1) | `rl_corridor_*_s<k>_mean.pt`, `rl_*.json`; 2-3 h per seed |
| 3. can the network hold the planner at all | `$P clone_check.py` | `clone_check.json` |
| 4. the figures | `$P paper_figs.py world headroom learning birds rig gazebo` | `../paper/figs/*.pdf` |

The chains that ran these for the deck are `overnight_24sep.sh`, `rig_queue.sh`, `rig_train.sh`, `rig_after.sh`:
their headers say which ladders and trainings, in what order, and why.

### Part 4 -- PX4 in Gazebo: the crane and the magpie

The crane (`voxfly.sh` starts PX4 + Gazebo per run; RUNS are decider:map):

```bash
OUT=first SEEDS="0 1 2" RUNS="planner:survey rule:voxel rule:local3d" ./voxfly.sh   # survey vs voxel map vs 3D planner
```

The camera rig, measured before it is trusted (the bench is a static drone in Gazebo, in its own partition):

```bash
RIG=v2l /usr/bin/python3 gz_rig.py --bench                       # headless bench; prints its PID
RIG=v2l /usr/bin/python3 gz_rig.py --cover gz_logan/rig_v2l/cover        # which camera sees the magpie where (~15 min)
RIG=v2l /usr/bin/python3 gz_rig.py --data gz_logan/rig_v2l/data --poses 600    # labelled pictures (~25 min)
kill <bench PID>
RIG=v2l PYTHONPATH=.yolo_pkgs $P rig_yolo.py --data gz_logan/rig_v2l/data
RIG=v2l PYTHONPATH=.yolo_pkgs $P rig_yolo.py --train --epochs 30 --init $PWD/gz_logan/rig_v2/yolo/run/weights/best.pt
RIG=v2l PYTHONPATH=.yolo_pkgs $P rig_yolo.py --score gz_logan/rig_v2l/data        # found / false alarms by size
RIG=v2l /usr/bin/python3 gz_rig.py --calib gz_logan/rig_v2l/cover gz_logan/rig_v2l/data --score score_yolo.json \
    --out urban/rig_calib_v2l.json                                # the calibration the fast simulator (part 3) flies
```

Retrain the detector on photo ground (what fixed "trees seen as magpies"; about 2 h, one command):

```bash
queue/yolo_photo.sh    # photo bench -> 800 places -> fine-tune on plain + photo -> old vs new in queue/yolo_photo.log
```

The magpie flights on the plain Logan world (`SENSE=rig`: the decider knows the bird only through the rig):

```bash
OUT=planner_rig SEEDS="0 1 2 3" DECIDERS=planner SENSE=rig RIG=v2l ./rigfly.sh
OUT=learner_rig SEEDS="0 1 2 3" DECIDERS=learner SENSE=rig RIG=v2l \
  CKPT=rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt ./rigfly.sh
```

### Part 5a -- wind (MuJoCo MJX twin of PX4; details in `wind/README.md`; run from `wind/`)

| step | command | produces, time |
|---|---|---|
| 1. the rule's ladder (1,024 paired flights per wind) | `$P ppo.py --ladder rule` | `runs/rule_ladder.json`; 5 min |
| 2. the flying learner (8 seeds) | `$P ppo.py --tag fly_hp --seed 0 --no-decide --hp-tau 4 --iters 1000 --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100` | `runs/fly_hp_s0_best.npz`; 45 min |
| 3. the crash-priced decider on a frozen flyer (3 seeds) | `$P ppo.py --tag stage2_c100 --seed 0 --flyer runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -30 --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100` | `runs/stage2_c100_s0_best.npz`; 30 min |
| 4. each checkpoint's ladder | `$P ppo.py --ladder runs/fly_hp_s0_best.npz` | `runs/*_best_ladder.json` |
| 5. the same in Gazebo (from the repo root) | `N=1 EPISODES="$(seq 0 19)" WINDS="18" DECIDERS="rule wind/runs/stage2_c100_s0_best.npz" OUT=ladder ./wind/gz_ladder.sh` | `gazebo_wind/ladder/flights.jsonl`; ~6 min a flight |
| 6. the charts | `python3 geo/deck_figs.py` (from the repo root) | `gazebo_geo/deck/wind_*.png` |

`queue/overnight_27sep.sh` is the chain that trained every seed and variant in the deck.

### Part 5b -- whole deliveries with GPS denied (GeoAnchor in PX4's loop)

`geo/geofly.sh` starts PX4 + Gazebo on a photo-floor world, GeoAnchor's own three layers on the live gimbal camera,
the uXRCE-DDS agent and `geo/agp_sim_bridge.py` (GeoAnchor's accepted fixes -> PX4's EKF2 as auxiliary position),
then runs one flight command and cleans everything up. It never uses GeoAnchor's `run.sh` (that syncs every run to
the logs repo); `GEOANCHOR_SYNC_ON_EXIT=0` is set for you.

Every flight: climbs on GPS to 75 m at the depot; flies the route at 7 m/s on PX4's own estimate (corners flown
through, the nose turned the short way); over the suburb, once GeoAnchor's fixes are arriving and PX4 is fusing them,
**GNSS is switched off** (`EKF2_GPS_CTRL 0`) for the rest of the flight; 50 m before the pad it comes down to 55 m,
settles, drops to 30 m where the pad's 1.80 m marker reads, and a lander takes over. Every marker read is also a
position fix for PX4 (GeoAnchor fades below 50 m). A rangefinder gives the height.

1. Does GeoAnchor match this floor at all (a 75 m sweep, replayed through GeoAnchor; ~20 min):

```bash
OUT=geo_rec_floor WORLD=geo_floor INSTANCE=5 ./wind/sitl.sh /usr/bin/python3 -u geo/record_sweep.py --out gazebo_geo/floor
$GA geo/frame_check.py gazebo_geo/floor                               # the image-top heading, measured
geo/replay.sh gazebo_geo/floor floor                                   # GeoAnchor's own layers on the recording
python3 geo/score.py geo/runs/<run> gazebo_geo/floor/flight_gt.jsonl   # fixes against the truth (median, p90, %<10 m)
OUT=geo_readh WORLD=geo_floor INSTANCE=5 ./wind/sitl.sh /usr/bin/python3 -u geo/read_height.py   # marker read by height
```

2. The four films (one at a time; outputs in `gazebo_geo/<OUT>/`; 5-10 min each, the magpie ~12 min):

```bash
# calm air, the rule lands it
OUT=geo_calm4 WORLD=geo_clean INSTANCE=5 geo/geofly.sh /usr/bin/python3 -u wind/gz_windland.py --geo \
  --episode 0 --w10 0 --decider rule --out gazebo_geo/geo_calm4/flights.jsonl --video gazebo_geo/geo_calm4/calm.mp4

# 9 m/s wind at 10 m (WindDrag plugin), the flying learner lands it
OUT=geo_wind10 WORLD=geo_clean SERVER_CONFIG=gz_logan/geo_clean/geo_w9.config INSTANCE=5 geo/geofly.sh \
  /usr/bin/python3 -u wind/gz_windland.py --geo --episode 0 --w10 9 --decider wind/runs/fly_hp_s0_best.npz \
  --out gazebo_geo/geo_wind10/flights.jsonl --video gazebo_geo/geo_wind10/wind9.mp4

# an unmapped 58 m crane on the way in: hand-over 90 m out at 75 m, rule + local 3D planner on the voxel map
CRANE_AT=147.33,-23.43 GEO_GOAL_OUT=90 OUT=geo_obst5 WORLD=geo_floor INSTANCE=5 geo/geofly.sh /usr/bin/python3 -u \
  gz_learner.py --geo --decider rule --seed 1 --magpie none --voxel --local3d --video gazebo_geo/geo_obst5/obstacles.mp4

# a hunter magpie, seen only through rig v2l (with the up camera) and the retrained detector; the learner trained
# with birds lands it (its landing runs at 0.1x speed, for the detector)
TREES_JSON=gz_logan/trees_logan.json RIG=v2l RIG_YOLO=gz_logan/rig_v2/yolo/run_geo/weights/best.pt \
  RIG_CALIB=urban/rig_calib_v2l.json GEO_LAND_SPEED=0.1 OUT=geo_magpie6 WORLD=geo_magpie_v2l INSTANCE=5 geo/geofly.sh \
  /usr/bin/python3 -u gz_learner.py --geo --decider learner --seed 0 --magpie hunter --sense rig \
  --ckpt rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt --video gazebo_geo/geo_magpie6/magpie.mp4
```

3. Deck material (internal only):

```bash
python3 geo/to_webm.py gazebo_geo/geo_calm4/calm.mp4 gazebo_geo/deck/calm4.webm   # VP8 clip + still
python3 geo/deck_figs.py                                                            # charts -> gazebo_geo/deck/*.png
```

### Part 5c -- the dashboard: flight logs and missions

`dashboard/` is a copy of GeoAnchor's dashboard (the original in `geoanchor-rt/dashboard` is untouched) with two
new pages and its own small server:

* **Flight log** -- any flight in `gazebo_geo/`, QGC-style: roll and pitch with their setpoints, heading, height
  (PX4 against Gazebo's truth), cruise and descent speed, PX4's position error, tilt, and a strip of what EKF2 was
  fusing (GNSS, GeoAnchor/AGP, rangefinder, dead reckoning), phases shaded; the track on the map with every fix.
  Hover to move one cursor across all graphs and the map; drag to zoom. **Live** follows the flight in the air.
* **Mission planner** -- the sixteen pads with their marker ids on the map; click to add waypoints, drag to move,
  right-click to delete; pick the pad (any of the sixteen), cruise height, speed, lander, wind. **Save** writes
  `geo/missions/<name>.json`; **Fly it in Gazebo** starts `geo/fly_mission.sh` (refused while Gazebo is up).

```bash
cd dashboard && NEXT_TELEMETRY_DISABLED=1 nice -n 19 npx next build && cd ..   # after any change to the pages
python3 dashboard/flight_server.py        # -> http://localhost:8010  (GeoAnchor | Flight log | Mission planner)
geo/fly_mission.sh geo/missions/example_pad4.json mission_example_pad4        # a mission by hand
```

`GEO_MISSION=<file>` is all the simulator needs: `geo/geo_cruise.py` takes the route, height and speed from it,
`px4_mission.py` the pad's place and marker ids (the landing camera then refuses our own pad's ids like any
neighbour's), `wind/gz_windland.py` the pad. `geofly.sh` copies PX4's `.ulg` into every flight folder.

---

## 4. Reading a result

Each GeoAnchor flight leaves in `gazebo_geo/<OUT>/`: `run.log` (`GNSS denied at ...`, a `geo {...}` report, the
landing line), the film (`.mp4`, 1600x900), PX4's own log (`*.ulg`), the record (`*_geo.json` / `geo_*.json`: every
step's PX4 estimate against Gazebo's truth, and every fix), `agp.jsonl` (what EKF2 fused, when), `live.jsonl`.

The `geo` line: `denied_s` (seconds with GNSS off), `fix_err_m` (GeoAnchor's fixes against the truth), `est_err_m`
(PX4's estimate against the truth after the denial), `gnss_restored_at` (null = GeoAnchor held all the way),
`rtf` (sim speed / real time). Report medians and tails (p90, p99, % within 5/10/20 m) -- never the mean or RMSE.

The 27 Sept flights (`results/geo_flights_2026-09-27.json`; one flight each: a demonstration, not a rate):

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

## 5. Self-checks (each fails loudly if its logic breaks)

```bash
python3 geo/geo_cruise.py --check                 # the route follower: corners flown through, turned the short way
$P urban/local_planner.py                         # the 3D planner passes a mast, flies a free line straight
python3 dashboard/flight_server.py --check        # PX4's log and the GeoAnchor record line up on one clock
$P wind/x500_mjx.py --check                       # the MJX drone against PX4's step responses
```

## 6. Things that bit us (read before changing the flight code)

* The world's GPS origin must be the Logan depot, or PX4 rejects every GeoAnchor fix.
* gz-sim 8's magnetometer ignores the world's field once spherical coordinates are set: PX4's heading is 10-20 deg
  off on the ground. So GNSS is denied only after flying on it (PX4 fixes its heading from motion).
* Without GNSS the barometer drifts 2-5 m in a descent: the drone carries a rangefinder (`x500_unified*_rng`).
* Keep the marker's fixes going to touchdown, with the jump check in `geo/geo_cruise.py` (`MarkerFix`).
* Ask the local planner without waiting (`gz_learner.Decider`), and keep it fast: at 2 s per plan the drone met
  the crane before the answer (27 Sept); weighted A* made it 0.1 s. Offline checks must include that delay.
* Never zero the velocity command mid-cruise: the old GNSS switch-off did, and PX4 braked at 30 deg of tilt.
* A detector trained on plain Gazebo ground fires on tree crowns over the photo floor: train on the ground you fly.
* Frames: measure them (`geo/frame_check.py`), never derive them.
* Stop simulators by PID or by their world file, never `pkill -f "gz sim"` (it kills the other partition's bench).

## Layout

```
geo/          part 5: worlds, the live GeoAnchor loop, cruise, films, deck figures, missions
wind/         the wind study (MJX twin, PPO, WindDrag Gazebo plugin, the Gazebo ladder)
urban/        the last-mile fast simulator, maps, deciders, RL, the local 3D planner
dashboard/    the forked dashboard + flight_server.py
queue/        the overnight chains that produced the deck's numbers (read their headers)
slides/       the deck's source (deck.json + one HTML per slide); docs/figures/ its charts
results/      small result summaries
gz_logan/     generated worlds and models (assets rebuilt by the scripts; not in git)
*.py (top)    parts 1-2 environments and training, PX4 missions, the pad, perception, the Gazebo landers and the rig
```
