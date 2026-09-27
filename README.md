# rl-landing

Landing a delivery drone on the right pad in a Logan (QLD) suburb, with GPS denied. PX4 flies in Gazebo; GeoAnchor
(a separate project) gives it an absolute position from a downward camera; the pad's markers take over for the last
30 m; a rule, a local 3D planner and reinforcement-learned policies decide the landing past cranes, wind and a magpie.

The study grew in five parts. Each part's code is still here and still runs:

| part | question | where |
|---|---|---|
| 1. sigma | does GeoAnchor's own uncertainty help a landing policy? (it does not; staleness hurts) | `approach_env*.py`, `baselines.py`, `calibrate.py`, `CALIBRATION.md` |
| 2. obstacles | trees, a bigger marker, the forest study on real PX4 | `px4_mission.py`, `make_pad.py`, `forest*.py`, `bigger_marker.py` |
| 3. last mile | Logan under CASA rules: headroom, a learner that beats the planner, birds | `urban/` |
| 4. Gazebo | PX4 with real pictures: a crane, the voxel map, a 360 camera rig, the magpie | `gz_learner.py`, `gz_voxel.py`, `gz_rig.py`, `gz_unified.py` |
| 5. wind + GeoAnchor | landing in gusts (MuJoCo MJX twin of PX4); whole deliveries with GPS denied | `wind/`, `geo/` |

The slide deck (51 slides, all five parts, with the films) is a private Claude artifact:
<https://claude.ai/artifact/QaYeAMh7cZThMmffKxjjNA>. Its source is in `slides/` (one HTML file per slide, order in
`slides/deck.json`); the charts are in `docs/figures/`. The films are **not** in git (see the imagery rule below).

---

## 0. Two rules before anything else

1. **Esri Wayback imagery is for measurement only.** The Gazebo floor (2023 capture) and GeoAnchor's reference map
   (2021 capture) come from Esri's Wayback archive. Nothing showing those pixels -- the films, their stills, the
   texture tiles, screenshots of the dashboard's map -- goes into a paper, a public deck, or git. `.gitignore`
   keeps all of it out; keep it that way. And never texture the floor with the same image GeoAnchor matches against.
2. **One simulation at a time.** Two PX4 + Gazebo instances on this laptop dragged the sim to 0.23x real time and
   broke PX4's position estimate (27 Sept). Every script here assumes it has the machine to itself.

---

## 1. What you need installed

This repo is not self-contained: it drives three other things by absolute path. If you clone it anywhere other
than `/home/sai/RL_projects/rl-landing`, change `R=` in `geo/geofly.sh`, `geo/fly_mission.sh` and
`launch_px4.sh`, and `P=` (PX4) in `launch_px4.sh` and `gz_rig.py`.

| what | where it is expected | used for |
|---|---|---|
| Ubuntu 24.04, NVIDIA GPU | | everything (the rendering and the RL use the GPU) |
| ROS 2 Jazzy (`/opt/ros/jazzy`) and Gazebo Harmonic (gz-sim 8) | system | Gazebo, `gz` CLI, ROS 2 for the AGP bridge |
| PX4-Autopilot **v1.16.2**, built with `make px4_sitl` | `/home/sai/thesis_2.0/PX4-Autopilot` | the flight software (SITL) |
| Micro XRCE-DDS Agent (`MicroXRCEAgent` on the PATH) | system | PX4 <-> ROS 2 (GeoAnchor's fixes go in as AGP) |
| GeoAnchor (`geoanchor-rt`, with its `.venv`) | `/home/sai/GeoAnchor/geoanchor-rt` | the visual position fix; also the Wayback fetcher |
| `px4_msgs` built in a ROS 2 workspace | `/home/sai/GeoAnchor/ros2_ws` | `geo/agp_sim_bridge.py` |
| system Python (`/usr/bin/python3`) with `pymavlink`, `opencv-python`, `numpy`, gz-transport 13 bindings (`python3-gz-transport13`), `pyulog`, `playwright` | system / `pip install --user` | the flights, the films, the dashboard server |
| conda env `mujoco_drones` (Python 3.11: `torch`, `jax[cuda]`, `mujoco`, `mujoco-mjx`, `scipy`) | `/home/sai/miniconda3/envs/mujoco_drones` | the deciders, all RL training, the MJX wind twin |
| Ultralytics YOLO, installed with `--no-deps` into `.yolo_pkgs/` | this folder | the magpie detector (`PYTHONPATH=.yolo_pkgs`) |
| Node 18+ | system | building the dashboard |

Build the wind plugin once (Gazebo loads it when a flight names a wind config):

```bash
cmake -S wind/gz_winddrag -B wind/gz_winddrag/build && cmake --build wind/gz_winddrag/build
```

## 2. Data you fetch yourself (not in git)

| data | how | ends up in |
|---|---|---|
| Logan 2023 LiDAR survey (QLD Government, ELVIS; 25 tiles, `DATA_2462013.zip`, 15.6 GB) | order the Logan tiles from <https://elevation.fsdf.org.au>, save the zip in this folder | read in place by `urban/lidar_map.py` |
| OpenStreetMap roads/land use and Microsoft building footprints | already here: `urban/logan_osm.json`, `urban/ms_logan.geojson` | |
| Esri Wayback captures of the delivery area (2023-06-29 floor, 2021-07-21 map) | GeoAnchor's `scripts/fetch_wayback_tile.py` (see its docstring; measurement use only) | `gz_logan/imagery/ref_tile_2023-06-29.tif`, `ref_tile_2021-07-21.tif` |

Then build the maps and worlds, in this order:

```bash
cd urban && /usr/bin/python3 lidar_map.py && /usr/bin/python3 logan_map_lidar.py   # -> urban/logan_map_lidar.npz
/usr/bin/python3 gz_logan.py && cd ..       # the Logan world for Gazebo: houses, trees, roads, 16 pads -> gz_logan/
/usr/bin/python3 make_pad.py                # our pad's texture: three markers (ids 2, 1, 0) on a 5 m mat
/usr/bin/python3 gz_rig.py --setup          # the drone models with the camera rig and the magpie bench
/home/sai/GeoAnchor/geoanchor-rt/.venv/bin/python geo/geo_world.py   # geo_floor / geo_clean / geo_magpie (photo floor)
/usr/bin/python3 calib_cam.py               # the landing camera's model -> cam_B.npy (already in git)
```

The small checkpoints the films use are in git: `wind/runs/fly_hp_s0_best.npz` (flying learner),
`wind/runs/stage2_c100_s0_best.npz` (crash-priced decider), `urban/rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt`
(the learner trained with birds, rig v2), `urban/clone_corridor.pt`, and the magpie detector
`gz_logan/rig_v2/yolo/run/weights/best.pt`. Everything else (checkpoints, datasets, flights) you regenerate.

## 3. Fly one GPS-denied delivery (part 5)

`geo/geofly.sh` starts PX4 + Gazebo on a photo-floor world, GeoAnchor's own three layers on the live gimbal camera,
the uXRCE-DDS agent and `geo/agp_sim_bridge.py` (GeoAnchor's accepted fixes -> PX4's EKF2 as auxiliary position),
then runs one flight command and cleans everything up. It never uses GeoAnchor's `run.sh` (that syncs every run to
the logs repo); `GEOANCHOR_SYNC_ON_EXIT=0` is set for you.

What every flight does: climbs on GPS to 75 m at the depot; flies the route at 7 m/s on PX4's own estimate (corners
flown through, the nose turned the short way); over the suburb, once GeoAnchor's fixes are arriving and PX4 is
fusing them, **GNSS is switched off** (`EKF2_GPS_CTRL 0`) for the rest of the flight; 50 m before the pad it comes
down to 55 m, settles, drops to 30 m where the pad's 1.80 m marker reads, and a lander takes over. Every marker read
is also a position fix for PX4 (GeoAnchor fades below 50 m). A rangefinder gives the height.

The four films of 27 Sept, exactly as flown (outputs in `gazebo_geo/<OUT>/`):

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

# a hunter magpie, seen only through the camera rig; the learner trained with birds lands it (landing at 0.1x speed)
TREES_JSON=gz_logan/trees_logan.json RIG=v2 RIG_YOLO=gz_logan/rig_v2/yolo/run/weights/best.pt \
  RIG_CALIB=urban/rig_calib_v2.json GEO_LAND_SPEED=0.1 OUT=geo_magpie4 WORLD=geo_magpie INSTANCE=5 geo/geofly.sh \
  /usr/bin/python3 -u gz_learner.py --geo --decider learner --seed 0 --magpie hunter --sense rig \
  --ckpt rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt --video gazebo_geo/geo_magpie4/magpie.mp4
```

Just the GeoAnchor part (no lander; holds 20 s over the pad at 30 m on the marker and reports):

```bash
OUT=smoke WORLD=geo_clean INSTANCE=5 geo/geofly.sh /usr/bin/python3 -u geo/geo_cruise.py
```

What each flight leaves in `gazebo_geo/<OUT>/`: `run.log` (the outcome lines: `GNSS denied at ...`, a `geo {...}`
report, the landing), the film (`.mp4`, 1600x900), PX4's own log (`*.ulg`, the one QGC reads), the GeoAnchor record
(`*_geo.json` or `geo_*.json`: every step's PX4 estimate against Gazebo's truth, and every fix), `agp.jsonl` (what
PX4's estimator fused, when), `live.jsonl` (for the dashboard), GeoAnchor's own run folder.

Reading a result. The `geo` line: `denied_s` (seconds with GNSS off), `fix_err_m` (GeoAnchor's fixes against the
truth), `est_err_m` (PX4's estimate against the truth after the denial), `gnss_restored_at` (null = GeoAnchor held
all the way; if it drops for 2 s, GNSS comes back and this says when), `rtf` (sim speed / real time). Report medians
and tails (p90, p99, % within 5/10/20 m) -- never the mean or RMSE.

Today's numbers are in `results/geo_flights_2026-09-27.json`:

| flight | lander | touchdown from centre | GNSS off | PX4 error (median, GNSS off) | sim speed |
|---|---|---|---|---|---|
| calm | rule | 0.06 m | 110 s | 2.3 m | 0.84x |
| wind 9 m/s | flying learner | 0.19 m | 117 s | 1.7 m | 0.76x |
| obstacles (1st try) | rule + planner | crashed into the crane | | | 0.79x |
| obstacles | rule + planner | 0.04 m, crane passed at 6.7 m | 117 s | 1.9 m | 0.76x |
| magpie | learner with birds | 0.10 m, 3 swoops, closest 1.9 m | 116 s | 1.9 m | 0.25x (0.1x landing) |

One flight each: a demonstration, not a rate.

Deck material from a flight:

```bash
python3 geo/to_webm.py gazebo_geo/geo_calm4/calm.mp4 gazebo_geo/deck/calm4.webm   # VP8 clip + still (internal only)
python3 geo/deck_figs.py                                                            # charts -> gazebo_geo/deck/*.png
```

## 4. The dashboard (fork of GeoAnchor's) -- flight logs and missions

`dashboard/` is a copy of GeoAnchor's dashboard (the original in `geoanchor-rt/dashboard` is untouched) with two
new pages and its own small server:

* **Flight log** -- any flight in `gazebo_geo/`, QGC-style: roll and pitch with their setpoints, heading, height
  (PX4 against Gazebo's truth), cruise and descent speed, PX4's position error, tilt, and a strip of what EKF2 was
  fusing (GNSS, GeoAnchor/AGP, rangefinder, dead reckoning), phases shaded; the track on the map with every fix.
  Hover anywhere to move one cursor across all graphs and the map; drag to zoom. **Live** follows the flight in
  the air.
* **Mission planner** -- the sixteen pads with their marker ids on the map; click to add waypoints, drag to move,
  right-click to delete; pick the pad (any of the sixteen), cruise height, speed, lander, wind. **Save** writes
  `geo/missions/<name>.json`; **Fly it in Gazebo** starts `geo/fly_mission.sh` (refused while another flight is up).

```bash
cd dashboard && npm ci && NEXT_TELEMETRY_DISABLED=1 npx next build && cd ..   # once, and after editing the pages
python3 dashboard/flight_server.py        # -> http://localhost:8010  (GeoAnchor | Flight log | Mission planner)
```

A mission by hand (the dashboard does the same):

```bash
geo/fly_mission.sh geo/missions/to_pad_4.json mission_to_pad_4     # -> gazebo_geo/mission_to_pad_4/
```

`GEO_MISSION=<file>` is all the simulator needs: `geo/geo_cruise.py` takes the route, height and speed from it,
`px4_mission.py` the pad's place and marker ids (the landing camera then refuses our own pad's ids like any
neighbour's), `wind/gz_windland.py` the pad. PX4's `.ulg` is copied into every flight's folder by `geofly.sh`.

## 5. The other parts

* **Wind (MuJoCo MJX twin of PX4, PPO, the Gazebo ladder)** -- `wind/README.md` has the whole recipe.
* **Last mile (fast simulator, headroom, the RL learners)** -- `urban/urban_env.py` is the environment,
  `urban/urban_headroom.py` the headroom study, `urban/urban_rl.py` the training (`--world corridor --eyes --map
  logan_map_lidar.npz --nav gps [--birds --hunter --rig]`), `urban/decider_server.py` serves any decider to Gazebo.
* **Gazebo stage 3 and the rig** -- `gz_learner.py` flies any decider in Gazebo (`--decider rule|planner|learner`,
  `--voxel --local3d` for the crane, `--magpie hunter --sense rig` for the bird); `gz_rig.py` renders the rig's
  training pictures and calibrates it; `rig_yolo.py --data/--train/--score` trains the magpie detector.
* **Parts 1-2** -- `HANDOFF.md`, `CALIBRATION.md` and `notebooklm/` explain the sigma study and its retractions.

## 6. Self-checks (each fails loudly if its logic breaks)

```bash
python3 geo/geo_cruise.py --check                 # the route follower: corners flown through, turned the short way
/home/sai/miniconda3/envs/mujoco_drones/bin/python urban/local_planner.py     # the 3D planner passes a mast, flies a free line straight
python3 dashboard/flight_server.py --check        # PX4's log and the GeoAnchor record line up on one clock
/home/sai/miniconda3/envs/mujoco_drones/bin/python wind/x500_mjx.py --check   # the MJX drone against PX4's step responses
```

## 7. Things that bit us (read before changing the flight code)

* The world's GPS origin must be the Logan depot, or PX4 rejects every GeoAnchor fix.
* gz-sim 8's magnetometer ignores the world's field once spherical coordinates are set: PX4's heading is 10-20 deg
  off on the ground. So GNSS is denied only after flying on it (PX4 fixes its heading from motion).
* Without GNSS the barometer drifts 2-5 m in a descent: the drone carries a rangefinder (`x500_unified_rng`).
* Keep the marker's fixes going to touchdown, with the jump check in `geo/geo_cruise.py` (`MarkerFix`).
* Ask the local planner without waiting (`gz_learner.Decider`), and keep it fast: at 2 s per plan the drone met
  the crane before the answer (27 Sept); weighted A* made it 0.1 s.
* Never zero the velocity command mid-cruise: the old GNSS switch-off did, and PX4 braked at 30 deg of tilt.
* Frames: measure them (`geo/frame_check.py`), never derive them.

## Layout

```
geo/          part 5: worlds, the live GeoAnchor loop, cruise, films, deck figures, missions
wind/         the wind study (MJX twin, PPO, WindDrag Gazebo plugin, the Gazebo ladder)
urban/        the last-mile fast simulator, maps, deciders, RL, the local 3D planner
dashboard/    the forked dashboard + flight_server.py
slides/       the deck's source (deck.json + one HTML per slide); docs/figures/ its charts
results/      small result summaries
gz_logan/     generated worlds and models (assets rebuilt by the scripts; not in git)
*.py (top)    PX4 missions, the pad, perception, the Gazebo landers and the rig
```
