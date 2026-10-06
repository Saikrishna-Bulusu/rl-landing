# shared/px4_gazebo: PX4 flying in Gazebo

The real flight software (PX4 v1.16.2 SITL) in Gazebo, on the Logan world, with the pad, the cameras and the
scripts that fly missions. Every condition's Gazebo flight starts here. The fast simulator
(`shared/fast_sim/`) is the cheap copy; this folder is where its numbers get measured and checked.

Run flight scripts from the repo root, `/home/sai/RL_projects/rl-landing`. Measurement scripts (`measure_*`,
`calib_cam.py`, `find_pad.py`) find their data beside themselves, so any folder works.

```bash
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

## 0. Rules

1. **One Gazebo at a time.** `pgrep -f "gz si[m]"` should print nothing before you start. Two at once dragged the sim to 0.23x real time and broke PX4's position estimate.
2. **Stop simulators by PID or by their world's `GZ_PARTITION`**, never `pkill -f "gz sim"`.
3. **Esri imagery stays out of git.** `gz_logan/imagery/` and `gz_logan/photo/` are ignored on purpose; see `gnss_denied/README.md`.
4. Check the machine first: `free -g`, `nvidia-smi`, AC power (`cat /sys/class/power_supply/A*/online` is 1).

## 1. Install notes (once)

This repo drives other software by absolute path.

| what | where it is expected | used for |
|---|---|---|
| Ubuntu 24.04, NVIDIA GPU | | everything |
| ROS 2 Jazzy (`/opt/ros/jazzy`) and Gazebo Harmonic (gz-sim 8) | system | Gazebo, the `gz` command, ROS 2 for the AGP bridge |
| PX4-Autopilot **v1.16.2**, built with `make px4_sitl` | `/home/sai/thesis_2.0/PX4-Autopilot` | the flight software |
| Micro XRCE-DDS Agent (`MicroXRCEAgent` on the PATH) | system | PX4 <-> ROS 2 (used by `gnss_denied/`) |
| system python (`/usr/bin/python3`): `pymavlink`, `opencv-python`, `numpy`, `python3-gz-transport13`, GDAL, `pyulog`, `playwright` | system or `pip install --user` | flights, films, the Logan world |
| conda env `mujoco_drones` (Python 3.11: `torch`, `jax[cuda]`, `mujoco`, `scipy`, `stable-baselines3`, `gymnasium`) | `/home/sai/miniconda3/envs/mujoco_drones` | the deciders (via `shared/fast_sim/decider_server.py`) |
| Ultralytics YOLO and torchvision: `pip install --no-deps --target .yolo_pkgs ultralytics torchvision polars` | `.yolo_pkgs/` at the repo root | the magpie detector (`PYTHONPATH=.yolo_pkgs YOLO_AUTOINSTALL=false`) |

If you clone the repo anywhere other than `/home/sai/RL_projects/rl-landing`, or PX4 sits somewhere else, find every
hard-coded path with:

```bash
grep -rl "/home/sai/RL_projects/rl-landing" --include=*.sh --include=*.py . | grep -v outputs
```

```bash
grep -rl "thesis_2.0/PX4-Autopilot" --include=*.sh --include=*.py . | grep -v outputs
```

The PX4 path (`P=` / `PX4=`) is in `launch_px4.sh`, `gz_rig.py`, `gz_unified.py`, `gz_voxel.py`, `make_pad.py` (`MODEL`) and
`../fast_sim/gz_logan.py` (`WORLD_SRC`). The repo path (`R=`) is in `launch_px4.sh`, `sitl.sh`,
`../../gnss_denied/geofly.sh`, `../../gnss_denied/fly_mission.sh` and the queue scripts.

The wind plugin (used by wind, weather and GeoAnchor flights) is built once (about a minute):

```bash
cmake -S conditions/wind/gz_winddrag -B conditions/wind/gz_winddrag/build
cmake --build conditions/wind/gz_winddrag/build
```

## 2. Launch PX4 + Gazebo

`launch_px4.sh` starts PX4 SITL with the x500 and a downward camera (airframe 4014, model `gz_x500_mono_cam_down`),
headless, in a world named `aruco`. It wipes `outputs/px4run[_N]/px4run` first: PX4's saved parameters piled up and
began failing the magnetometer check, and a clean start fixed it. It sets the world paths itself because PX4's
generated `gz_env.sh` points at a path that no longer exists.

Variables it reads:

| variable | meaning |
|---|---|
| `WORLDS_DIR` | folder holding `aruco.sdf` (default: PX4's own worlds); for Logan: `shared/px4_gazebo/gz_logan/<world>` |
| `EXTRA_MODELS` | extra model folder, `shared/px4_gazebo/gz_logan/models` |
| `EXTRA_PLUGINS` | folder of extra Gazebo plugins (the WindDrag build) |
| `SERVER_CONFIG` | a sky or wind config file (`conditions/weather/gz_weather.py`, `conditions/wind/gz_wind.py`) |
| `PX4_INSTANCE` | several PX4s side by side (ports 14540+N) |

Two ways in:

1. Start PX4 and a mission world by hand, in one terminal (about a minute to "home set" in `px4.log`; leave it running; stop it by its PID):

```bash
./shared/px4_gazebo/launch_px4.sh
```

2. `sitl.sh`: PX4 + Gazebo on a Logan world, then **one command**, then cleanup. It waits for "home set", sets the sim speed, runs your command with `MAV_PORT=14540+INSTANCE`, and stops everything even on a kill. Output goes to `outputs/gazebo_wind/<OUT>/` (`px4.log`, `run.log`, `run.err`).

```bash
OUT=steps INSTANCE=1 ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u shared/px4_gazebo/px4_steps.py --out outputs/gazebo_wind/steps/steps.csv
```

   Variables: `OUT` (required), `WORLD` (a folder under `gz_logan/`, default `voxfast_v2`), `SERVER_CONFIG`, `SPEED` (default 1.0), `INSTANCE`, `RIG`. Expect `outputs/gazebo_wind/steps/steps.csv` with PX4's step responses (how it answers velocity commands; `conditions/wind/x500_mjx.py --check` compares against it).

GeoAnchor flights start from `gnss_denied/geofly.sh` instead, which adds GeoAnchor and the AGP bridge around the same launch.

## 3. Missions

`px4_mission.py` is the whole pipeline on real PX4:

1. Take off to cruise altitude.
2. Fly to where GeoAnchor *believes* the pad is. That belief is wrong by a replayed real error (`../fast_sim/sigma_pool.npz`).
3. Below the detection ceiling, look for the marker.
4. Once **our** marker is decoded, fly on what the camera sees, centre on it and descend.
5. If the marker is never found, abort: climb and return. Never land blind.

1. The first check: arm, climb, translate, land (needs `launch_px4.sh` running):

```bash
python3 shared/px4_gazebo/px4_bridge.py --check
```

2. Five missions (about 5 minutes each; prints one `=> landed miss ... m` line per mission):

```bash
MISSION_JSON=outputs/px4_mission.json python3 shared/px4_gazebo/px4_mission.py --runs 5
```

3. Prove the safety layer: expect a marker that is not in the world; it must abort, not land:

```bash
MISSION_JSON=outputs/px4_mission.json python3 shared/px4_gazebo/px4_mission.py --wrong-id 7
```

Other flags: `--pool` (the error pool), `--scale`, `--aim mean|latest`, `--policy <ppo .zip>`, `--search-k`, `--search-alt`, `--video out.mp4`, `--seed0`, `--vary-forest`, `--forest0`.

Environment variables: `GEO_MISSION` (a dashboard mission file: the pad's place and marker ids), `TREES_JSON` (trees for crash scoring; `gz_logan/trees_logan.json` for the Logan world), `MISSION_JSON` (where results go), `MISSION_TRACE`, `RIG` (a gimbal rig).

**Hazard: always set `MISSION_JSON`.** Its default is the bare name `px4_response.json`, written into whatever folder you are in. Run from `shared/px4_gazebo/` it would overwrite the tracked measurement of PX4's lags. From the repo root it makes a stray file there.

`gz_learner.py` flies the fast simulator's planner or learner on PX4 (section 7). `px4_steps.py` records how PX4 answers velocity commands (section 2, item 2).

## 4. Perception

`px4_perception.py` turns PX4 + Gazebo into the task: ArUco detection from the downward camera, and GeoAnchor's error injected into the believed pad position.

- Pictures are read straight from Gazebo's transport (no ROS in the loop, so the latency is honest).
- **The identity check is a hard safety layer**, not an observation: a wrong marker's id is reported as *nothing seen*. In the cage study a neighbour's marker fooled an unprotected drone 138 times out of 150; making it look like no marker let the abort the policy already had handle it.
- GeoAnchor estimates the drone's position, not the pad's. The pad is known exactly, so the believed pad is `pad - err`, not `pad + err` (the self-test checks the sign).

Self-test (8 seconds, no simulator needed except for its last item):

```bash
python3 shared/px4_gazebo/px4_perception.py --selftest
```

Expect five numbered checks and `selftest complete`. Item 5, the live camera, prints `no frames yet -- is SITL up and the drone airborne?` when no simulator runs; that is fine.

## 5. The pad: three markers

`make_pad.py` generates the pad (not hand-drawn): a 5.0 m white mat with three ArUco markers (dictionary `DICT_4X4_50`) side by side.

| id | size | centre along the mat | readable from |
|---|---|---|---|
| 2 | 1.80 m | -1.275 m | 28.6 m down to 0.76 m |
| 1 | 0.15 m | 0 (the model origin: do not move) | 2.4 m down to 0.06 m |
| 0 | 1.20 m | +0.875 m | 19.1 m down to 0.51 m |

Why three: a marker only decodes from `h = size * 1280 / (2 * 34 px * tan(hfov/2))`. A 1.20 m marker is readable only below 19 m,
which is under a mature tree, so the drone would hunt for the pad from inside the canopy. 1.80 m reads from 28.6 m. 2.40 m was tried
and buys nothing over 1.80 m, and under a tight 35 s budget it delivered less (86.7% against 98.0%). Markers cannot nest (a
centre hole destroys the outer marker's data bits; all 50 ids were tested). The bands overlap, so there is no height with nothing
to see; the script asserts it.

1. Dry run: builds the pad in memory, checks it against its own pixels and prints the ladder (a few seconds, writes nothing):

```bash
python3 shared/px4_gazebo/make_pad.py
```

   Expect `ok` lines and `ok  the bands overlap: no altitude with nothing to see`, then `(dry run -- pass --write to install)`.

2. Install our pad into PX4's model tree (`/home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/models/arucotag`; outside this repo, and it replaces the model there):

```bash
python3 shared/px4_gazebo/make_pad.py --write
```

`../fast_sim/gz_logan.py` reuses the painter for the neighbours' pads: sixteen pads in all, ours at Gazebo (120, 0) with ids 2, 1, 0, and
`logan_pad_1` to `logan_pad_15` with ids `3k+2, 3k+1, 3k`. The landing refuses every neighbour's id, like any wrong marker. Where they are:
`gz_logan/pads_logan.json`.

## 6. Camera model

| camera | picture | field of view | note |
|---|---|---|---|
| landing camera (`x500_mono_cam_down`) | 1280 x 960, 30 Hz | 99.7 deg across | focal length 539.94 px, from the `camera_info` topic |
| gimbal camera (rig v2 and later) | 1280 x 720 | 2.0 rad across | held down by PX4's gimbal; its picture lags at 5 Hz, so project with its own IMU attitude |

Pixel to ground: `dNED = (h / f) * R(yaw) * B * (ex, ey)` where `h` is the height, `f` the focal length, `R(yaw)` the live yaw
read from the vehicle every frame, and `B` a fixed camera-to-body rotation solved once and stored in `cam_B.npy` (forced orthonormal; it
comes out at 0.964 and 0.947). An older single fitted matrix baked in one yaw and drove the drone 40 m the wrong way at another heading.

Measured constants (`CALIBRATION.md`, with how to re-measure each after any change to the airframe, camera, marker or PX4 build):

| what | file | measured by |
|---|---|---|
| velocity lag, 0.69 s sideways and 0.23 s up/down | `px4_response.json` | `measure_tau.py` |
| decode rate against marker size | `detect_curve_60m.json` (also `detect_curve.json`, `detect_curve_old60.json`) | `measure_detect.py` |
| camera-to-body rotation | `cam_B.npy` | `calib_cam.py` |
| Gazebo ENU to PX4 NED (+90 deg, residual 0.05 m) | `PAD_XY` in `px4_mission.py` | `find_pad.py` |

Decode rate (marker id 2, `detect_curve_60m.json`): 78% at 20-25 px wide, 88% at 25-30, 99.5% at 30-35, 100% above. The
fast simulator is fitted to this curve. `CALIBRATION.md` still quotes the older 35 px rule of thumb and a table from before the
curve changed. `find_pad.py` and `calib_cam.py` print a residual; if it is not small, do not use the result.

Re-measure (needs `launch_px4.sh` running; each prints its result):

```bash
python3 shared/px4_gazebo/measure_tau.py
```

```bash
python3 shared/px4_gazebo/find_pad.py
```

```bash
python3 shared/px4_gazebo/calib_cam.py
```

```bash
python3 shared/px4_gazebo/measure_detect.py
```

Frames: Gazebo is ENU (x east, y north, z up), PX4 is NED (z down). From the depot, PX4 north is Gazebo y and PX4 east is
Gazebo x. Our pad is ENU (120, 0) = NED (0, 120). Measure frames, never derive them.

## 7. The Logan worlds

`../fast_sim/gz_logan.py` builds the plain world (houses, trees, roads, sixteen pads) from the LiDAR survey into `gz_logan/`.
The scripts below add their own worlds and drones on top. Run each once:

1. The plain Logan world (run from `shared/fast_sim/`; needs the maps from its README):

```bash
cd shared/fast_sim && /usr/bin/python3 gz_logan.py
```

2. Stage 3: a 58 m crane, street lights, a traffic light, new trees, a shed, and the x500 with a depth camera:

```bash
/usr/bin/python3 shared/px4_gazebo/gz_voxel.py --setup
```

3. The 360-degree camera rig and the magpie bench:

```bash
/usr/bin/python3 shared/px4_gazebo/gz_rig.py --setup
```

4. Rig v2 and its variants (the _rng drone with a rangefinder and the photo-floor worlds come from `gnss_denied/geo_world.py`):

```bash
/usr/bin/python3 shared/px4_gazebo/gz_unified.py --setup
```

5. The magpie world:

```bash
/usr/bin/python3 shared/px4_gazebo/gz_learner.py --setup
```

Rig variants (`gz_unified.py`): v2 (gimbal landing camera, OAK-D ahead, wide cameras left, right, back), v2f (a fixed down camera
instead of the gimbal), v2t (back camera tilted up 30 deg), v2u (v2t plus an up camera), v2l (the up camera with the back camera
level: the one the magpie flights use). Select with `RIG=<name>`.

Layout of `gz_logan/` (each world is a folder with `aruco.sdf`; Gazebo is told the world is named `aruco`):

| folder | what | in git |
|---|---|---|
| `worlds/` | the plain Logan world | no (rebuilt) |
| `models/` | `logan_*` (houses, trees, roads, pads), `x500_*` drones | the `x500_*` drones yes, `logan_*` no |
| `voxfly/`, `voxfast/`, `voxbench/` (+ `_v2*`) | stage 3: video flights, bulk flights, the bench | the `aruco.sdf` files |
| `rig*/`, `rigfly*/` | the camera rig bench, data and flights (+ variants) | the `aruco.sdf` files; `rig_v2/yolo/run*/weights/best.pt` (the two magpie detectors) |
| `geo_*/` | the photo-floor worlds, made by `gnss_denied/geo_world.py` | the `aruco.sdf` files and relative links |
| `weather/`, `wind_v2/`, `windyard/` | skies (`conditions/weather/`), the wind world and the wind-landing yards (`conditions/wind/`) | mostly |
| `birds/` | the magpie world | no |
| `imagery/`, `photo/` | Esri captures; renders | **never** (Esri) |
| `trees_logan.json`, `pads_logan.json` | tree crash scoring; the sixteen pads | yes |

Generated or heavy parts (`materials/`, `meshes/`, `data*/`, `cover*/`, `yolo/ds*`, `gz.log`) are ignored by `.gitignore` and rebuilt by the scripts above.

The two magpie detectors: `gz_logan/rig_v2/yolo/run/weights/best.pt` (plain ground) and `.../run_geo/weights/best.pt` (retrained on photo ground: use
this one over the photo floor). Running the rig, detector and magpie flights is in `conditions/magpie/README.md`; the crane in
`conditions/static_obstacles/README.md`.

## 8. Trees and the forest

- `forest_spec.py`: the tree sizes in one place, with no dependencies (the point-mass env, the world file and the runtime spawner must agree).
- `make_trees.py`: puts the tree model into the Gazebo world. **Visual only, no collision:** a brush against a canopy would tumble the drone and the next mission would start from a wreck. The crash is scored geometrically in `px4_mission.py` with the env's own test. The canopies still hide the marker from the camera, which the point-mass model cannot show.
- `forest.py`: replants a different forest at run time in about a second (a full SITL restart takes 45 s).
- `trees.json`: the default layout (`TREES_JSON` overrides it).

Studies that use them: `conditions/static_obstacles/` (`ab_pad.sh`, `run_forest20.sh`, `bigger_marker.py`).

## 9. Files

| file | what |
|---|---|
| `launch_px4.sh`, `sitl.sh` | PX4 + Gazebo; with a command and cleanup |
| `px4_bridge.py` | MAVLink link to PX4 (offboard velocity setpoints at 20 Hz); `--check` |
| `px4_mission.py`, `px4_perception.py`, `px4_steps.py` | missions, perception, step responses |
| `make_pad.py`, `make_trees.py`, `forest.py`, `forest_spec.py`, `trees.json` | the pad and the trees |
| `calib_cam.py`, `find_pad.py`, `measure_tau.py`, `measure_detect.py`, `CALIBRATION.md` | the measurements |
| `cam_B.npy`, `cam_to_ned.npy`, `detect_curve*.json`, `px4_response.json` | what they measured |
| `gz_voxel.py`, `gz_rig.py`, `gz_unified.py`, `gz_learner.py` | the four setups and the landers that fly them |
| `gz_logan/` | worlds, models and detectors (section 7) |

## 10. Things that bit us

- PX4 offboard mode fails silently three ways: setpoints must already be streaming before the mode switch, they must keep arriving faster than 2 Hz, and PX4 is NED (a descent is a positive `vz`).
- Setting poses from the flight process stalls every thread (29 -> 2 Hz); `gz_learner.Mover` uses a helper process.
- Ask the local planner without waiting (`gz_learner.Decider`) and keep it fast: at 2 s per plan the drone met the crane before the answer.
- Gazebo's compass ignores the world's magnetic field once lat/lon is set; PX4's heading is 10-20 degrees off on the ground.
- Gazebo's GPS has no delay but PX4's EKF assumes 110 ms; the launch scripts set `EKF2_GPS_DELAY=0`.
- Sixteen big textures plus a training on the same GPU drop Gazebo to 0.15-0.3x real time; keep one GPU job beside Gazebo.
- Gazebo's model origin sits at the drone's feet, which swing back when it pitches: compare the centre of mass, not the origin.
