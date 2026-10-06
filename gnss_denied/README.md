# gnss_denied: GeoAnchor flying PX4

Whole deliveries in Logan with GPS switched off. PX4 flies in Gazebo. GeoAnchor (a separate project, never edited
here) looks down through the drone's gimbal camera, matches the picture to an old aerial photo and says where the
drone is. PX4's position filter (EKF2) takes those fixes in place of GPS. In the last 50 m the pad's marker gives
the position instead. A lander (rule, flying learner, planner or magpie learner) then puts the drone down.

Run every command in this file from the repo root, `/home/sai/RL_projects/rl-landing`.

```bash
GA=/home/sai/GeoAnchor/geoanchor-rt/.venv/bin/python
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

## 0. Rules (read first)

1. **Esri rule.** The Gazebo floor is an Esri Wayback photo (2023 capture) and GeoAnchor's map is another (2021).
   They are for measuring only. Films, stills, texture tiles and dashboard map screenshots never go into git, a
   paper or a public deck. `.gitignore` already blocks `shared/px4_gazebo/gz_logan/imagery/`, `photo/`, `*/meshes/`,
   `*/materials/`, `*.mp4`, `*.tif` and all of `outputs/`. Never texture the floor with the picture GeoAnchor
   matches against: the floor is 2023, the map is 2021.
2. **One Gazebo at a time.** Two PX4 + Gazebo pairs on this laptop dragged the sim to 0.23x real time and broke
   PX4's position estimate. `INSTANCE=5` is the one these scripts use.
3. **Never GeoAnchor's own `run.sh`.** It pushes every run to the GeoAnchor logs repo. `geofly.sh` starts the layers
   itself and sets `GEOANCHOR_SYNC_ON_EXIT=0`.
4. **GeoAnchor is read-only from here.** Check `free -g`, `nvidia-smi` and AC power (`cat /sys/class/power_supply/A*/online` is 1) before a long flight.

## 1. Prove the stack is alive (no Gazebo, 1 s)

1. The route follower:

```bash
python3 gnss_denied/geo_cruise.py --check
```

   Expect: `route flown in 79 s: slowest 2.4 m/s, corner -141 deg (left), command change <= 2.00 m/s^2, at most 3.7 m off the route`. An assert fails loudly if the corner is not turned the short way.

2. The dashboard's log reader (PX4's log and GeoAnchor's record on one clock). It reads a flight that already exists in `outputs/gazebo_geo/`:

```bash
python3 dashboard/flight_server.py --check
```

   Expect lines like `stress_e6: 1645 samples at 10 Hz, max height 75 m, PX4 vs truth median 1.0 m, ...`.

## 2. Build the worlds (once)

The worlds need two aerial photos and the Logan world from `shared/px4_gazebo/` (its README says how). Both photos
come from GeoAnchor's `scripts/fetch_wayback_tile.py` (its docstring has the flags). They are the 2023-06-29 capture
(the floor) and the 2021-07-21 capture (GeoAnchor's map), saved as:

- `shared/px4_gazebo/gz_logan/imagery/ref_tile_2023-06-29.tif`
- `shared/px4_gazebo/gz_logan/imagery/ref_tile_2021-07-21.tif`

3. The photo-floor worlds (GeoAnchor's python, because it has `cv2`, `pyproj` and `rasterio`; time not measured, minutes):

```bash
$GA gnss_denied/geo_world.py
```

   It prints each world's file. It first asserts that Gazebo and lat/lon convert both ways to 1 micrometre and that the pad lands at -27.660121, 153.142424.

4. Rig variants of the same worlds (v2l is the rig with the up camera):

```bash
$GA gnss_denied/geo_world.py --variant v2l
$GA gnss_denied/geo_world.py --stress
```

Worlds, all under `shared/px4_gazebo/gz_logan/` (each `aruco.sdf` is tracked; the floor texture and meshes are not):

| world | what is in it |
|---|---|
| `geo_floor` | the photo floor, draped roofs and tree crowns, plus stage 3's crane, lights, traffic light, trees and a shed round the pad |
| `geo_clean` | `geo_floor` without those ten obstacles, for wind and plain flights (one hazard per flight) |
| `geo_magpie` | `geo_clean` plus the magpie and its nest |
| `geo_magpie_v2l` | `geo_magpie` flown by the rig with the up camera |
| `geo_stress`, `geo_stress_v2l` | `geo_floor` plus the magpie: the full-stack stress test |

Every world has the depot's real GPS position (Gazebo's default is Zurich, and EKF2 then rejects every fix), a
downward rangefinder on the drone (without GNSS the barometer drifts 2-5 m in a descent), and the films' chase
camera. The roofs and crowns carry the same photo (as flat blocks they hid 40% of every suburban frame from the
matcher). `geo_clean`, `geo_magpie` and the other variants link `meshes` and `materials` from `geo_floor` with
**relative** links (`../geo_floor/meshes`); the script makes them that way.

## 3. What `geofly.sh` does

`gnss_denied/geofly.sh` starts everything one flight needs, runs one command, and cleans up.

1. PX4 SITL + Gazebo on `WORLD`, through `shared/px4_gazebo/launch_px4.sh`, with EKF2 told to accept GeoAnchor as Aux Global Position (AGP).
2. GeoAnchor's three layers (output, processing, data) on the live gimbal camera, with their own message sockets under `/tmp/geoanchor_rl/<INSTANCE>`.
3. `MicroXRCEAgent` on UDP port 8890+INSTANCE (PX4 <-> ROS 2).
4. `gnss_denied/agp_sim_bridge.py`, the bridge from GeoAnchor's accepted fixes to PX4 (section 4).
5. Your command, with `MAV_PORT=14540+INSTANCE`, and a trap that stops everything by PID and by its own `GZ_PARTITION`, never `pkill`.

It reads these variables:

| variable | meaning |
|---|---|
| `OUT` (required) | the folder `outputs/gazebo_geo/<OUT>/` |
| `WORLD` | a folder under `shared/px4_gazebo/gz_logan/` (default `geo_clean`) |
| `INSTANCE` | PX4 instance, default 5 (ports 14540+N, 8890+N; ROS domain 40+N) |
| `SERVER_CONFIG` | the WindDrag wind, for example `shared/px4_gazebo/gz_logan/geo_clean/geo_w9.config` |
| `AGP_DELAY` | `EKF2_AGP_DELAY` in ms, default 100 |
| `SPEED` | sim speed, default 1.0 |
| `RIG` | camera rig name, default `v2` |

The wind needs the WindDrag plugin built once (about a minute, needs cmake):

```bash
cmake -S conditions/wind/gz_winddrag -B conditions/wind/gz_winddrag/build
cmake --build conditions/wind/gz_winddrag/build
```

A finished flight leaves, in `outputs/gazebo_geo/<OUT>/`: `run.log` (the result lines), `run.err`, `px4.log`,
`agp.jsonl` (what EKF2 fused, when), `bridge.out`, `xrce.log`, `ga_*.out`, `geoanchor/` (GeoAnchor's own records),
PX4's `*.ulg`, and what the command wrote (the film, `flights.jsonl`).

## 4. The flight, the bridge and the 4-fixes-in-15-s rule

`gnss_denied/geo_cruise.py` is the flight. In order:

1. Straight up at the depot on GPS to 75 m.
2. The route at 75 m and 7 m/s, flown on PX4's own estimate. The corners are flown through (not stopped at), and the nose turns the short way.
3. Over the suburb GNSS is switched off (`EKF2_GPS_CTRL 0`) once GeoAnchor is steadily in PX4's loop. **Steadily means:** PX4 reports it is fusing AGP, and at least 4 fixes were sent in the last 15 s (`HEALTHY = 4` in `geo_cruise.py`). The first try denied GNSS on the first fused fix, no more came, EKF2 lost its position and PX4 landed blind. If PX4 drops GeoAnchor for 2 s afterwards, GNSS comes back and the record says when (`gnss_restored_at`).
4. 50 m before the pad it comes down to 55 m, settles, then goes straight down to 30 m. That is the height where the pad's 1.80 m marker reads (88% of frames at 30 m, 13% at 50 m: `read_height.py`). From 50 m out every marker read is also a position fix for PX4.
5. The lander takes over (section 5). On its own, `geo_cruise.py` stops at 30 m and reports.

The bridge, `gnss_denied/agp_sim_bridge.py`, sends only accepted fixes, with finite values, `eph = max(sigma, 0.1 m)` and no
height claim. One difference from GeoAnchor's own bridge: the time stamp goes out as 0, so PX4 stamps each fix on
arrival in its own clock and `EKF2_AGP_DELAY` carries the pipeline's lag. Reason: in lockstep SITL PX4's clock is
Gazebo's, which falls behind the wall clock whenever the sim runs slow (the magpie flights do), so a converted stamp
would be wrong by however far it fell behind. It also logs every fix sent or skipped and PX4's own flags
(`cs_gnss_pos`, `cs_aux_gpos`) once a second.

## 5. Flights

Each command runs one flight. Time is on this laptop, 1x speed unless said.

1. Smoke test: one GNSS-denied cruise to 30 m over the pad (5-10 minutes).

```bash
OUT=smoke WORLD=geo_clean INSTANCE=5 gnss_denied/geofly.sh /usr/bin/python3 -u gnss_denied/geo_cruise.py
```

```bash
tail -3 outputs/gazebo_geo/smoke/run.log
```

   Expect `GNSS denied at t=...`, then a `geo {...}` line with `denied_s` near 110, `gnss_restored_at: null`, and PX4's `est_err_m` median near 2 m.

2. Calm air, the rule lands it (5-10 minutes):

```bash
OUT=geo_calm4 WORLD=geo_clean INSTANCE=5 gnss_denied/geofly.sh /usr/bin/python3 -u conditions/wind/gz_windland.py --geo --episode 0 --w10 0 --decider rule --out outputs/gazebo_geo/geo_calm4/flights.jsonl --video outputs/gazebo_geo/geo_calm4/calm.mp4
```

   Expect `landed miss 0.06 m, 25.4 s` as the last `run.log` line (the 27 Sept flight).

3. 9 m/s wind at 10 m (the WindDrag plugin), the flying learner lands it (5-10 minutes):

```bash
OUT=geo_wind10 WORLD=geo_clean SERVER_CONFIG=shared/px4_gazebo/gz_logan/geo_clean/geo_w9.config INSTANCE=5 gnss_denied/geofly.sh /usr/bin/python3 -u conditions/wind/gz_windland.py --geo --episode 0 --w10 9 --decider conditions/wind/runs/fly_hp_s0_best.npz --out outputs/gazebo_geo/geo_wind10/flights.jsonl --video outputs/gazebo_geo/geo_wind10/wind9.mp4
```

4. An unmapped 58 m crane on the way in. The hand-over is 90 m out at 75 m; the rule plus the local 3D planner on the voxel map does the rest (5-10 minutes):

```bash
CRANE_AT=147.33,-23.43 GEO_GOAL_OUT=90 OUT=geo_obst5 WORLD=geo_floor INSTANCE=5 gnss_denied/geofly.sh /usr/bin/python3 -u shared/px4_gazebo/gz_learner.py --geo --decider rule --seed 1 --magpie none --voxel --local3d --video outputs/gazebo_geo/geo_obst5/obstacles.mp4
```

5. A hunter magpie seen only through rig v2l and the retrained detector. The learner trained with birds lands it. The landing runs at 0.1x speed so the detector keeps up (about 12 minutes; needs `.yolo_pkgs`, see `shared/px4_gazebo/README.md`):

```bash
TREES_JSON=shared/px4_gazebo/gz_logan/trees_logan.json RIG=v2l RIG_YOLO=$PWD/shared/px4_gazebo/gz_logan/rig_v2/yolo/run_geo/weights/best.pt RIG_CALIB=$PWD/shared/fast_sim/rig_calib_v2l.json GEO_LAND_SPEED=0.1 OUT=geo_magpie6 WORLD=geo_magpie_v2l INSTANCE=5 gnss_denied/geofly.sh /usr/bin/python3 -u shared/px4_gazebo/gz_learner.py --geo --decider learner --seed 0 --magpie hunter --sense rig --ckpt rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt --video outputs/gazebo_geo/geo_magpie6/magpie.mp4
```

   The detector must be `run_geo` (retrained on photo ground); the plain-ground one fires on tree crowns over the photo floor.

6. A mission from the dashboard's planner (route, pad, height, speed, lander, wind in one json file):

```bash
gnss_denied/fly_mission.sh outputs/gnss_denied/missions/example_pad4.json mission_example_pad4
```

   Output goes to `outputs/gazebo_geo/mission_example_pad4/`. `GEO_MISSION=<file>` is all the simulator needs: `geo_cruise.py` takes route, height and speed, `shared/px4_gazebo/px4_mission.py` the pad's place and marker ids, `conditions/wind/gz_windland.py` the pad. The dashboard refuses to start it while another Gazebo is up; the script itself does not check, so look first (`pgrep -f "gz si[m]"` prints nothing when the machine is free).

7. The full-stack stress test (GNSS denied + 9 m/s wind + crane + magpie) is `all_conditions/queue/stress_30sep.sh` (a smoke flight, then ten). It waits for a marker file from the wind queue first; read its header before starting it. Chains for the other flights are in `gnss_denied/queue/` (`geo_calm10_30sep.sh` is ten calm flights, `overnight_27sep_b.sh` the crane and magpie films).

## 6. Does GeoAnchor match this floor at all? (about 20 minutes)

Record a 75 m sweep, replay it through GeoAnchor's own layers, score the fixes against Gazebo's truth.

1. Fly and record the sweep (video + a truth file per frame):

```bash
OUT=geo_rec_floor WORLD=geo_floor INSTANCE=5 shared/px4_gazebo/sitl.sh /usr/bin/python3 -u gnss_denied/record_sweep.py --out outputs/gazebo_geo/floor
```

2. Measure which way the image top points (needs the sweep; fails loudly if the frames do not match the floor):

```bash
$GA gnss_denied/frame_check.py outputs/gazebo_geo/floor
```

3. Replay (prints the run folder `outputs/gnss_denied/runs/floor_<stamp>/`, then the score):

```bash
gnss_denied/replay.sh outputs/gazebo_geo/floor floor
```

4. Score a run by hand (median, p90, p99, share within 5, 10, 20 m; never the mean):

```bash
python3 gnss_denied/score.py outputs/gnss_denied/runs/<run> outputs/gazebo_geo/floor/flight_gt.jsonl
```

5. How high can the camera read the pad:

```bash
OUT=geo_readh WORLD=geo_floor INSTANCE=5 shared/px4_gazebo/sitl.sh /usr/bin/python3 -u gnss_denied/read_height.py
```

   Expect a table by height from 60 m to 25 m, and `outputs/gazebo_geo/read_height.json`.

`geo_logan_replay.yaml` is GeoAnchor's config for these runs (its paths point into `outputs/`; `geofly.sh` and
`replay.sh` set them for you). `stores/` holds GeoAnchor's reference store built from the 2021 tile; it has Esri
pixels in it (`preview.png`, `tiles/`), so it stays out of any paper or deck.

## 7. Reading a result

`outputs/gazebo_geo/<OUT>/run.log` has `GNSS denied at ...`, a `geo {...}` line and the landing line.

| field | meaning |
|---|---|
| `denied_s` | seconds with GNSS off |
| `fix_err_m` | GeoAnchor's fixes against the truth |
| `est_err_m` | PX4's own estimate against the truth after the denial |
| `gnss_restored_at` | null means GeoAnchor held all the way |
| `rtf` | sim speed over real time |

Report medians and tails (p90, p99, share within 5, 10, 20 m), never the mean or RMSE. The eight 27 Sept flights
(one each: demonstrations, not rates) are in `gnss_denied/results/geo_flights_2026-09-27.json`; its stored commands
use the old paths. For example: calm, rule, 0.06 m from centre, GNSS off 110 s, PX4 error median 2.3 m, sim 0.84x.

To see a flight (roll, pitch, height, speed, what EKF2 fused, the map track): the dashboard, `dashboard/README.md`.

## 8. Things that bit us

- The world's GPS origin must be the Logan depot, or PX4 rejects every GeoAnchor fix.
- gz-sim 8's magnetometer ignores the world's field once lat/lon is set, so PX4's heading is 10-20 degrees off on the ground. GNSS is therefore denied only after flying on it (PX4 fixes its heading from motion). Avoid turns right after denial.
- Never zero the velocity command mid-cruise: the old GNSS switch-off did and PX4 braked at 30 degrees of tilt.
- Keep marker fixes going to touchdown with the jump check (`MarkerFix` in `geo_cruise.py`).
- Frames are measured, never derived (`frame_check.py`).
- Stop simulators by PID or by their world's partition, never `pkill -f "gz sim"` (it kills the other partition's bench).
- `geo_world.py` makes `meshes` and `materials` links relative. An absolute link breaks if the repo moves.

## Files

| file | what |
|---|---|
| `geofly.sh` | one flight: PX4 + Gazebo + GeoAnchor + bridge + your command |
| `geo_cruise.py` | the cruise, GNSS denial, descent to 30 m; `--check` |
| `agp_sim_bridge.py` | GeoAnchor fixes into EKF2 as AGP |
| `geo_world.py` | the photo-floor worlds, `--variant`, `--stress` |
| `geo_film.py` | the film (chase camera, GeoAnchor's view, map, status bar) |
| `record_sweep.py`, `frame_check.py`, `replay.sh`, `score.py` | does GeoAnchor match this floor |
| `read_height.py` | marker read rate by height |
| `fly_mission.sh` | a dashboard mission, end to end |
| `geo_logan_replay.yaml` | GeoAnchor's config for the replay and the live flights |
| `stores/` | GeoAnchor's reference store (Esri pixels: never published) |
| `queue/` | overnight chains for these flights |
| `results/` | the 27 Sept table |
