# shared/fast_sim: the fast simulator for the last kilometre

Thousands of Logan deliveries flown at once on the GPU, to ask: how much better could a decider be, and can a
learned one get there? PX4 and Gazebo are far too slow for that (thousands of landings need days). This folder
also holds the Logan maps, the deciders (rule, planners, the learner), the local 3D planner and the server that
lets Gazebo flights use the same deciders.

Run every command in this folder **from this folder**. The scripts load their data by bare name
(`logan_map_lidar.npz`, `rig_calib_v2l.json`, ...).

```bash
cd /home/sai/RL_projects/rl-landing/shared/fast_sim
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

## 1. What it is, in plain words

`urban_env.py` is one class, `UrbanEnv`, that holds N flights as PyTorch tensors (N = 4,096 for the ladders and the learner's tests) and advances
them all together, one step for every drone at once. A "step" is 0.25 s of simulated time. Agents read the
observation and hand back a velocity command, the way PX4 offboard mode takes one. The drone is a point mass with
PX4's measured lags (0.69 s sideways, 0.23 s up and down); the world is the real Logan map.

It is **not** a Gymnasium or PettingZoo environment. Gymnasium is a standard plug shape (`reset()` and `step()`
returning observation, reward, terminated, truncated, info) so any training library can connect. This simulator has
the same idea but its own batched shape:

- `reset()` starts all N flights together, `step(cmd)` takes an (N, 3) command and returns only the observation.
- The reward is not in the environment. `urban_rl.py` computes it.
- Finished flights freeze until every flight is done ("synchronous generations"); single drones cannot be reset.
- Nothing declares observation or action spaces.

PettingZoo is for several learning agents. Here exactly one agent learns (our drone); the helicopter, other
drones, people and the magpie follow scripts. One drone per Python call would be far too slow, so the plug was
dropped. The earlier single-drone studies in `archive/` are Gymnasium environments. A wrapper would take a day or
two and would change no result. Details: `papers/research/aerial_gym_comparison.md`.

Sources of every number are in the top docstring of `urban_env.py` (measured on PX4, CASA and ASTM rules, or
assumed and marked so).

## 2. Files

| file | what |
|---|---|
| `urban_env.py` | the simulator (`python urban_env.py` is its self-check, 20 s) |
| `urban_agents.py` | the deciders: `Rule`, and `MPC` planners at three levels of information (what sensors show now, + imagined intruders, + the true future as an oracle) |
| `urban_headroom.py` | the headroom ladder: rule, best planner, oracle, all on the same episodes |
| `urban_rl.py` | the learner: copies the planner, then PPO; 14 candidate manoeuvres every 0.5 s; ClearML logging |
| `clone_check.py` | can the learner's own network hold the planner at all |
| `rig_sense.py` | the camera rig's stand-in for the fast simulator (detection chance by size, blind directions, a bird tracker), from numbers measured in Gazebo (`rig_calib*.json`) |
| `local_planner.py` | the local 3D planner (A* on the survey plus the depth camera's voxel map); self-check by running it |
| `decider_server.py` | serves the planner or learner to a Gazebo flight over stdin and stdout, every 0.25 s of sim time |
| `why_cond.py`, `rl_why.py`, `sens.py`, `sens/` | why flights fail per condition; sensitivity to the simulator's assumptions |
| `paper_figs.py`, `plot_*.py` | figures from saved results |
| `clearml_import.py` | puts past logs into ClearML (`--dry-run` first) |

Map builders and their data:

| file | what |
|---|---|
| `logan_map.py` | the old map: OpenStreetMap roads and land use, Microsoft building footprints (4,582). Inputs `logan_osm.json`, `ms_logan.geojson` are in git |
| `lidar_map.py` | the 2023 LiDAR survey onto the map's 1 m grid |
| `logan_map_lidar.py` | the map the study flies: every surveyed house, tree and pole, and the pads |
| `lidar_frame.py`, `plot_lidar.py` | checks and a picture of the survey against the old map |
| `gz_logan.py` | the Logan world for Gazebo (writes into `../px4_gazebo/gz_logan/`) |
| `DATA_2462013.zip` | the survey (QLD Government, ELVIS, 25 tiles, 15.6 GB). Not in git |

## 3. Build the maps (once)

1. Get the survey: order the Logan tiles from <https://elevation.fsdf.org.au> and save the zip here as `DATA_2462013.zip`.

2. The old map (time not measured; the LiDAR step reads it):

```bash
/usr/bin/python3 logan_map.py
```

   Expect `logan_map.npz` (about 10 MB).

3. The survey on the map's grid (about 20 minutes; the system python has GDAL; nothing is unzipped):

```bash
/usr/bin/python3 lidar_map.py
```

   Expect `lidar_logan.npz` (58 MB), `lidar_logan.json`, and a cache `lidar_mga.npz` (3.5 minutes to rebuild alone).

4. The map the fast simulator flies:

```bash
/usr/bin/python3 logan_map_lidar.py
```

   Expect `logan_map_lidar.npz` (66 MB) and the pad counts printed.

5. The Gazebo world (needs `make_pad.py`'s output installed first, see `../px4_gazebo/README.md`):

```bash
/usr/bin/python3 gz_logan.py
```

   Expect the Logan patch (240 m crop round the pad) written to `../px4_gazebo/gz_logan/`.

## 4. The 1,324 pads

`logan_map_lidar.npz` has `pads`, one delivery pad per qualifying house. The rule (`logan_map_lidar.py`):

- 1,518 surveyed houses have 80 to 500 m^2 of roof on residential land.
- Each gets a pad moved inside its own yard to the spot furthest from a road (at least 12 m back) that has nothing over 2 m within 2.5 m and clear ground (nothing over 0.3 m within 1.5 m) under the mat.
- A house with no such spot gets no deliveries, as a real service would refuse it. 1,324 pads remain.

Why 2.5 m: in the forest study every crash happened where the clear column above the pad was under 4.2 m wide, and 0.4 m
more covers the marker-fix error on descent. The old map's rule (4 m clear of mapped obstacles) barely counted trees:
58% of those pads had a tree over 5 m within 3 m. Logan LiDAR also exposed that the old map lost a strip of houses in
the south (near y = -1400 m).

## 5. Run the study

Times are on the RTX 5070 Laptop (8 GB). One headroom ladder uses about 3.3 GB and one training about 2 GB; never more
than one of each next to a Gazebo flight. Check `nvidia-smi` first.

1. Self-checks (a minute in all):

```bash
$P urban_env.py
$P local_planner.py
```

   Expect nine numbered lines (the last: `9. hunter magpie: touched 10.8% of dives ...`) and `self-checks passed: mast passed at 2.3 m via 15 waypoints; ...`.

2. Headroom: how far is the rule from a perfect decider (30-60 minutes per ladder):

```bash
$P urban_headroom.py --map logan_map_lidar.npz --nav gps
```

   Writes `headroom_lidar_gps.json`. The file name is built from the flags, so the same flags overwrite the saved result in git; add `--tag _rerun` to keep both. It prints the headroom ladder and a guard: an oracle that scores below the rule is broken, and the run stops. Add `--birds --hunter` for the magpie; add `--rig --tag _v2l` with `RIG_CALIB=rig_calib_v2l.json` for the camera rig:

```bash
RIG_CALIB=rig_calib_v2l.json $P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter --rig --tag _v2l
```

   That one writes `headroom_lidar_gps_birds_hunter_rig_v2l.json`.

3. Train the learner (2-3 hours per seed; copies the planner, then PPO):

```bash
$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --seeds 0 1 2
```

   Writes `rl_<tag>_s<k>_{greedy,sampled,mean,last}.pt` and `rl_*_corridor_0_1_2.json`. Use the same extra flags as step 2 (`--birds --hunter --rig --tag _v2l`, `--fog`, `--unmapped`, `--cameras`). `--no-clearml` turns live logging off.

4. Can the network hold the planner at all (a clone, flown greedily):

```bash
$P clone_check.py
```

   Writes `clone_check.json`.

5. Figures (from saved results only; nothing is re-simulated):

```bash
$P paper_figs.py world headroom learning birds rig gazebo
```

   Writes `../../outputs/paper/figs/<name>.pdf` and a png preview.

6. Why flights still fail, per condition (descriptive only):

```bash
$P why_cond.py
```

7. Sensitivity of the ladder and the learners to the simulator's assumptions:

```bash
$P sens.py clear mytry
```

   Writes `sens/clear_mytry.json` (do not reuse `base`: that is a saved result). `sens/run_all.sh` runs every variation.

The deciders' chains that ran these for the deck are in `conditions/*/queue/` and `conditions/magpie/rig_*.sh`; their
headers say which ladders and trainings, in what order, and why.

## 6. ClearML

`urban_rl.py` logs live to project `rl-landing/fast_sim`, one task per seed. Imported older logs go to
`rl-landing/fast_sim/history`:

```bash
$P clearml_import.py --dry-run --only fast_sim
```

   Once everything is already there, expect `would import: 0 to rl-landing/wind/history, 0 to rl-landing/fast_sim/history`. `~/clearml.conf` holds the credentials and is never printed or committed.

## 7. Gazebo flights use the same deciders

`decider_server.py` runs under the conda python (PyTorch lives there; PX4's and Gazebo's python bindings live in the
system python). `../px4_gazebo/gz_learner.py` starts it with the working folder set to here, so `--ckpt` names a file in
this folder. Gazebo reports what Gazebo has; helicopters, other drones, people and cars are reported absent.

## 8. What is in git, and where results go

- In git: all code, `sigma_pool.npz` (GeoAnchor's error model), `logan_osm.json`, `ms_logan.geojson`, every `headroom_*.json` and `rl_*.json` result, `rig_calib*.json`, `sens/*.json`, `why_cond.json`, and two checkpoints: `clone_corridor.pt` and `rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt` (the learner the Gazebo flights use).
- Not in git, rebuilt by section 3: `DATA_2462013.zip`, `lidar_*.npz`, `logan_map*.npz`.
- The other `rl_*.pt` checkpoints are on disk here only (`*.pt` is ignored).
- Logs and pictures: `../../outputs/fast_sim/`.
