# Weather: fog and mist (and a little wind)

Does the landing survive bad weather? Two answers, one from each simulator:

* **Gazebo (real PX4, real pictures):** five skies, ten Logan deliveries each. Wind is harmless. Fog costs landings,
  but the drone gives up safely instead of crashing.
* **Fast simulator:** fog switched on for every flight (`--fog`). The pad marker is hidden in drifting patches, as often
  as Gazebo's fog hid it. The rule falls from 94% to 69% clean deliveries; a learned decider wins back 85-90%.

Rain was tried and dropped: Gazebo's particle spray ignored every rotation meant to point it down, and real rain's
camera effects (water on the lens, lost contrast) are not particles anyway. Fog covers visibility.

Everything runs from the repo root, `/home/sai/RL_projects/rl-landing`. `P` is the conda python:

```bash
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

Only one Gazebo at a time. Check the machine first (`nvidia-smi`, `free -g`, `cat /sys/class/power_supply/A*/online`
= 1).

## The five skies (`gz_weather.py`)

| sky | what |
|---|---|
| calm | the Logan world as flown so far (the baseline, same seeds) |
| breeze | 3 m/s from the south-east, swelling +-20% (Beaufort 2) |
| gusty | 7 m/s from the south-east, gusting to ~10 m/s (Beaufort 4) |
| mist | calm; thin particle fog over the delivery area, the camera looks through it |
| fog | calm; thick fog: from 60 m up, roofs 60 m away are barely visible |

Wind reaches the x500 through its propellers (Gazebo's rotor drag against the air), so it shows as lean, not as a
second force. Not modelled: water on the lens, rain on the rotors, any GPS or IMU effect.

## A. Fog and wind in Gazebo

1. Build the five worlds (seconds; once):

```bash
python3 conditions/weather/gz_weather.py
```

Expect: `shared/px4_gazebo/gz_logan/weather/<sky>/aruco.sdf` and `server.config` for each sky, plus `fog.png` and
`rain.png` textures. Each sky has its own `server.config` because a plugin in the world would silently drop PX4's own
systems.

2. Fly ten deliveries under one sky (about an hour: the simulator runs at 0.16-0.3x real time, up to 8 min of wall
   clock per flight; seeds 0..9 in every sky, so the skies pair flight for flight; flight 0 is filmed):

```bash
./conditions/weather/run_weather.sh fog 10
```

Expect: `outputs/weather_runs/fog/` with `m_<i>.json` (one result per flight), `trace_<i>.csv`, `missions.log`,
`px4.log`, `rtf.txt` (sim speed), `flight0.mp4`; at the end `outputs/queue/fog.done` saying
`fog: 10 of 10 flights recorded`. The script KILLS every PX4 and Gazebo running on the machine before it starts (and
again at the end): do not start it while anything else is flying. Repeat for `calm`, `breeze`, `gusty`, `mist`.
(The flight is the Part 2 mission: the drone is told roughly where the pad is, with a simulated position error, flies
there, finds the marker and lands; `shared/px4_gazebo/px4_mission.py`.)

3. Summarise all skies:

```bash
python3 conditions/weather/gz_weather.py report
```

Expect: `conditions/weather/weather_report.json` and a table per sky: flights, landed on the pad, outcomes, median miss,
the altitude of the first marker read, the tilt it took to hold station while centring, and the sim time.

What it found (10 flights each, 24 Sept):

| sky | landed on the pad | median miss | tilt while centring (median) |
|---|---|---|---|
| calm | 10 of 10 | 0.37 m | 0.8 deg |
| breeze | 10 of 10 | 0.35 m | 1.9 deg |
| gusty | 10 of 10 | 0.35 m | 4.7 deg |
| mist | 9 of 10 (1 safe abort) | 0.38 m | 1.2 deg |
| fog | 7 of 10 (3 safe aborts) | 0.38 m | 0.5 deg |

The marker is first read about 50 m up in every sky. When the fog hides the pad the drone aborts rather than landing
blind, and the landings it does make are as accurate as in clear air.

## B. Fog in the fast simulator

Run these from `shared/fast_sim/` (the maps and checkpoints are read by bare name there):

```bash
cd shared/fast_sim
```

How fog works there: a drifting patch hides the ground now and then. The share of time the pad marker is still read
(1.0 at 5 m, 0.92 at 15 m, 0.77 at 25 m, 0.48 at 35 m, 0.36 at 46 m up) was measured in Gazebo's fog (24 Sept, 20
flights); patches last about 2 s. Assumed, not measured: the LiDAR survey and depth map are not degraded by fog.
`$P urban_env.py` runs the simulator's self-checks; check 8 re-measures that share at each height.

1. How far is the rule from a perfect decider, in fog (30-60 min):

```bash
$P urban_headroom.py --map logan_map_lidar.npz --nav gps --fog
```

Expect: `headroom_lidar_gps_fog.json` and the ladder printout. Clear versus fog, corridor world, clean deliveries:

| decider | clear | fog |
|---|---|---|
| rule | 93.7% | 68.8% |
| best planner using only what the sensors show (the shield planner) | 94.5% | 71.9% |
| all-knowing planner (told every intruder's true state; the better of its two forms) | 95.4% | 95.7% |

Fog is an information problem: a planner that knows everything is not hurt by it, the ones that must see are.

2. Train the learner in fog (copies the planner, then PPO; 2-3 h a seed; add `--no-clearml` to switch logging off):

```bash
$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --seeds 0 1 2 --fog
```

Expect: `rl_corridor_eyes_lidar_gps_fog_s<k>_{mean,sampled,greedy,last}.pt` and
`rl_eyes_lidar_gps_fog_corridor_0_1_2.json`. Scored on the ladder's 4,096 flights: 85.5%, 89.5%, 85.3% clean for seeds
0, 1, 2 (the planner it copied: 71.9%). The training logs are `outputs/fast_sim/rl_cond_fog*.log`.

3. How much the fog result depends on the simulator's assumptions (no retraining; one condition per call):

```bash
$P sens.py fog base
```

Expect: `sens/fog_base.json`. The other variations (`SENS_LEAVE=10,120`, `SENS_T_MAX=400`, `SENS_DRONE_SEP=50`) are in
`sens/run_all.sh`.

4. The figure (fog bars, with the other conditions) and the Gazebo-sky bars:

```bash
$P paper_figs.py headroom gazebo
```

Expect: `outputs/paper/figs/headroom.pdf` and `gazebo.pdf` (the latter reads `weather_report.json` from step A3).
`plot_conditions.py --out conditions.png` draws the learning curves of every condition beside the clear baseline.

Fog together with a magpie and unmapped structures is in `all_conditions/README.md`.
