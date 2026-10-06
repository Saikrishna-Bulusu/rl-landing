# All conditions at once

Every condition in one flight, to test that the whole stack works together. This is an integration check, not a
landing rate: each run is one flight, and the result depends on where the crane and the magpie happen to be.

| file | what it is |
|---|---|
| `queue/stress_30sep.sh` | the full-stack stress test in PX4 + Gazebo (below) |
| `stress_report.py` | reads every stress flight's own files and writes the table `results/stress_2026-09-30.json` |
| `queue/land_gate.sh` | the 26 Sept chain that re-flew the crane landings and the live magpie flights after the landing gate changed |
| `queue/overnight_24sep.sh` | the 24 Sept chain that ran magpie flights, headroom ladders and trainings for several conditions in one night |

To pick your own combination of conditions instead, use [`../mix_and_match/`](../mix_and_match/README.md).

## The stress test (PX4 + Gazebo)

One delivery with everything switched on:

- GNSS denied once GeoAnchor's visual fixes are in PX4's loop (the 4-fixes-in-15-s rule, see [`../gnss_denied/`](../gnss_denied/README.md))
- wind of W10 = 9 m/s from a direction drawn per seed (the WindDrag plugin, see [`../conditions/wind/`](../conditions/wind/README.md))
- the unmapped crane on the approach, at a spot drawn per seed (see [`../conditions/static_obstacles/`](../conditions/static_obstacles/README.md))
- the hunter magpie, known only through rig v2l's live YOLO detector and tracker (see [`../conditions/magpie/`](../conditions/magpie/README.md))
- the magpie-trained learner deciding on the voxel map with the local 3D planner, then the marker landing
- no flying learner (Sai's choice, 30 Sept)

Before you start: one simulator at a time, the laptop on AC power (the script refuses to fly on battery), and the
setup in the root README done once.

1. One smoke flight (about 15 to 25 minutes, the detector slows the simulation):

   ```bash
   all_conditions/queue/stress_30sep.sh 0
   ```

   Output: `outputs/gazebo_geo/stress_e0/` with the PX4 log, the film, `run.log` and `manifest.json` (the git commit and
   the SHA-256 of every file the flight used).

2. More seeds, one after another (it touches `outputs/queue/stress_30sep.done` after the last one):

   ```bash
   all_conditions/queue/stress_30sep.sh 1 2 3 4 5 6 7 8 9
   ```

3. The table, from the flights' own files:

   ```bash
   python3 all_conditions/stress_report.py
   ```

   Output: `all_conditions/results/stress_2026-09-30.json`, one row per flight plus totals.

**Status (6 Oct 2026).** Seeds 0 to 7 were flown on 30 Sept (`outputs/gazebo_geo/stress_e0` to `stress_e7`); seeds 8
and 9 were never flown, and the report was never run on the full set. The ACRA draft does not use these flights.

## All conditions in the fast simulator

The fast simulator mixes conditions with flags, one learner per combination (run from `shared/fast_sim/`, see
[`../shared/fast_sim/`](../shared/fast_sim/README.md)). For example the learner trained with the magpie, fog and the
unmapped crane together:

```bash
cd shared/fast_sim
/home/sai/miniconda3/envs/mujoco_drones/bin/python urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --birds --fog --unmapped --seeds 0 1 2
```

The fast simulator has no wind. Wind exists only in the MJX twin and in Gazebo.
