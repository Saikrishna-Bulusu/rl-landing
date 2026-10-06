# Magpie: dodging a bird that keeps aiming at you

A territorial magpie nests near about half the pads and dives at the drone from behind (August to November in Queensland).
It keeps steering during the dive. Its two guesses are accepted (Sai, 25 Sept): it stops correcting 0.3 s before it arrives
(`HUNTER_T_COMMIT`) and it turns at 2 g (`A_TURN` 20 m/s^2). No bird result counts unless the drone sees the bird through
a camera rig, so the same bird is run three ways: handed the truth (`omni`), seen by the drone's own two cameras
(`--cameras`), seen by a six-camera rig through a detector and tracker (`--rig`).

Two worlds, same bird: the **fast simulator** (GPU, thousands of flights, `shared/fast_sim/`) trains and measures;
**Gazebo with PX4** (real pictures, one flight at a time) checks that the fast simulator told the truth.

Setup used below (run `P=...` once in each shell):

```bash
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

* Fast-sim steps run **from `shared/fast_sim/`** (they read maps, checkpoints and calibrations by bare name).
* Gazebo steps run **from the repo root**, one Gazebo at a time (see the root README, rule 2).
* Check the machine first: `free -g`, `nvidia-smi`, `cat /sys/class/power_supply/A*/online` (must be 1).
  A ladder needs about 3.3 GB of GPU, a training about 2 GB, the rig flights 1.6 GB.

---

## 1. How much room is there? The headroom ladder

Rule against the best same-sensor planner against an all-knowing planner, on the same 4,096 flights (corridor and random
worlds). Run it before any training: if there is no room, there is nothing to learn.

```bash
cd shared/fast_sim
```

1. The bird handed to the decider as exact truth (the upper bound). A few minutes to under an hour (old notes say 30-60 min).

```bash
$P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter
```

   Writes `headroom_lidar_gps_birds_hunter.json`. Expect "guard: ... the bound is valid" and a ladder near +0 pp.

2. The bird seen through the rig's detector and tracker.

```bash
$P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter --rig
```

   Writes `headroom_lidar_gps_birds_hunter_rig.json`.

3. A different rig: point `RIG_CALIB` at its calibration and name the output with `--tag`.

```bash
RIG_CALIB=rig_calib_v2l.json $P urban_headroom.py --map logan_map_lidar.npz --nav gps --birds --hunter --rig --tag _v2l
```

   Writes `headroom_lidar_gps_birds_hunter_rig_v2l.json`. Calibrations on disk: `rig_calib.json` (first rig),
   `_half` (detector needs twice the pixels), `_yolo`, `_v2`, `_v2t`, `_v2u`, `_v2l`. The own-cameras bound is `--cameras` instead of `--rig`.

4. How late the bird stops correcting changes everything. CPU only, 10 points (5 reaction times, truth and rig).

```bash
$P ../../conditions/magpie/hunter_sweep.py
$P ../../conditions/magpie/hunter_sweep.py --plot ../../outputs/fast_sim/hunter_sweep.png
```

   Writes `hunter_sweep.json`. Without any dodging 12% of dives touch the drone; at 0.3 s the rule is struck on 30% of
   flights with the truth and 40% through the rig.

5. The bar chart of all sensing modes.

```bash
$P ../../conditions/magpie/plot_sensing.py --out ../../outputs/fast_sim/sensing.png
```

Struck flights among magpie flights (corridor, 4,096 flights; best same-sensor planner and best all-knowing planner, from the json files):

| bird sensing | rule | planner | all-knowing | headroom over rule |
|---|---|---|---|---|
| truth | 27.7% | 30.0% | 30.9% | -0.2 pp (none) |
| 360-degree rig (first CNN detector) | 38.9% | 38.8% | 37.7% | +2.5 pp |
| rig, half the detection range | 41.6% | 38.2% | 37.5% | +3.5 pp |
| rig v2l (up camera, YOLO calibration) | 38.9% | 36.3% | 34.1% | +3.4 pp |
| own two cameras | 45.1% | 39.9% | 38.0% | +3.9 pp |

The room is small, which is why the next step is a copy check and not a long training.

## 2. Train the learner against the hunter

The learner starts as a copy of the planner (it sees the planner's forecast for each of 14 manoeuvres), then PPO looks
for what the planner misses. ClearML logs live to project `rl-landing/fast_sim` (task `rl_eyes_lidar_gps_birds_hunter_rig_corridor_s<k>`).
`--no-clearml` turns that off.

```bash
cd shared/fast_sim
```

1. Train three seeds. About 2-3 h per seed. Run seeds 0 and 1 together, seed 2 fifteen minutes later (the start takes ~1 GB extra).

```bash
$P urban_rl.py --world corridor --gens 200 --eyes --map logan_map_lidar.npz --nav gps --birds --hunter --rig --seeds 0 1 2
```

   Writes `rl_corridor_eyes_lidar_gps_birds_hunter_rig_s<k>_{mean,sampled,greedy,last}.pt` and
   `rl_eyes_lidar_gps_birds_hunter_rig_corridor_<seeds>.json` (log, each read-out's test result, paired against rule and planner).
   For the up-camera rig add `--tag _v2l` and set `RIG_CALIB=rig_calib_v2l.json`.

2. Pair the three seeds flight by flight against the best same-sensor planner. About 5 min.

```bash
$P ../../conditions/magpie/pair_magpie.py
```

   Writes `pair_magpie.json`. Expect "CHECK vs headroom run: shield 69.775 expected, safeplan 73.51 expected". Result: the learner is
   at 77.5 / 78.2 / 77.0% clean, +3.5 to +4.8 pp over the best planner and +7.2 to +8.5 over the planner it copied.

3. What the learner actually does (top-down film, CPU only).

```bash
CUDA_VISIBLE_DEVICES="" nice -n 15 $P ../../conditions/magpie/video_birds.py --seed 71 --n 256
```

   Writes `birds_video_h264.mp4` in this folder: move it to `outputs/films/` (films are never committed).

The learner's gain is more deliveries around the bird, not fewer strikes (1,024 flights: struck 39.2% against the rule's 38.3%).

## 3. The rig bench (Gazebo, static drone)

Measure what the cameras see before trusting them. The bench is a drone standing still in its own Gazebo partition.
The rig variants (`RIG=v2` gimbal, `v2f` fixed camera, `v2t` back camera tilted up, `v2u` plus a top camera, `v2l` up camera
with the back camera level) are built once.

1. Build the rig models and worlds, once.

```bash
/usr/bin/python3 shared/px4_gazebo/gz_rig.py --setup
/usr/bin/python3 shared/px4_gazebo/gz_unified.py --setup
/usr/bin/python3 shared/px4_gazebo/gz_learner.py --setup
```

2. Start the bench. It prints a PID: write it down.

```bash
RIG=v2l /usr/bin/python3 shared/px4_gazebo/gz_rig.py --bench
```

3. Coverage: the magpie at 160 directions and 6 ranges, which camera sees it and how many pixels. About 15 min.

```bash
RIG=v2l /usr/bin/python3 shared/px4_gazebo/gz_rig.py --cover shared/px4_gazebo/gz_logan/rig_v2l/cover
```

   Expect "camera model check" near 1 pixel (median). Add `--report DIR` for the chart, `--views DIR` for the six-view sheet.

4. Labelled pictures (Gazebo labels the bird itself: render with and without it, difference). About 25 min.

```bash
RIG=v2l /usr/bin/python3 shared/px4_gazebo/gz_rig.py --data shared/px4_gazebo/gz_logan/rig_v2l/data --poses 600
```

5. Empty places over the pads as negatives (what stops the detector boxing pad markers).

```bash
RIG=v2l /usr/bin/python3 shared/px4_gazebo/gz_rig.py --negatives shared/px4_gazebo/gz_logan/rig_v2l/data
```

6. Stop the bench (the GPU is needed for training).

```bash
kill <bench PID>
```

Coverage in the first rig: the magpie is about 80 px at 5 m, 20 px at 20 m, 10 px at 40 m, 6-7 px at 60 m, every direction covered.
Do not put ring cameras close above the body: they saw the propellers as small birds. They sit on a 30 cm mast, the down camera on a boom.

## 4. The detector (YOLO)

Fine-tuned Ultralytics YOLO26n on the bench pictures, scored on drone places it never trained on, with the images made
worse (noise, blur, JPEG, brightness). Needs `PYTHONPATH=.yolo_pkgs` so that the training environment's numpy and torch stay untouched.

1. Make the YOLO-format set. A few minutes.

```bash
RIG=v2l PYTHONPATH=.yolo_pkgs $P conditions/magpie/rig_yolo.py --data shared/px4_gazebo/gz_logan/rig_v2l/data
```

2. Fine-tune from the weights already in use. 30 epochs.

```bash
RIG=v2l PYTHONPATH=.yolo_pkgs $P conditions/magpie/rig_yolo.py --train --epochs 30 --init $PWD/shared/px4_gazebo/gz_logan/rig_v2/yolo/run/weights/best.pt
```

3. Score it: found and false alarms per picture, by size.

```bash
RIG=v2l PYTHONPATH=.yolo_pkgs $P conditions/magpie/rig_yolo.py --score shared/px4_gazebo/gz_logan/rig_v2l/data
```

   Writes `score_yolo.json` and `detect_vs_size_yolo.png` in the data folder.

4. Write the calibration the fast simulator flies with (found rate by pixel size, distance error from size).

```bash
RIG=v2l /usr/bin/python3 shared/px4_gazebo/gz_rig.py --calib shared/px4_gazebo/gz_logan/rig_v2l/cover shared/px4_gazebo/gz_logan/rig_v2l/data --score score_yolo.json --out shared/fast_sim/rig_calib_v2l.json
```

5. Retrain on photo ground (about 2 h, one command). It fixed "trees seen as magpies": every earlier picture was plain Gazebo ground, and over the photo floor
   (`gnss_denied/`) the detector boxed crowns and roofs. Nothing else may fly while it runs.

```bash
conditions/magpie/queue/yolo_photo.sh
```

   Expect old against new in `outputs/queue/yolo_photo.log`: found 75% to 81%, false boxes 0.57 to 0.002 per picture. Use `rig_v2/yolo/run_geo/weights/best.pt`.

Results: YOLO at threshold 0.5 finds 88% with 0.0015 false alarms per picture (the small CNN `rig_detect.py` found 86.5% with 0.0885).
Small birds are the weak spot (4-7 px: YOLO 27%, CNN 64%). The CNN is still here and trains with
`$P conditions/magpie/rig_detect.py --train <data folder>` then `--score`. Timing in Gazebo on this GPU: about 54 ms per
set of six pictures (`results/detector_latency_2026-09-30.json`).

## 5. Magpie flights in Gazebo with PX4

`rigfly.sh` starts PX4 and Gazebo itself for each run, flies every seed and decider in turn, and stops them. It runs at 0.1x
real time (six HD cameras), so one flight is long (the script times out at 1 h). `SENSE=rig` means the decider knows the bird only
through the cameras; `SENSE=omni` hands it the truth. Results go to `outputs/gazebo_rig/<OUT>/` (`flights.jsonl`, one film
and one `*_rig.jsonl` per flight, `failsafes.txt`).

1. The planner through the rig, three seeds.

```bash
OUT=planner_rig SEEDS="0 1 2" DECIDERS=planner SENSE=rig RIG=v2l ./conditions/magpie/rigfly.sh
```

2. The learner through the rig.

```bash
OUT=learner_rig SEEDS="0 1 2" DECIDERS=learner SENSE=rig RIG=v2l CKPT=rl_corridor_eyes_lidar_gps_birds_hunter_rig_v2l_s0_mean.pt ./conditions/magpie/rigfly.sh
```

3. The same seeds handed the truth, for comparison.

```bash
OUT=planner_omni SEEDS="0 1 2" DECIDERS=planner SENSE=omni ./conditions/magpie/rigfly.sh
```

4. Does the fast simulator's rig behave like the real thing? Tracked share, position and direction error by distance.

```bash
$P conditions/magpie/rig_compare.py outputs/gazebo_rig/planner_rig_v2l_final/planner_s0_rig.jsonl
```

5. Test a tracker change without flying: replay a flight's logged detections.

```bash
RIG=v2l $P conditions/magpie/rig_replay.py outputs/gazebo_rig/planner_rig_v2l_final/planner_s0_rig.jsonl
```

Two or more runs can fly side by side with `INSTANCE=N` (its own Gazebo partition, PX4 port 14540+N); never two with the same number.
Check `failsafes.txt`: 0-2 at the end is normal (disarm), more means the loop fell behind.

The plain-ground magpie (no rig, bird handed to the decider) and its films:

```bash
./conditions/magpie/gazebo_birds_v2.sh
```

   Needs the SITL up at 0.1x first (the command is in the script's header). The eight earlier flights landed 8/8, 0.20-0.99 m from centre.

What the flights showed (seeds 0-2, hunter magpie, 25-26 Sept):

| run | outcome |
|---|---|
| planner, truth | 3/3 landed, about 3 dives each |
| planner, rig, first detector | 2 struck, 1 landed off the pad |
| planner, rig, YOLO | 1 timeout, 2 struck |
| planner, rig v2l, final landing | 1 landed, 2 struck |
| learner, rig v2l, final landing | 2 landed, 1 struck |

Three seeds a row is a demonstration, not a rate. The open problem is phantoms: with the rig the tracker sometimes follows
clutter (pad markers, shadows, roof edges) when the real bird is far, and the fast simulator's scattered false alarms
do not reproduce that. Say so wherever a rig result is quoted.

## 6. Chains that already ran (`queue/`, plus the `rig_*.sh` and `gz_*.sh` files here)

Each header says what it waits for and what it writes. Their gates are markers and `rig.log` in `outputs/queue/`.
Read before re-running: they use absolute PIDs from the night they ran.

| file | what it did |
|---|---|
| `rig_v2.sh`, `rig_negatives.sh`, `rig_detector_run.sh` | the first mast rig: coverage, pictures, detector, calibration (and its `_half` twin) |
| `rig_queue.sh` | headroom ladders: truth, own cameras, rig, rig at half range |
| `rig_train.sh`, `rig_after.sh` | the rig learner, 3 seeds (two GPU slots) |
| `gz_compare.sh`, `gz_pairs.sh` | Gazebo flights, rig against truth, and one pair film per seed |
| `queue/rigv2_cover.sh`, `queue/rigv2_detector.sh`, `queue/rigv2_learn.sh`, `queue/rigv2_fly.sh` | rig v2 (gimbal): stages A to D |
| `queue/variants.sh`, `queue/variant_ladders.sh`, `queue/v2l_bench.sh` | v2t, v2u, v2l benches, calibrations and ladders |
| `queue/rig_pipeline.sh VARIANT` | one variant through bench, detector, ladder, learner, flights |
| `queue/after_yolo.sh`, `queue/yolo_flights.sh` | score YOLO, then fly the planner with YOLO in the loop |
| `queue/yolo_photo.sh` | the photo-ground retrain (section 4) |
| `queue/magpie_final.sh` | six final flights on the up-camera rig, planner and learner |

## 7. What came out of it

* With the truth the reflex alone is enough: no planner or learner beats the rule (-0.2 pp). Birds are a speed problem first.
* Through the rig, room appears (+2.5 pp over the rule) and the learner takes it: +3.5 to +4.8 pp over the best planner (section 2).
* The up camera doubled how often a bird above the drone is truly seen (20% to 40%) and cut phantom alarms from 54% to 11-18%.
* The fast simulator's rig is the CNN's calibration unless you set `RIG_CALIB`; swapping detectors means recalibrating, then re-running ladders and training.

## 8. Things that bit us

* Never `pkill -f "gz sim"`: it kills the bench in the other partition. Stop by PID.
* PX4 skips `PX4_SIM_SPEED_FACTOR` when it attaches to a pre-placed drone: `rigfly.sh` sets the real-time factor itself; without it PX4's failsafe landed the drone in place.
* The bench must render on the NVIDIA chip (hybrid laptop); on the Intel chip it ran at 0.12x.
* Let the flight loop place the magpie, not the video thread, or the cameras photograph a stale bird.
* Frame check: `rig_dir_err_deg` must stay under 3 degrees before any live result is believed.
