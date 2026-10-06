# archive/: the early studies

Three studies that came before the wind, magpie and GeoAnchor work. They are finished, but the code still runs.

| Folder | What it was | Its answer |
|---|---|---|
| `sigma/` | Part 1. Does GeoAnchor's own uncertainty (sigma, in metres) help a learned landing policy? Fast point-mass simulator. | No. Sigma gave exactly nothing over no channel (+0.1 points, p=0.97, 64 seeds); the staleness stopwatch hurt (-5.9, p=0.012); a tuned hand rule beat every learned arm. |
| `part2/` | Part 2. Trees, a bigger marker, the forest on real PX4. Most of its code lives elsewhere (see section 2). Only the matcher test is here. | A bigger pad is far safer among tall trees: in the fast study 1.20 m crashed 44.8% and 2.40 m 0.0%, because a small marker cannot be decoded from above a tall tree. The real-PX4 forests compare 1.80 m with 1.20 m. |
| `tello/` | The Tello cage: the same last-metres problem at toy scale, in Sai's 3 x 2 x 2 m cage. | Rule 57%, planner 91%, learner 93%, all-knowing 93% of 4,096 flights delivered. |

`sigma/HANDOFF.md` is the full story of part 1 with its run log (long; the headline is near its middle).

## Two rules for everything here

1. **Run from the study's own folder.** These scripts read and write next to themselves with bare file names. Use `cd archive/sigma` or `cd archive/tello` first. Their shell drivers already `cd` there themselves.
2. **`sigma_pool.npz` is a link.** The one tracked copy lives in `shared/fast_sim/`; `sigma/` and `tello/` each hold a link to it. Writing to that name replaces the shared file, so give new pools a new name.

Python: `P=/home/sai/miniconda3/envs/mujoco_drones/bin/python`. System `python3` is fine for the numpy-only scripts.
Training here is CPU only (`CUDA_VISIBLE_DEVICES=""` in the drivers). Before a long run, check the machine: `free -g`, `df -h`, AC power.

---

## 1. Part 1: sigma (`archive/sigma/`)

Start here:

```bash
cd archive/sigma
```

1. Turn GeoAnchor's results into the sigma pool and the calibration numbers (1 minute). Expect ENCE 0.313 and a conformal k near 1.97 at 90% coverage. Writes `sigma_pool_new.npz` and `sigma_pool_new_report.json`.

```bash
python3 calibrate.py ~/GeoAnchor/results/s8_train --exclude-scene Scene_21 --out sigma_pool_new.npz
```

2. Run the scripted baselines. This is the decision gate: a learned policy has to beat these (10 minutes). Writes `baselines.json`.

```bash
python3 baselines.py --pool sigma_pool.npz --tune --episodes 400
```

3. Train the three arms, none, staleness and sigma (30 to 60 minutes per arm, about 2 GB RAM). Writes `runs/ablation.json`.

```bash
$P train.py --pool sigma_pool.npz --all --steps 2000000 --n-envs 8
```

4. Compare `runs/ablation.json` with `baselines.json`. The claim lives in the high-sigma column.
5. The powered version, with trees and 8 or more seeds (overnight). It is resumable: rerun with `resume` after a crash and only the missing runs start. It skips a job when under 3 GB RAM is free. Output goes to `overnight/`.

```bash
bash run_overnight.sh
```

6. Optional: watch it hourly. This one restarts stalled jobs, so start it only when you mean it.

```bash
bash watchdog.sh
```

7. Read the finished overnight runs (`overnight/`) against the rule and the ceiling.

```bash
$P morning_report.py
```

8. Optional, and **not part of this study's own run**: the same missions on real PX4. This needs a running PX4 plus Gazebo, one at a time, from the repo root (about 5 minutes per mission).

```bash
./shared/px4_gazebo/launch_px4.sh
```

```bash
MISSION_JSON=outputs/px4_mission.json python3 shared/px4_gazebo/px4_mission.py --runs 5
```

`MISSION_JSON` keeps the result file out of the repo root; without it `px4_mission.py` writes `px4_response.json` into whatever folder you ran it from.

### Before any long run

```bash
$P preflight.py --env approach
```

It proves the agent is really in the control loop (the lesson of this study). Use `--env cage` for the cage.

### What else is in `sigma/`

| Files | What they are |
|---|---|
| `approach_env.py` and `_v2` to `_v5` | The landing problem, five versions (v5 adds trees). `outdoor_env.py` is the MuJoCo x500 version of the task. |
| `train.py`, `train_v2.py` to `train_v5.py` | The matching trainers. `train_v5.py` is the one the overnight run uses. |
| `rule_agent.py`, `headroom.py`, `head_to_head.py`, `paired_eval.py` | The hand rule, how much any decider could gain, and paired comparisons on identical episodes. |
| `sigma_v2.py`, `make_pool_v2.py`, `sigma_vs_trees.py` | A better sigma predictor, its pool, and the test of sigma among trees. |
| `clone_teacher.py`, `rl_left.py`, `sensing_*.py` | Can the policy copy the rule, is anything left for RL, does better obstacle sensing help. |
| `x500*.py`, `render_x500*.py`, `compare_physics.py` | The measured x500 in MuJoCo, and whether it changes the answer. |

### Where the results are

| Where | What |
|---|---|
| `sigma/results/*.json` | Baselines at three error correlations, headroom, head-to-head. Small, in git. These are the evidence. |
| `sigma/results/{final,final07,sweep,sweep07,...}/` | Per-seed `ablation.json` files of the finished ablations. In git. |
| `sigma/pools/` | The sigma pools from other matchers and quantiles, used by the matcher test. |
| `HANDOFF.md` | The story, the retractions, the conclusions. |
| `outputs/runs_*`, `outputs/out_final`, `outputs/sv2`, `outputs/v5`, ... | Raw run folders (policies, logs). Not in git. |

---

## 2. Part 2: the marker and the forest (`archive/part2/` and friends)

The scripts for this part sit with the code they drive, not here:

| Where | Files |
|---|---|
| `archive/part2/` | `gazebo_matchers.sh`, `sweep_search_k.py`, `join_videos.py` |
| `conditions/static_obstacles/` | `bigger_marker.py`, `ab_pad.sh`, `run_forest20.sh`, `run_n20.sh`, `parse_forest.py` |
| `shared/px4_gazebo/` | `px4_mission.py`, `make_pad.py`, `make_trees.py`, `forest.py` |

Steps 2 to 4 start PX4 and Gazebo. **Run one at a time, with nothing else simulating.** Run all of them from the repo root.

1. How big must the marker be? A fast study, no Gazebo (5 minutes). Prints delivery rate against marker size. It re-runs the 1.20 m against 2.40 m comparison (44.8% against 0.0% crashes) with the measured detector.

```bash
python3 conditions/static_obstacles/bigger_marker.py
```

2. Old pad against new pad, the same five missions (30 minutes per arm). Seeds match, so only the paint changes.

```bash
./conditions/static_obstacles/ab_pad.sh old
```

```bash
./conditions/static_obstacles/ab_pad.sh new
```

3. Twenty different forests, both pads (about 3 hours). It switches the pad itself between arms. Writes `conditions/static_obstacles/results/forest18*` and `forest12*`, and logs to `outputs/logs/`. **It edits `shared/px4_gazebo/px4_mission.py` in place** (the tag sizes, with `sed`), once per arm. Afterwards run `git diff shared/px4_gazebo/px4_mission.py` and restore it if the diff is not what you intended.

```bash
./conditions/static_obstacles/run_forest20.sh
```

4. Does a better matcher deliver more? Three sigma pools through real PX4 with matched seeds (about 3 hours). Logs go to `gazebo_match/` at the repo root; `N=10` is the default number of missions.

```bash
./archive/part2/gazebo_matchers.sh
```

### Where the results are

| Where | What |
|---|---|
| `conditions/static_obstacles/results/forest12_0.json` to `forest12_19.json` | Delivered or crashed per forest, 1.20 m pad. In git. |
| `conditions/static_obstacles/results/missions_pad*.json`, `trees_pad*.json` | The five-mission pad comparisons and the tree layouts they flew. In git. |
| `conditions/static_obstacles/results/forest18*` | The 1.80 m arm of step 3. Not in git yet; it appears when step 3 runs. |
| `shared/px4_gazebo/CALIBRATION.md` | The detector's measured size ramp (0% at 10-15 px to 100% by 35 px). |
| `outputs/match*`, `outputs/px4run*` | Raw mission logs. Not in git. |

---

## 3. The Tello cage (`archive/tello/`)

A DJI Tello delivers to its customer's ArUco pad inside a 3 x 2 x 2 m cage, among model houses and trees, seeing the floor through a 45 degree mirror. The same ideas as the Logan work, at toy scale: a rule, a planner, and a learner copied from the planner and then trained with PPO. It runs as tensors on the GPU (or CPU).

Start here:

```bash
cd archive/tello
```

1. Check the cage and score the rule, the planner and the all-knowing planner on 2,048 flights (6 seconds). Expect `self-checks passed`, then rule about 57%, planner about 91%, all-knowing about 93% delivered.

```bash
$P cage.py
```

2. Run one delivery in the simulator, with no drone (1 second). Expect `simulated outcome: delivered`. The cage layout comes from `layout.json` if you make one, otherwise from `layout_example.json`.

```bash
$P fly.py --sim --decider planner
```

3. Same, with the trained learner.

```bash
$P fly.py --sim --decider learner --ckpt cage_rl_s0_greedy.pt
```

4. Train the learner (3 seeds, 120 rounds; long, GPU). Writes `cage_rl_s{0,1,2}_{greedy,sampled}.pt` and `cage_rl_0_1_2.json`. These files are already here, so this replaces them.

```bash
$P train.py --seeds 0 1 2 --gens 120
```

5. Redraw the data behind the paper's cage figure (a few seconds). Writes `cage_fig.json`.

```bash
$P figdata.py
```

6. The older MuJoCo version (the sigma ablation in `TelloLandingAviary`, which needs `~/RL_projects/MuJoCo-drones-gym`). The first drivers re-run the 3-million-step ablation, 8 jobs in parallel, CPU only, hours. They stop on their own if RAM falls under 3 GB or disk under 5 GB, and log to `outputs/logs/`. They write new `cage_sweep2/`, `cage_final2/`, `ctrl_*/` folders here and do not touch `results/`.

```bash
bash cage_rerun.sh
```

```bash
bash cage_control.sh
```

7. Render a trained policy into a GIF and a contact sheet. `--run` points at one finished run.

```bash
MUJOCO_GL=glfw $P render_cage.py --arm sigma --episodes 3 --run results/cage_final2/sigma_s0
```

### Flying the real Tello

Only with Sai in the cage. `fly.py` talks to the Tello over UDP. Check first, motors off, then the axes, then fly. ESC or Ctrl-C lands. Every `[knob]` in `cage.py` is a guess until measured in the cage.

```bash
$P fly.py --check-camera
```

```bash
$P fly.py --check-axes
```

```bash
$P fly.py --decider planner
```

### Where the results are

| Where | What |
|---|---|
| `cage_rl_0_1_2.json`, `cage_fig.json` | The learner's validation curves and the ladder on 4,096 flights. In git. |
| `cage_rl_s*_{greedy,sampled}.pt` | The three learners' weights. |
| `results/cage_final2/`, `results/cage_sweep2/` | The 16-seed, 3-million-step sigma ablation after the action-space fix. |
| `results/cage_final/`, `results/cage_sweep/` | The first run, wasted by an action-space bug (84 runs). Kept as history. |
| `results/ctrl_final/`, `results/ctrl_sweep/` | The control arms (`random`, `clock`) that showed the staleness gain was only a clock. |
| `media/` | Three episode GIFs with contact sheets. |
| `queue/tello_train.sh` | The overnight chain that trained the learner after the YOLO fine-tune. |
| `outputs/tello/` | Raw flight logs. Not in git. |

The cage's sigma ablation ended like part 1: a calibrated sigma was not distinguishable from a meaningless input of the same shape (+1.7 points, p=0.40, 16 seeds).
