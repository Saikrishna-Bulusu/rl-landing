# Wind: learning to land in gusty wind

The last 30 m of a delivery in a gusty wind. A rule lands the drone up to 12 m/s of wind at 10 m up (W10, the wind at
10 m height). Past that it waits and gives up. A learned flyer (the push it adds on top of the rule) and a learned
decider (the up/down choice every 0.04 s) land it in much stronger wind. It trains in a fast GPU copy of the drone
(MuJoCo MJX), then is checked in Gazebo with real PX4.

Sai, 26 Sept 2026: RL on gradually increasing wind; the learner in charge of the flying and the deciding; MuJoCo MJX;
all of it for ACRA (4 Oct). The wind paper comes next (`papers/wind/`, not started).

Everything below runs from the repo root, `/home/sai/RL_projects/rl-landing`. `P` is the conda python:

```bash
P=/home/sai/miniconda3/envs/mujoco_drones/bin/python
```

Training and evaluation need no Gazebo. Only step 8 does, and only one Gazebo at a time (`N=1`).

## 0. Run the wind condition alone

1. Check the machine first (training takes ~2 GB of the 8 GB GPU; Gazebo and training share the GPU badly):

```bash
nvidia-smi
free -g
cat /sys/class/power_supply/A*/online
```

Expect: no other job on the GPU, 10+ GB free RAM, `1` for AC power.

2. Check the MJX drone against PX4 in Gazebo's recorded step responses (~10 s, no Gazebo needed):

```bash
$P conditions/wind/x500_mjx.py --check
```

Expect: ten rows (N/E/U steps, then 8 m/s north), Gazebo and MJX 10/63/90% times within ~0.05 s of each other and
tilt within ~2 deg (the table in section 1), then two "steady wind" lines (tilt in 3-12 m/s wind, without and with body
drag). It reads `outputs/gazebo_wind/steps/steps.npz`. If that file is missing, one Gazebo flight records it:

```bash
OUT=steps ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u shared/px4_gazebo/px4_steps.py --out outputs/gazebo_wind/steps/steps.csv
```

3. The rule's ladder: 1,024 paired flights at each of W10 = 0, 3, 6, 9, 12, 15, 18, 20 m/s (~5 min):

```bash
$P conditions/wind/ppo.py --ladder rule
```

Expect: eight JSON lines (landed, crashed, hard, timeout per wind) and `conditions/wind/runs/rule_ladder.json`.
The rule lands 100% to 12 m/s, 78% at 15, 6% at 18, 0% at 20 (the table in section 3).

4. Train the flyer (flying only; the rule keeps its height and gate). The 8-seed `fly_hp` recipe (filters the push's
   slow part, 4 s) takes the seed as `--seed 0` ... `7` (~35 min a seed on the RTX 5070 Laptop):

```bash
$P conditions/wind/ppo.py --tag fly_hp --seed 0 --no-decide --hp-tau 4 --iters 1000 --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100
```

Expect: a line every 10 rounds (`it 130 W 11 band 0.97 landed ...`), `W` rising from 4 to 20 m/s, and every 100
rounds a validation line ending `BEST` when it improves. Files: `conditions/wind/runs/fly_hp_s0.{jsonl,npz}`,
`fly_hp_s0_it<N>.npz`, and `fly_hp_s0_best.npz` (the one to use).

   The paper's decider sits on a different flyer: `fly_s0_best.npz`, trained WITHOUT `--hp-tau` (no high-pass). To
   remake that flyer (its exact original settings were not recorded; these match the 8-seed recipe):

```bash
$P conditions/wind/ppo.py --tag fly --seed 0 --no-decide --iters 1000 --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100
```

5. Train the decider on the frozen flyer (the paper's: crash -100, hard landing -100, shaping at the learner's discount
   0.9998). About 11 min a seed:

```bash
$P conditions/wind/ppo.py --tag stage2_c100h100g --seed 0 --flyer conditions/wind/runs/fly_s0_best.npz --no-fly --lr 1e-4 --iters 600 --crash-r -100 --hard-r -100 --shape-g 0.9998 --w-ok 0.95 --envs 4096 --steps 64 --eval-every 100
```

Expect: `conditions/wind/runs/stage2_c100h100g_s0_best.npz`, validation (W 9/12/15/18) near 1.00/0.99/0.98/0.96.
The first crash-priced recipe (`stage2_c100`: crash -100, hard -30, default shaping) is the same command with
`--tag stage2_c100 --crash-r -100 --hard-r -30` and without `--shape-g`. Seeds 1 and 2: change `--seed`.
Add `--no-clearml` to any training command to switch logging off (see step 10).

6. Score a checkpoint on the ladder (~5 min); the result lands next to it:

```bash
$P conditions/wind/ppo.py --ladder conditions/wind/runs/stage2_c100h100g_s0_best.npz
```

Expect: `conditions/wind/runs/stage2_c100h100g_s0_best_ladder.json`; landed 99.8 / 98.2 / 98.6 / 96.2 / 85.8 % at W10
9 / 12 / 15 / 18 / 20 m/s (crashed 0.0 / 0.0 / 0.1 / 1.0 / 6.5 %).

7. Watch it: rule (left) and learner (right) fly the same yard in the same gusts, one video per wind (minutes, GPU;
   each wind is one 1,024-flight batch per policy):

```bash
$P conditions/wind/render.py
```

Expect: `outputs/wind/videos/wind_w9_yard<N>.mp4`, `wind_w15_...`, `wind_w18_...`, and a line per wind
(`W10 18: rule: TIMEOUT at 90.0 s ... | learner: LANDED at ... s, miss ... m`). Default learner is
`stage2_c100h100g_s0_best.npz`. Other options: `--ckpt`, `--w10 12 18`, `--episode 7`, `--frames <dir>` for stills,
`--clearml` to upload the clips (project `rl-landing/wind`).

8. The Gazebo check: the same yards, starts and gusts flown by PX4 in Gazebo. Once (worlds for 20 yards, then the wind
   plugin):

```bash
$P conditions/wind/export_yards.py --seed 10000 --first 40
python3 conditions/wind/gz_windland.py --setup --episodes 20
cmake -S conditions/wind/gz_winddrag -B conditions/wind/gz_winddrag/build
cmake --build conditions/wind/gz_winddrag/build
```

Expect: `outputs/gazebo_wind/yards_s10000.json` (40 yards; a yard is the same for any batch size), worlds in
`shared/px4_gazebo/gz_logan/windyard/e<i>/aruco.sdf`, and `libWindDrag.so` in `conditions/wind/gz_winddrag/build/`.
`--setup` writes only the worlds. The wind configs `w0/w9/w12/w15/w18.config` beside each world (mean speed, the
yard's own wind direction, seed = the speed) were made once and are not regenerated by any script here; a clean
clone has none (`gz_wind.config()` writes one config: `python conditions/wind/gz_wind.py --config w9 --mean 9 --dir <deg>`
puts it in `wind_v2/`, not in the yard's folder). Then one lane of flights (each flight ~6 min; it restarts PX4 and Gazebo for every flight, and
skips flights already in `flights.jsonl`, so a stopped ladder resumes):

```bash
N=1 EPISODES="$(seq 0 19)" WINDS="9 12 15 18" DECIDERS="rule conditions/wind/runs/stage2_c100h100g_s0_best.npz" OUT=ladder ./conditions/wind/gz_ladder.sh
```

Expect: one JSON line per flight in `outputs/gazebo_wind/ladder/flights.jsonl` (`episode`, `w10`, `decider`,
`outcome`), logs in `outputs/gazebo_wind/ladder/l0/`. Leave `N=1`: its default is 4 lanes, and two simulators at once
broke PX4 on this laptop (27 Sept). Smaller try: `EPISODES="0 1 2" WINDS="12"`.

9. Figures and diagnostics (each prints what it writes, under `outputs/gazebo_wind/figs/`):

```bash
$P conditions/wind/figs.py --ladder conditions/wind/runs/fly_hp_s0_best.npz conditions/wind/runs/stage2_c100h100g_s0_best.npz
```

Writes `mjx_vs_gazebo.png` (steps and hover tilt) and `ladder.png` (landed / failure kinds against wind, rule and
learners). Add `--gz outputs/gazebo_wind/ladder` to put the Gazebo flights beside them. Flight-by-flight agreement
of MJX and Gazebo:

```bash
$P conditions/wind/figs.py --pairs outputs/gazebo_wind/ladder conditions/wind/runs/rule_ladder.json
```

The other diagnostics (all take a checkpoint; see each file's docstring):

| script | question it answers |
|---|---|
| `trace.py runs/fly_s0_best.npz --w10 15 --episode 3` | one flight, rule against learner: height, distance, the push against the gust |
| `behaviour.py runs/<ckpt>.npz --w10 15 18` | what the learner does that the rule does not (height against time, vertical option by height) |
| `timing_check.py runs/<ckpt>.npz --w 9 18` | does it chatter (climb/descend reversals) when Gazebo's two delays come together |
| `rate_check.py`, `latency_check.py` | the same with one delay at a time (special cases of `timing_check.py`) |
| `cmd_replay.py <flight.csv> --yard 4 --w 9` | does MJX answer a Gazebo flight's vertical commands as Gazebo did |
| `estimate_check.py runs/<ckpt>.npz --w 9` | is the decider fragile to height and vertical-speed estimate errors (Gazebo's estimator is noisier than MJX's) |
| `overnight_report.py > queue/overnight_report.md` | the 26-27 Sept night: ladders per recipe and seed, Gazebo against rule |

These take their paths from the folder they sit in (run from the repo root as `$P conditions/wind/<script>.py ...`;
the `runs/` in a docstring means `conditions/wind/runs/`).

10. ClearML: `ppo.py` logs live to project `rl-landing/wind` (one task per run and seed: outcomes, return, curriculum
    wind, PPO losses, validation landed, speed). Past runs are in `rl-landing/wind/history` (and the fast simulator's
    in `rl-landing/fast_sim` and `rl-landing/fast_sim/history`). It reads `~/clearml.conf`. `--no-clearml` turns it off;
    if ClearML is unreachable training carries on without it. Import past logs (safe to repeat; already-imported tasks
    are skipped; run it with `--dry-run` first, it must end `would import: 0 to rl-landing/wind/history, 0 to rl-landing/fast_sim/history`
    when everything is already there):

```bash
$P shared/fast_sim/clearml_import.py --dry-run --only wind
```

## Which checkpoint is which

| checkpoint (`conditions/wind/runs/`) | what | in git |
|---|---|---|
| `fly_s0_best.npz` | the one frozen flyer under the paper's decider; no 4 s high-pass; `render.py`'s flyer | yes |
| `stage2_c100h100g_s0_best.npz` | the paper's decider: crash -100, hard -100, shaping at gamma 0.9998 (3 seeds: `_s0`..`_s2`) | yes |
| `fly_hp_s0_best.npz` | the filtered flyer, seed 0 (8 seeds `fly_hp_s0`..`s7`: a different recipe, with `--hp-tau 4`) | yes |
| `stage2_c100_s0_best.npz` | the first crash-priced decider (crash -100, hard -30) | yes |
| `*_ladder.json`, `rule_ladder.json` | the 1,024-flight ladder of each | yes (the `_best` ones) |
| everything else (`*_it<N>.npz`, `*.jsonl`, `*.out`, other seeds) | training history | no (`runs/` stays whole: scripts read it as one set) |

Other recipes tried (queue chains below): `stage2_c300` (crash -300), `stage2_c100h100` (hard -100 with default
shaping: it pulsed climb/descend at PX4's thrust limit and sank in Gazebo), `stage2_c100h100f02[g]` (a price 0.2 on each
reversal), `stage2hp_c100` (on the filtered flyer: worse).

## Results (MJX ladder, 1,024 paired flights per wind; Gazebo, 20 yards per wind)

Landed % (MJX), the crashed share in brackets where it matters:

| W10 (m/s) | 12 | 15 | 18 | 20 |
|---|---|---|---|---|
| rule | 100 | 78 | 6 | 0 |
| `fly_s0` (flyer only) | 100 | 99 | 46 | 4 |
| `fly_hp` (8 seeds, median) | 100 | 99 | 51 | 7 |
| `stage2_c100` (3 seeds, median) | 99 | 98 | 87 (3.3) | 64 (12.5) |
| `stage2_c100h100g_s0` (the paper's) | 98 | 99 | 96 (1.0) | 86 (6.5) |

Landed of 20 in Gazebo (`outputs/gazebo_wind/ladder_rule/flights.jsonl`, same yards and gusts for every decider):

| decider | 9 m/s | 12 | 15 | 18 |
|---|---|---|---|---|
| rule | 20 | 19 | 12 | 0 |
| `fly_hp_s0` | 20 | 20 | 18 | 6 |
| `stage2_c100` s0 | 20 | 19 | 19 | 15 (3 crashed) |
| `stage2_c100h100g` s0 | 20 | 18 | 19 | 18 |

New on 4 Oct, not in the ACRA paper (`papers/research/wind_algorithm_options.md`): a 15-line disturbance-observer
feed-forward (no learning) lands 80.6% at W10 18, against the learned flyer's 44-55%. And our model holds about 26 m/s
of steady wind at the drone; PX4's 45 deg tilt limit binds first (`papers/research/x500_wind_limit.md`; the body drag
CdA 0.04 m^2 is assumed, no published rating exists).

## Speed (RTX 5070 Laptop; MJX decisions per second, one decision = 0.04 s of every drone)

| drones at once | 256 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|
| decisions / s, GPU shared with other jobs (4 Oct) | does not compile | 47,000 | 130,000 | 210,000 |

On an idle GPU (6 Oct) one round of 4,096 drones x 64 decisions takes about 0.2 s of flying plus 0.2 s of learning
(4 passes x 8 minibatches): about 650,000 decisions/s. What the paper's runs really took (their logs, two trainings
sharing the GPU): the flyer `fly_s0` 1,000 rounds in 34 min, each decider seed (`stage2_c100h100g`) 600 rounds in
11 min. Their validation curves flatten after round 400-500: open `rl-landing/wind/history` in ClearML, or read the `val`
field of `runs/<tag>_s<seed>.jsonl`.

256 fails to compile: an XLA bug in the GPU code generator (a concatenate fusion), not memory. The same limit is why
`render.py` always flies a batch of 1,024. Two trainings side by side leave the GPU better used (8,192 drones cost 1.6x
the time of 4,096).

## Queue chains (what trained and flew every recipe; read their headers before re-running)

| chain (`conditions/wind/queue/`) | what |
|---|---|
| `overnight_27sep.sh` | the night of 26-27 Sept: 8 flyer seeds, the first deciders, the first Gazebo ladder |
| `wind_recipes_29sep.sh` | seeds 1-2 of `stage2_c300` and `stage2_c100h100` |
| `wind_flip_30sep.sh`, `wind_flip2_30sep.sh` | the reversal price, then shaping at the learner's discount (`stage2_c100h100g`) |
| `wind_gz.sh`, `wind_gz_30sep.sh`, `wind_gz2_30sep.sh`, `wind_gz2_resume.sh` | the Gazebo ladders |

Their markers and logs go to `outputs/queue/`. Several wait on a literal PID of a process that no longer exists (a
dead PID can be reused by an unrelated process): do not start them as they are. The commands inside them are the
recipes used in steps 4-8 above.

---

The sections below are the reference for how the drone, the wind and the task are built.

## 1. A GPU drone that flies like PX4 in Gazebo (`x500_mjx.py`)

The rig v2 x500 as PX4's Gazebo model defines it, one rigid body in MJX with every force applied by us:

| part | source | values |
|---|---|---|
| airframe | x500_base + gimbal + rig v2 camera link | 2.584 kg, inertia 0.0291/0.0291/0.0444 kg m^2 about the CoM |
| motors | Gazebo MulticopterMotorModel | T = 8.55e-6 w^2, w = 150 + 850 u rad/s, lag 12.5 ms up / 25 ms down |
| propeller drag | same | 8.06e-5 \|w\| v_perp per rotor, against the air |
| body drag | added (Gazebo had none) | 0.5 rho CdA \|v\| v, CdA 0.04 m^2 horizontal, 0.05 vertical |
| PX4 | PositionControl, AttitudeControl, RateControl, control allocation | default x500 gains; hover throttle 0.836 |

PX4's code paths emulated in JAX: velocity PID with acceleration feed-forward, thrust vector with the 45 deg tilt
limit, vertical priority with a 0.3 horizontal margin, both anti-windups; bodyzToAttitude; reduced-attitude control
with yaw weight 0.4; rate PID with PX4's integral factor and saturation freezing; the pseudo-inverse of the CA_ROTOR
effectiveness with PX4's normalisation and sequential desaturation (airmode off).

**Checked against PX4 in Gazebo** (`shared/px4_gazebo/px4_steps.py`: the same velocity commands, Gazebo's true CoM in
sim time; `python conditions/wind/x500_mjx.py --check`):

| command step | Gazebo 10/63/90% | MJX 10/63/90% | tilt Gazebo / MJX |
|---|---|---|---|
| 0 -> 3 m/s north | 0.22/0.68/1.24 s | 0.20/0.64/1.24 s | 26.2 / 26.8 deg |
| 0 -> 8 m/s north | 0.29/0.85/1.51 s | 0.28/0.80/1.52 s | 44.1 / 45.6 deg |
| 0 -> 1.5 m/s down | 0.06/0.20/0.32 s | 0.04/0.20/0.36 s | - |
| 0 -> 1.5 m/s up | 0.10/0.34/0.50 s | 0.08/0.32/0.48 s | - |

(Gazebo's model origin is at the drone's feet, which swing back when it pitches to accelerate: compare the CoM.)
4,096 drones step at ~21 M physics steps/s on the RTX 5070 Laptop; a PPO round of 4,096 drones x 64 decisions ~0.3 s.

## 2. The same wind in both simulators (`gz_winddrag/`, `land_env.wind_step`)

Mean wind W10 (at 10 m) with a log profile (z0 0.3 m, suburb); MIL-F-8785C low-altitude turbulence (first-order
filters, sigma_w = 0.1 W10, length scales from height); 1-cosine gusts (one per 15 s on average, 0.2-0.6 W10, 1.5-5 s,
within +-0.5 rad of the mean). The Gazebo system plugin sets the world's wind (which Gazebo's propeller drag uses) and
applies the body drag; it holds still air until switched on, then starts from its seed, so every decider meets the
same gusts. Hovering at 30 m in steady wind, tilt Gazebo / MJX: 5.76 / 5.73 deg (9.2 m/s, propeller drag only),
10.36 / 10.43 (9.2 m/s), 22.48 / 22.6 (15.8 m/s). `gz_wind.py` writes the wind world and a server config per wind
(`python conditions/wind/gz_wind.py --config w7 --mean 7 --dir 90`).

## 3. The task (`land_env.py`)

The last 30 m of a delivery: the drone hovers 30 m up, 15-25 m from a back-yard pad, a house (10 x 8 m, 4-7 m tall)
2.5-8 m from the pad, a tree beside it in 60% of yards (Logan LiDAR: 58%). Landed = touchdown within 0.75 m of the
centre, sinking < 1 m/s, sliding < 0.5 m/s; crashed = within 0.35 m of house or tree; otherwise hard/off the pad, or
timeout at 90 s.

**Today's rule** (gz_learner's landing): the approach law (0.69 s lead, sqrt and linear braking), descent 1.5 then
0.8 m/s over the pad, and below 6 m only when centred and still (the gate). 1,024 paired flights per wind:

| W10 (m/s) | 0 | 6 | 9 | 12 | 15 | 18 | 20 |
|---|---|---|---|---|---|---|---|
| landed | 100% | 100% | 100% | 100% | 78% | 5.5% | 0% |
| gave up (90 s) | 0 | 0 | 0 | 0 | 22% | 94% | 99% |
| mean time | 30 s | 32 s | 38 s | 51 s | 75 s | 89 s | 90 s |

The rule does not crash in strong wind: it waits for a stillness gusts never give.

## 4. The learner (`ppo.py`)

PPO in plain JAX (no RL library). It acts through the channel PX4 offers over MAVLink: the rule's velocity setpoint
plus an acceleration feed-forward (|a| <= 3 m/s^2 per axis, the flying), and a vertical choice every 0.04 s -- the
rule's own, down 1.5, down 0.6, hold, up 1 m/s (the deciding). It starts as the rule (its 'rule' option favoured,
feed-forward ~0) and learns where to leave it. Observation (50 numbers, `land_env.OBS_DIM`): pad offset (coarse and
fine), height, velocity, lean (where the wind shows), the rule's setpoint and gate state, house and tree, last action,
last 4 (velocity, lean).
Reward: landed +10..15 (by precision), hard/off-pad -10, crash -15, gave up -5, -0.05/s; gamma 0.9998. (Stage 2 changes
the prices: crash -100, hard -30 or -100; `--crash-r`, `--hard-r`, and `--flip-r` for a price on each climb/descend
reversal, `--shape-g` for the shaping's discount.)
**Curriculum**: flights draw W10 from [0, W] and its top 3 m/s; W starts at 4 m/s and rises 1 m/s once `--w-ok` of the
top band lands (default 0.8, the runs above use 0.95: 80% advanced too early, strong-wind results fell as it moved
on), up to 20 m/s.
**Two stages**: stage 1 trains the flyer alone (`--no-decide`); stage 2 freezes it (`--flyer`) and trains only the
decider (`--no-fly`). Checkpoints are chosen on validation yards (seed 20000) and scored on test yards (seed 10000),
never the other way round.

## 5. Gazebo check (`gz_windland.py`, `gz_ladder.sh`)

MJX evaluation yard i is built in Gazebo (pad, house, tree), the drone flies out in still air, holds at the start,
the wind switches on, the rule or the learner (numpy, checked equal to the JAX policy: `gz_windland.py --check-obs`)
lands it. Flight-by-flight agreement: `python conditions/wind/figs.py --pairs <gz_dir> conditions/wind/runs/rule_ladder.json`.
`gz_windland.py --geo` is the GNSS-denied version (`gnss_denied/geofly.sh` starts it).
