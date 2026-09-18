# rl-landing — handoff for Claude Code

Built 18 Sept 2026. Everything here was written and smoke-tested in a cloud
container against **synthetic** data. Nothing has touched the real GeoAnchor
fixes. The first job is to run it on real data and find out whether the study is
viable — the code is instrumented to tell you.

---

## What this is

Testing one claim: **a landing policy given a calibrated, metre-valued
localization uncertainty makes better approach decisions than one given only a
position estimate, or than one given PerchRL's staleness proxy.**

Three arms, identical except the uncertainty channel:

| arm | what the policy is told |
|---|---|
| `none` | position estimate only |
| `staleness` | `exp(-λ·t_since_fix)` — PerchRL's `ρ_t`, a stopwatch |
| `sigma` | GeoAnchor's calibrated σ, in metres |

The result is the **high-σ success column**, not the overall number. Every arm
should commit and succeed when σ is small; they can only separate where the
uncertainty is large.

---

## Setup

```bash
mkdir -p ~/RL_projects/rl-landing && cd ~/RL_projects/rl-landing
# copy these files in, then:
git init
conda activate mujoco_drones        # already has numpy, torch, SB3, gymnasium
pip install stable-baselines3 gymnasium    # if the env lacks them
```

No MuJoCo dependency in this stage — see "Why no MuJoCo" below.

---

## Run order

### 1. Calibrate — produces the replay pool AND two publishable numbers

```bash
python3 calibrate.py ~/GeoAnchor/results/s8_train --exclude-scene Scene_21
```

Outputs `sigma_pool.npz` plus `sigma_pool_report.json`.

Prints three things worth keeping regardless of whether the RL project proceeds:

- **ENCE** — the published metric (MUSE, arXiv 2605.17421) for "does predicted
  σ match actual error". `CLAUDE.md`'s `12→12, 21→21, 38→45` is an unnamed,
  eyeballed version of exactly this. Reporting the number makes it comparable
  to other work.
- **Conformal k and held-out coverage** — upgrades the claim from "σ tracks
  error" to "the true position is within k·σ at least 90% of the time",
  distribution-free, with a proof.
- Error percentiles (median/p90/p99 and fractions within 5/10/20 m). Never the
  mean — `CLAUDE.md`'s hard rule, enforced in code.

**These two numbers strengthen the GeoAnchor paper on their own, using data
already on disk. Do this step even if everything after it is abandoned.**

The σ predictor here is deliberately minimal: bin by `best_inliers`, predict σ
as the 0.68 quantile of |error| in the bin, fitted leave-one-scene-out. It uses
only columns present in `all_results.csv`, so no harness patch and no `.pkl`
parsing. To use the real step16 estimator instead, replace `predict_sigma()` —
everything downstream is agnostic to where σ came from.

Leakage guard is enforced: `pred_error_m` is the label; `pdm_at_k`,
`recall_at_k`, `retrieval_gt_rank`, `PDE`, `R_i` are banned as features. CV is
grouped by scene, never by row.

### 2. Baselines — THE DECISION GATE

```bash
python3 baselines.py --pool sigma_pool.npz --tune --episodes 400
```

Three scripted policies, each given an **equal tuning budget** on the same
episodes, evaluated on held-out seeds:

- `dive` — straight down, no search (the strawman; WaveLander's equivalent)
- `fixed_spiral` — descend to the detection ceiling, sweep a fixed disk (σ-blind)
- `sigma_spiral` — sweep a disk scaled by σ (σ-aware, hand-written)

This exists because *Leveling the Playing Field* (arXiv 2506.17832) showed that
RL-vs-classical gaps largely vanish once the tuning asymmetry is removed. If a
hand-tuned `sigma_spiral` already matches what PPO achieves, **that is the
finding** — same shape as GeoAnchor's finding 1 and MUSE's note that the
empirical-covariance baseline is "quite competitive". Report it; don't tune it
away.

**Read the table before training anything.** Two failure modes:

- If `fixed_spiral` scores ~100% at every σ, the task is too easy. Tighten
  `--t-max` (it's really a battery reserve) or `--vxy-max` (an airframe limit).
  **Do not weaken the baselines, and do not touch the camera constants** —
  `FOV_DEG`, `PX`, `MIN_PX`, `PAD_SIZE` are the physics of seeing a 1 m marker
  and they set the whole structure of the problem.
- If `dive` succeeds equally at low and high σ, σ carries no usable signal here
  and the RL result won't either. Stop and investigate.

On synthetic data with a GeoAnchor-shaped heavy tail I got:

```
policy          succ%   lo-sig     mid  hi-sig
dive             44.0    80.8%   30.6%    6.0%
fixed_spiral     72.0   100.0%   83.3%   12.0%
sigma_spiral     64.0   100.0%   62.5%   10.0%
```

Note `sigma_spiral` came out **worse** than `fixed_spiral`. That is either a
crudely parameterised hand policy or a real signal that σ is hard to exploit
with a fixed rule. Either way it is the number to beat, and it is *good* for the
paper that the σ-aware scripted policy is not automatically better — it means
the learned result, if it lands, is doing something a rule cannot.

### 3. Train

```bash
python3 train.py --pool sigma_pool.npz --all --steps 2000000 --n-envs 8
```

~30–60 min per arm on the Legion CPU; the env is numpy point-mass, so the GPU
buys little. Writes `runs/ablation.json` and a summary table split by σ.

Check `runs/tb` in tensorboard if you want curves (`--tb ""` disables).

### 4. Compare

`runs/ablation.json` vs `baselines.json`. The claim lives or dies in the
`hi-sig` column.

---

## Why no MuJoCo in this stage

WaveLander trains its landing policy on "a single prismatic joint constrained to
vertical translation" with a randomized-gain velocity tracker — no quadrotor
dynamics at all. At 50 m under a velocity controller, attitude contributes
nothing to a search-versus-commit decision, and a numpy point mass trains in
minutes instead of hours.

MuJoCo stays where it earns its keep: the existing `tello_landing_aviary.py`
cage env, for the Tello hardware sanity check (stage 2 below). Nothing in
`MuJoCo-drones-gym` is modified by this project.

---

## What's deliberately not built yet

- **Asymmetric actor-critic** (privileged critic, per WaveLander/DPRL). It would
  help — but privileged information given to one arm and not another is exactly
  the asymmetry *Leveling the Playing Field* condemns. Add it to all three arms
  at once or not at all. Add when the arms are converging poorly.
- **Evidential (NIG) σ head** per SCREP. The binned quantile predictor is
  sufficient to produce a pool. Add when you want a per-fix σ on the Pi in one
  forward pass.
- **The cage/Tello variant.** Stage 2. Reuses the existing MuJoCo aviary with σ
  scaled to cage dimensions.
- **ArduPilot SITL bridge.** Stage 3 — and mostly already written:
  `geoanchor-rt/scripts/watch_extnav.py` already injects a position with a
  configurable σ into a GNSS-denied ArduCopter over MAVLink ODOMETRY, and the
  17 Sept run proved EKF3 fuses it (walked to 19.81 m of 20 m injected in under
  5 s). That is the σ-injection harness. Do not write a second one. The policy
  side is one message: `SET_POSITION_TARGET_LOCAL_NED` with the velocity mask.
- **Gazebo.** Stage 4, and only for a rendered camera that can see an ArUco
  marker. If the marker pose can be stubbed with PerchRL's geometry-dependent
  noise model, it may never be needed.

---

## Known shortcuts (`ponytail:` comments in the source)

- `approach_env.py` — lateral guidance is a plain P controller with a velocity
  cap, not a full PID. The integral term only matters for steady-state offset
  against wind, which this env does not model. Upgrade when wind is added.

## Design decisions worth not undoing

- **σ is fixed within an episode.** It is a property of the scene, altitude band
  and matcher — not something that re-rolls twice a second. An earlier version
  re-drew σ and let the error bearing wander every fix, and simply descending
  then swept an accidental search pattern, which made the σ-blind policy
  artificially good and nearly destroyed the experiment.
- **Error magnitude is drawn per fix from the real errors in that σ decile.** So
  `P(error | σ)` stays exactly the empirical distribution. This is the point of
  replaying rather than simulating: calibration does not transfer for free, and
  randomizing σ's marginal does not preserve its relationship to the error.
- **The camera model is the difficulty.** `h_detect ≈ 30.6 m` for a 1 m pad at
  640 px / 60° — above that the marker is under 20 px and undetectable. Footprint
  radius is `h·tan(fov/2)` ≈ 17.3 m at 30 m. Because the footprint *shrinks* as
  you descend, diving the moment the marker appears drives it back out of frame;
  descent has to be gated on alignment (WaveLander's trick, implemented in
  `baselines._descend_aligned`). Without that gate every policy scored ~2%.
- **A bug worth remembering.** GeoAnchor estimates the *drone's* position; the
  pad is a delivery coordinate known exactly. So the believed pad sits at
  `pad − err`, not at the believed drone position. Getting this backwards made
  the guidance target run away and produced 1–5% success across the board.

---

## Self-checks

```bash
python3 approach_env.py     # physics + ablation wiring + the study's precondition
```

The important assertion: commit-and-dive must succeed at low σ and fail at high
σ. If that gap closes, there is nothing for a σ-aware policy to learn and no
amount of training will produce a result. On synthetic data it reports 95% vs
33%. **Run this against the real pool first.**

---

## Files

```
calibrate.py     GeoAnchor CSVs -> ENCE, conformal coverage, sigma_pool.npz
approach_env.py  the 6-dim / 2-action approach task + self-check
baselines.py     three scripted policies, equal tuning budget, eval harness
train.py         PPO, three ablation arms, σ-split evaluation
```

---

# RUN LOG — 18 Sept 2026, real GeoAnchor data

Everything above this line was written against synthetic data. Everything below
is what actually happened on `results/s8_train`.

## Bugs fixed before anything would run correctly

1. **`calibrate.load()` counted every fix twice.** The per-matcher directories
   (`satellite-CAMP-XFEAT_LG-yp`) and the aggregate directory (`satellite`) both
   satisfy `ref in basename(dirname)`, and the aggregate CSV is the
   concatenation of the three per-matcher ones. Now excludes `d == ref`.
   *3138 + 1046×3 rows were being read where 1046 was intended.*
2. **`--exclude-scene` was inescapable.** `action="append"` with a literal
   default appends to it rather than replacing, so `Scene_21` could never be
   included and was printed twice. Default is now `None`, resolved after parse.
3. **`approach_env.demo()` could only run on a synthetic pool.** It now takes an
   optional pool path: `python3 approach_env.py sigma_pool.npz`.
4. **`baselines.py` did not expose `--t-max` / `--vxy-max`**, the two difficulty
   knobs its own docstring tells you to reach for. Wired through to both the
   tuner and the eval, and recorded in `baselines.json` under `env`.

## Step 1 — calibration on real data (PUBLISHABLE, independent of the RL work)

```
XFEAT_LG / satellite / excluding Scene_21
440 fixes, 4 scenes (09, 10, 16, 22), 1 absurd (>1000 m) held out

p50  11.62 m   p90  34.34 m   p99 127.52 m
within  5 m: 17.3%    within 10 m: 42.7%    within 20 m: 78.6%

ENCE = 0.313   (leave-one-scene-out)
conformal @ alpha=0.10: k = 1.97, held-out coverage 90.0% over 4 scene folds
  -> the true position is within 1.97 * sigma at least 90% of the time
```

These are the two numbers the handoff promised would stand on their own.
`sigma_pool.npz` holds 440 (sigma, error) pairs; sigma spans 8.75–80.08 m,
median 13.82 m, with 20.7% of fixes above 25 m.

## Step 1b — the precondition, on the real pool

```
python3 approach_env.py sigma_pool.npz
ok  h_detect=30.6 m  footprint@30m=17.3 m
    dive succeeds 98% at low sigma vs 42% at high -- 56% gap
```

**The study is viable.** The gap is real on real data (synthetic was 95/33).

## Step 2 — the decision gate, and what it exposed

At stock difficulty (`T_MAX=120`) the task was too easy: `fixed_spiral` scored
100% / 100% / 88.2%. Median episode finishes in 29 s against a 120 s budget, so
the battery reserve was not binding at all. Per this document's own instruction,
`--t-max` was tightened to **45 s** (still ~1.5x a nominal descent) — the camera
constants were not touched. All numbers below and all training use `--t-max 45`.

**The tuning grids were badly centred.** On the first real-pool run *both*
spirals selected the smallest value their grid offered (`r_fixed=8`,
`k_sigma=0.75`) — the optimum was outside the grid, so the "equal budget" was
being spent on candidates neither policy wanted. Grids were re-centred downward,
symmetrically, same 12 candidates each: `r_fixed ∈ (3,6,12)`,
`k_sigma ∈ (0.2,0.4,0.8)`.

```
policy          succ%   acq%   miss50   miss90    t50  lo-sig     mid  hi-sig
dive             83.5   83.8    0.00m   22.38m  27.4s  100.0%  100.0%   48.0%
fixed_spiral     91.0   94.0    0.00m    0.00m  29.3s  100.0%  100.0%   71.7%
sigma_spiral     91.0   93.5    0.00m    0.00m  29.4s  100.0%  100.0%   71.7%
```

With a fairly-centred grid the sigma-aware and sigma-blind scripted rules become
**exactly indistinguishable** (71.7% vs 71.7%). On the badly-centred grid
sigma_spiral looked worse (61.4 vs 70.9); that gap was a grid artefact, not a
finding. Report the 71.7/71.7.

## The mechanism, which is not the one the file was written around

Both grids hit their floor a second time, so the sweep radius was swept to zero
directly:

```
r_fixed  h_hold   succ%  hi-sig
    0.0      30    91.0    71.7     <- no search at all
    3.0      30    91.0    71.7
    8.0      30    89.8    67.7
    0.0      34     0.0     0.0     <- above h_detect=30.6: never acquires
```

**The spiral contributes nothing.** `r_fixed=0` matches the best tuned sweep.
Every point `fixed_spiral` scores over `dive` (91.0 vs 83.5, 71.7 vs 48.0 at
high sigma) comes from *holding altitude just under the detection ceiling*, not
from searching laterally. What the hold buys is time for successive replayed
fixes to pull the believed pad position around — temporal averaging, not spatial
coverage.

That reframes the decision the policy is actually making. It is not "how wide do
I sweep", it is **"how long do I loiter before I commit"** — and with a 45 s
battery reserve that is a real tradeoff, and one sigma is the natural input to.
It also explains why scaling a search radius by sigma buys nothing: the sweep
was never the operative variable.

If the sigma arm beats staleness below, this is the mechanism to describe.

---

## Step 3 — the 5-seed ablation (env as originally written, rho=0)

```
arm          seeds      succ%           hi-sig%            range
none             5   86.0 +/- 2.8   54.8 +/- 8.9   [39.0, 60.2]
staleness        5   86.8 +/- 1.3   56.9 +/- 4.1   [49.6, 59.3]
sigma            5   87.7 +/- 0.7   60.0 +/- 2.4   [56.1, 62.6]
scripted baseline                       71.7

sigma - none      = +5.2 pts  p ~ 0.21
sigma - staleness = +3.1 pts  p ~ 0.15
sigma - baseline  = -11.7 pts p < 0.0001
```

Ordering is the hypothesised one but **not significant at 5 seeds**. The more
robust effect is variance: sigma cuts seed-to-seed spread ~4x versus `none`
(2.4 vs 8.9; worst seed 56.1 vs 39.0). Telling the policy how wrong it is makes
training *reliable*, which may be the more defensible claim than the mean.

## Step 4 — two action-space redesigns, both falsified, and why

**Attempt 1: the search radius was inert.** Swept directly, `r_fixed=0` ties the
best tuned sweep (91.0 / 71.7 hi-sig). The camera footprint at the ceiling
(17.3 m) already dwarfs the disk worth sweeping.

**Attempt 2: replaced it with a chase gain** (low-pass on the guidance target).
Motivated by a real measurement -- at high sigma the believed pad jumps 51 m
(p90) between 0.2 s control steps while the aircraft can fly 1.0 m, so it is
chasing noise. But the gain swept monotone to its boundary:

```
chase g   0.05   0.10   0.20   0.40   0.70   1.00
hi-sig     7.0   24.4   32.6   41.9   55.8   70.9
```

Damping strictly hurts. **Both levers pin at a boundary**, which is the signal
that the action space was never the binding constraint.

**The actual cause: the error's temporal model.** Freezing the error magnitude
within an episode instead of re-drawing it every fix:

```
                     dive              hold+chase
iid re-draw      86.5 / hi 40.7      93.5 / hi 70.9
persistent       71.8 / hi 36.0      71.8 / hi 36.0   <- identical
```

Under a persistent error, loitering buys **exactly zero**. All of the baseline's
advantage came from `_new_fix()` re-drawing the magnitude iid every 0.5 s, so
hovering at the ceiling re-rolls the dice until a small draw lands. That is a
lottery, not an estimation problem, and sigma only tells you how long it takes.

## Step 5 — the error model, measured rather than assumed

`sample_id` is sequential frames (`L09_0001, L09_0002, ...`), so the temporal
correlation is measurable from data already on disk:

```
lag-1 RANK autocorrelation of |error|
  Scene_09  +0.257    Scene_16  +0.468
  Scene_10  +0.358    Scene_22  +0.550
  pooled    +0.363
```

Rank, not Pearson: the error is heavy-tailed and one 160 m fix would otherwise
set the statistic, and the consumer is a Gaussian copula whose parameter is the
rank correlation. **Stated assumption:** this is the correlation at the
*dataset's* frame spacing, not the env's `FIX_DT`. `rho` is a knob
(`ApproachEnv(rho=...)`) with 0.363 as its data-grounded default.

`_new_fix` now draws through an **AR(1) Gaussian copula**: `z_t = rho*z_{t-1} +
sqrt(1-rho^2)*eps`, `u_t = Phi(z_t)`, `mag_t = empirical_quantile(bin, u_t)`.
Because `z_t` is standard normal at stationarity, `u_t` is exactly uniform and
`P(error | sigma)` stays EXACTLY the measured distribution. An AR(1) on the raw
magnitude would have distorted that marginal. Asserted in `demo()`.

## Step 6 — and the search lever is a knife-edge

Action space is now 3-dim (vertical rate, search radius, chase gain) so neither
lateral lever is dropped a priori. Sweeping rho against search radius:

```
 rho | r=0    r=6    r=12   r=20    best
0.00 | 57.7  49.3  39.4  40.8     r=0
0.36 | 70.4  57.7  49.3  46.5     r=0   <- the measured value
0.60 | 71.8  69.0  62.0  49.3     r=0
0.80 | 78.9  74.6  66.2  64.8     r=0
0.95 | 67.6  64.8  60.6  56.3     r=0
1.00 | 38.0  49.3  49.3  50.7     r=20  <- only here
```

Spatial search pays **only in the exact limit rho=1**. At the measured rho the
tuned `sigma_search` selects `k_sigma=0` and no damping -- it degenerates
*literally* into `fixed_search`, which is why both score 74.8%.

**So at the data-grounded error model, a scripted sigma-aware policy has nothing
to do.** That part of the study is settled and it is a negative result worth
reporting. The open question is only whether a *learned* policy can use sigma
where a rule cannot -- which is the original ablation, and is what the matched
hyperparameter sweep below is for.

Note also that hi-sigma success is non-monotone in rho (57.7 at 0, peaking 78.9
at 0.8, collapsing to 38.0 at 1.0). Task difficulty is strongly rho-dependent,
so rho must be reported alongside any success number from this env.
