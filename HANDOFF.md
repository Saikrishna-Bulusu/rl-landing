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

## Step 7 — matched hyperparameter budget for PPO

The scripted baselines were given a 12-candidate tuning sweep while PPO ran on
hand-written defaults. That is a tuning asymmetry running the *other* way, and
the same paper that motivates the baselines condemns it. Hyperparameters are now
flags on `train.py`, and each arm got its own 12-candidate grid
(lr x ent_coef x net_arch), 600k steps, one seed, scored on held-out eval seeds.

```
arm         best HP                                  hi-sig @600k   (default HP @2M)
none        --lr 1e-3 --ent-coef 0.02 --net 256 128      66.3            54.8
staleness   --lr 3e-4 --ent-coef 0.02 --net 256 128      48.9            56.9
sigma       --lr 1e-3 --ent-coef 0.02 --net 128 64       70.7            60.0
```

**The original PPO hyperparameters were badly undertuned.** `none` reached 66.3
at 600k steps where it had managed 54.8 at 2M on defaults. The earlier
"scripted baseline beats every learned arm by 11.7 points" conclusion is
therefore partly retracted: a meaningful part of that gap was the tuning
asymmetry, not the method. Report the matched-budget comparison, not the first
one.

Caveats to carry into the writeup:

- These winners are the **max of 12 noisy single-seed runs**, so they are
  optimistically biased (winner's curse). Only the multi-seed run at fixed HP is
  an unbiased estimate.
- The per-arm winners differ in architecture (`none` and `staleness` took
  256-128, `sigma` took 128-64). That follows from the equal-budget-per-arm
  protocol, but it does mean the arms now differ in more than the uncertainty
  channel. State it. The alternative -- one shared HP set chosen on one arm --
  would advantage whichever arm it was chosen on.
- `staleness` candidate #1 collapsed to 0.0% success. Training instability at
  lr=1e-4 / low entropy; it was the worst candidate regardless.

## Step 8 — final ablation (running)

8 seeds x 3 arms at 2.5M steps, per-arm tuned HP, rho=0.363, t_max=45.
Seeds were raised from 5 to 8 because statistical power, not training length,
was the binding constraint on the earlier result.

## Step 8 RESULT — final ablation, 8 seeds x 2.5M steps, per-arm tuned HP, rho=0.363

```
arm          n        succ%          hi-sig%          range
none         8   90.2 +/- 2.6   68.2 +/-  8.4   [52.8, 76.4]
staleness    8   89.6 +/- 2.7   67.3 +/-  8.7   [51.2, 76.4]
sigma        8   91.8 +/- 3.4   73.9 +/- 11.4   [54.5, 89.4]
scripted baseline  92.0          74.8

sigma - none      = +5.7 pts  t=+1.14  p=0.255  d=+0.57
sigma - staleness = +6.6 pts  t=+1.30  p=0.192  d=+0.65
 none - staleness = +0.9 pts  t=+0.21  p=0.831  d=+0.11
sigma - baseline  = -0.9 pts  t=-0.23  p=0.819
```

### Four conclusions, in order of how well they are supported

1. **PPO now MATCHES the tuned scripted baseline** (73.9 vs 74.8, p=0.82). The
   earlier 11.7-point deficit is gone once PPO gets the same tuning budget the
   baselines got. This is the cleanest result in the study and it cuts against
   the first draft of the finding: the classical-beats-learned gap here was
   substantially a tuning artefact, exactly the mechanism *Leveling the Playing
   Field* describes -- only pointed at RL rather than at the classical side.

2. **PerchRL's staleness proxy does NOT replicate.** `staleness` and `none` are
   indistinguishable (+0.9 pts, p=0.83, d=0.11). A stopwatch carries no usable
   signal in this task. Since sigma is fixed within an episode here, the
   staleness channel `exp(-lambda*t_since_fix)` is a deterministic sawtooth
   uncorrelated with actual error -- so this is arguably a property of the env
   design, not a refutation of PerchRL. Say so; do not overclaim it.

3. **sigma > none and sigma > staleness directionally, medium effect, NOT
   significant.** d = 0.57 and 0.65 at n=8. Required seeds for 80% power at
   alpha=0.05: **~49 per arm vs none, ~37 vs staleness** -- 111-147 runs,
   9-11 h at current throughput. Eight seeds was never going to settle this.
   Either commit that compute or report it as a non-significant trend with the
   effect size and the power calculation attached. Do not report +5.7 points as
   a result without the p-value.

4. **RETRACTED: the "sigma makes training more reliable" claim.** At 5 seeds on
   default hyperparameters sigma had ~4x LOWER seed variance (2.4 vs 8.9). At
   8 seeds with tuned hyperparameters the ordering REVERSES -- sigma has the
   HIGHEST variance (11.4 vs 8.4 vs 8.7). The earlier variance effect was an
   artefact of undertuned hyperparameters. It does not survive. Do not use it.

### What this means for the study

The headline that survives is narrow and honest: **at GeoAnchor's real error
distribution and its measured temporal correlation, a calibrated sigma gives a
landing policy no advantage that reaches significance at n=8, a hand-written
rule needs sigma even less (it tunes the sigma term to zero), and a properly
tuned learned policy exactly matches that rule.** The calibration contribution
(ENCE, conformal coverage, the autocorrelation measurement) is unaffected by any
of this and stands on its own.

The single highest-value next experiment is NOT more seeds at rho=0.363. It is
rho=0.8, where the task is hardest and where holding still stops being a free
win (hi-sigma peaks at 78.9 there and collapses to 38.0 at rho=1.0). If sigma
cannot separate the arms at rho=0.8 either, the negative result is solid and
cheap. If it can, that is the regime to report and the reason to spend the
150-run budget.

## Step 9 — rho=0.8, and the caveat that turned out to matter most

Re-ran the 8-seed x 3-arm ablation at rho=0.8 (2.5M steps, same per-arm tuned
HP) because that is where the task is hardest and where holding still stops
being a free win.

```
                      rho=0.363 (as measured)      rho=0.80
arm            succ%      hi-sig%           succ%      hi-sig%
none           90.2   68.2 +/-  8.4         85.3   57.3 +/- 7.1
staleness      89.6   67.3 +/-  8.7         84.8   57.6 +/- 6.7
sigma          91.8   73.9 +/- 11.4         89.4   68.0 +/- 8.4
baseline              74.8                         83.5

sigma - none        +5.7  p=0.255 d=0.57    +10.7  p=0.006 d=1.37  *
sigma - staleness   +6.6  p=0.192 d=0.65    +10.4  p=0.007 d=1.36  *
 none - staleness   +0.9  p=0.831 d=0.11     -0.3  p=0.930 d=-0.04
sigma - baseline    -0.9  p=0.819           -15.5  p<0.001
```

**The sigma channel separates, decisively, in the harder regime.** The effect
roughly doubles (+5.7 -> +10.7 points) and goes from non-significant to
p=0.006 with a large effect size (d=1.37). Both sigma comparisons survive
Bonferroni correction for the three pairwise tests (alpha = 0.05/3 = 0.017).

**And rho=0.8 is approximately the CORRECT value, not an arbitrary harder
setting.** The caveat flagged in Step 5 -- "rho was measured at the dataset's
frame spacing, not the env's FIX_DT" -- is now resolved with data:

```
median spacing between consecutive dataset frames: 7.3 m
at 5.0 m/s (= ApproachEnv.VXY_MAX)  -> 1.5 s between frames
AR(1) rescaling rho(t) = rho(dt)^(t/dt) to the 0.5 s fix interval:

  at  5 m/s -> rho = 0.707
  at 10 m/s -> rho = 0.543
  at 15 m/s -> rho = 0.400
```

The frame-to-frame 0.363 is a **lower bound**. Fixes arrive closer in time than
dataset frames do, so they are more correlated. rho=0.363 corresponds to flying
at ~17 m/s, faster than the env's own airframe limit. `calibrate.py` now
performs this rescaling and writes rho=0.707 into the pool as the default.

**So the earlier non-significant result was measured at the wrong operating
point.** The Step 8 run was not wrong, it was answering the question at a speed
the aircraft cannot fly.

### What is now supported, and what is not

SUPPORTED:
- A calibrated metre-valued sigma beats both a position-only policy and
  PerchRL's staleness proxy, at the operationally correct error correlation,
  p<0.01, large effect, Bonferroni-corrected, 8 seeds.
- The staleness proxy does not replicate at EITHER rho (+0.9 / -0.3, both
  p>0.8). Caveat from Step 8 still stands: sigma is constant within an episode
  here, so the staleness channel is structurally uninformative in this env.

NOT SUPPORTED, AND DO NOT CLAIM:
- "PPO loses to the scripted baseline at rho=0.8" (-15.5 pts). The
  hyperparameters were tuned at rho=0.363 and reused at 0.8. That is precisely
  the tuning asymmetry this project has already been burned by twice. The
  arm-vs-arm comparison is unaffected (each arm carries its own equal-budget
  tuning), but the learned-vs-scripted comparison at rho=0.8 is confounded and
  must not be reported until the HP sweep is re-run at rho=0.8 (36 runs, ~1 h).
- A tuned sigma-aware SCRIPTED rule still collapses onto the sigma-blind one at
  rho=0.8 (83.5 vs 83.5), as it did at rho=0 and rho=0.363. Three for three.
  Whatever sigma is worth here, a hand-written rule cannot extract it -- which
  is what makes the learned result interesting.

### Next, in priority order

1. **Re-run the HP sweep at rho=0.707** (36 runs, ~1 h) and then the 8-seed
   ablation (24 runs, ~2 h). This makes the learned-vs-scripted comparison
   honest and puts every number at the correct operating point. Until this is
   done the headline is "sigma beats staleness", not "learned beats scripted".
2. Report rho sensitivity as a curve (0.36 / 0.54 / 0.71 / 0.8), since the
   value depends on assumed ground speed and difficulty is non-monotone in rho.
3. Only then consider the ~50-seed run, and only for whichever comparison is
   still borderline.

## Step 10 — FAIR TEST at rho=0.707 (HP re-tuned at the same rho). READ THIS ONE.

Sweep re-run at rho=0.707 (36 runs), winners taken per arm, then 8 seeds x 3
arms at 2.5M steps. Scripted baselines re-tuned at rho=0.707 as well. This is
the first comparison in the project with NO tuning asymmetry anywhere.

```
version        overall        hi-sigma        per-seed median
none          86.7 +/- 0.8   57.0 +/- 2.4         56.9
staleness     87.0 +/- 3.2   60.3 +/- 7.6         58.1
sigma         87.9 +/- 3.0   63.2 +/- 8.6         60.6
hand rule     94.2                81.9

sigma vs none        +6.2  p=0.0504  d=0.98   borderline; fails Bonferroni (0.0167)
sigma vs staleness   +2.9  p=0.4679  d=0.36   no evidence
 none vs staleness   -3.3  p=0.2481  d=-0.58  no evidence
sigma vs hand rule  -18.7  p<0.0001           hand rule wins decisively
```

### The rho=0.8 result (Step 9) DID NOT REPLICATE

Step 9 reported sigma vs staleness = +10.4, p=0.007, and called the core claim
supported. At rho=0.707 with matched tuning it is +2.9, p=0.47. **Withdraw the
Step 9 claim.** Do not put it in front of Nabin.

**Cause is confounded, and that is my error.** Two things changed between Step 9
and Step 10 at once: rho (0.8 -> 0.707) AND where the hyperparameters were tuned
(rho=0.363 -> rho=0.707). Cannot attribute the difference. Most likely
explanation is that the HPs borrowed from rho=0.363 happened to suit the sigma
arm better than the other two, manufacturing an advantage -- which is the exact
failure mode this project has now hit THREE times in three different directions.

Also note sigma's mean is carried by two lucky seeds (74.8, 77.2). Mean gap 6.2,
median gap 3.7. Report medians alongside means for this env; the seed
distribution is skewed.

### What IS established

1. **A tuned scripted rule beats tuned PPO at the corrected operating point**,
   63.2 vs 81.9, p<0.0001, with equal 12-candidate budgets on both sides and no
   asymmetry remaining. This is the cleanest result in the project. Same shape
   as GeoAnchor finding 1 and MUSE's "empirical covariance is quite
   competitive".
2. **A sigma-aware scripted rule tunes itself into the sigma-blind one** at
   rho = 0, 0.363, 0.707 and 0.8. Four for four. A hand rule cannot use sigma.
3. **The staleness proxy does not help at any rho tested.** Caveat unchanged:
   sigma is constant within an episode here, so the channel is structurally
   uninformative in this env -- an env-design property, not a refutation of
   PerchRL.
4. Calibration results (ENCE 0.313, conformal k=1.97 at 90% coverage,
   per-scene error autocorrelation, and the frame-spacing -> fix-rate
   rescaling to rho=0.707) are untouched by any of the above.

### The disambiguating experiment, if the sigma question is pursued

Run rho=0.8 with HPs tuned AT rho=0.8 (36 + 24 runs, ~3 h). That changes only
rho relative to Step 10 and only the HP source relative to Step 9, so whichever
way it lands, the confound is resolved. Do this BEFORE spending the ~50-seed
budget on anything.

Standing lesson from this project, stated plainly because it has now happened
three times: **every single wrong conclusion here came from an unequal
comparison or an unchecked assumption -- never from a bug in the physics.**

## Stage 2 prepared (19 Sept, overnight) — cage_env.py

`cage_env.py` subclasses the existing `TelloLandingAviary` and adds the same
three-arm sigma ablation. Self-check passes:

```
ok  obs=27-dim  cage sigma 0.17-0.55 m vs 0.25 m landing radius  rho=0.707
    median cage sigma 0.26 m, scale 0.0187 (= 1.5/80)
```

**Bug found in `MuJoCo-drones-gym/tello_landing_aviary.py` (NOT fixed there).**
It declares `observation_space` as shape (18,) but `_computeObs` returns 26:
the base aviary contributes 20 elements, not the 12 the inline comment assumes.
`observation_space.contains(obs)` is False, which SB3 would reject. `cage_env`
derives the space from a real observation instead. Decide separately whether to
patch it upstream -- this project has otherwise not modified that package.

**Sigma scaling is a judgement call, recorded in the module docstring.**
Scaled by operating half-extent (1.5 / 80), giving median sigma 0.26 m against
a 0.25 m landing radius. Rejected: scaling by success radius (gives an absurd
4.6 m in a 3 m cage) and by altitude (0.55 m, defensible but harder).
`sigma_scale` is a constructor argument so it can be swept.

**The copula code is duplicated from `approach_env`, deliberately.** A 130-run
experiment was importing that module while this was written, and editing it
would have changed the code under jobs that had not yet started. Unify into a
shared `replay.py` once the overnight chain finishes.

Stage 2 training has NOT been started: the CPU is saturated by the overnight
chain, and it was to begin only after those experiments finish.

# ============================================================
# OVERNIGHT 18->19 SEPT: 216 runs. THE SIGMA EFFECT DOES NOT SURVIVE.
# ============================================================

Six stages, 00:13 -> 05:43, no crashes, memory guard reported 13.7-14.3 GB free
at every checkpoint. 132 point-mass runs + 84 cage runs.

## The headline: 20 seeds kills the sigma result

```
                 none      staleness    sigma     hand rule
rho=0.363        69.3        69.7        70.4       74.8
rho=0.707        58.5        60.1        58.2       81.9
rho=0.80         52.1        55.2        57.0       83.5

               sigma - none            sigma - staleness
rho=0.363   +1.1  p=0.76             +0.7  p=0.84
rho=0.707   -0.4  p=0.89             -2.0  p=0.52
rho=0.80    +4.9  p=0.32             +1.8  p=0.52
```

**Step 9's rho=0.8 result (+10.4, p=0.007, "the core claim is supported") was a
small-sample artefact.** At 8 seeds it looked significant. At 20 seeds the same
comparison is +1.8 with p=0.52. Withdraw it completely.

95% confidence intervals on the sigma advantage, which is the honest way to
state a null:

```
rho=0.363   sigma - none  [-5.7, +7.9]   sigma - staleness  [-5.8, +7.1]
rho=0.707   sigma - none  [-5.6, +4.9]   sigma - staleness  [-7.9, +4.0]
rho=0.80    sigma - none  [-4.8, +14.6]  sigma - staleness  [-3.8, +7.4]
```

So: **any sigma advantage larger than roughly 5-15 points is excluded at 95%**,
while the hand-written rule beats every learned arm by 5 / 22 / 27 points. This
is a well-powered negative result, not an absence of evidence.

## What the study now says, and it is publishable as a negative

1. A calibrated sigma gives a LEARNED landing policy no measurable advantage
   over a position-only policy, or over PerchRL's staleness proxy, at any of
   three error-correlation settings, with 20 seeds per arm.
2. A tuned scripted rule beats every learned arm at every setting, under equal
   12-candidate tuning budgets on both sides.
3. A sigma-aware SCRIPTED rule tunes its sigma term to zero at rho = 0, 0.363,
   0.707 and 0.8. Four for four.
4. PerchRL's staleness proxy does not replicate. Caveat unchanged: sigma is
   constant within an episode here, so that channel is structurally
   uninformative in this env.
5. The calibration contribution (ENCE 0.313, conformal k=1.97 at 90% coverage,
   per-scene error autocorrelation, frame-spacing -> fix-rate rescaling to
   rho=0.707) is untouched and stands on its own.

**The seed-count lesson is the methodological one worth reporting.** The same
comparison read as p=0.007 at n=8 and p=0.52 at n=20. Eight seeds is not enough
to make a claim in this env, and the effect sizes that looked "large" (d=1.37)
were inflated by exactly that.

## Stage 2 (cage): 84 runs WASTED by an action-space bug. Fixed.

Every one of the 84 cage runs scored 17.0-17.2% -- a 0.1% spread across 12
different hyperparameter settings and 16 seeds. That is not learning, it is a
constant.

Cause: `TelloLandingAviary` exposes RAW MOTOR RPM as its action space,
`Box(0, 21713.7)`. A PPO Gaussian policy emits roughly [-3, +3], which clips to
[0, 3] RPM against a HOVER_RPM of 14476 -- **0.02% of the thrust needed to stay
airborne**. The drone fell out of the sky on every step regardless of the
policy, and landed within 0.25 m by luck 17% of the time.

Fixed in `cage_env.py`: action space is now `Box(-1, 1, (4,))`, mapped to
`HOVER_RPM * (1 + 0.30 * a)`. Verified: a=-1 descends to 0.00 m, a=0 holds
1.92 m, a=+1 climbs to 2.99 m.

**A self-check now asserts flight authority** (`demo()` in cage_env.py):
descend/hover/climb must all be reachable. This is the check whose absence cost
84 runs -- an un-normalised action space passes every other test in the file and
then silently returns a constant score.

Also changed: `pad_moves` now defaults to **False**. A drifting pad is a second,
unrelated source of uncertainty that would confound the localisation-uncertainty
question being measured.

After the fix, learning is real but slow: 800k steps took ep_rew_mean from
-2.6e4 to -1.3e4 and success from 3% to 11%. The repo's own `train_landing.py`
uses 3M steps with a stationary->slow->moderate curriculum, so the cage needs
roughly 3M steps per run. Re-running the cage properly is ~3.5 h
(36 sweep @ 1.5M + 48 ablation @ 3M).

# ============================================================
# CAGE RE-RUN (19 Sept, action space fixed). 84 runs, 0 crashes.
# ============================================================

Preflight passed before launch (`preflight.py`). Policy influence confirmed on
real runs: the HP sweep spanned 0.2%-24.8% success, versus 0.25 points of
spread in the broken version.

## Results: 16 seeds x 3 arms @ 3M steps, rho=0.707

```
arm          n   success%        median   range         median miss
none        16   7.8 +/- 5.9      6.1     [0.0, 19.0]      1.15 m
staleness   16  14.9 +/- 6.1     16.4     [6.5, 27.5]      0.74 m
sigma       16  13.2 +/- 6.1     11.6     [5.0, 27.8]      0.79 m
                                          (landing radius 0.25 m)

sigma - none        +5.4  p=0.0109  d=+0.90  95%CI[+1.2, +9.6]   SIGNIFICANT
sigma - staleness   -1.6  p=0.4483  d=-0.27  95%CI[-5.8, +2.6]
 none - staleness   -7.0  p=0.0009  d=-1.17  95%CI[-11.2,-2.9]   SIGNIFICANT
```

Both survive Bonferroni correction (alpha = 0.05/3 = 0.0167).

## DO NOT REPORT THIS AS "SIGMA HELPS". The staleness channel is a CLOCK.

Measured directly:

```
staleness value, 4 different episodes: byte-identical
  max difference between any two episodes:  0.00e+00
  correlation(staleness value, actual error): +0.0026
```

Sigma is constant within an episode, so `exp(-lambda * t_since_fix)` is a
deterministic sawtooth -- the SAME sawtooth in every episode, carrying zero
information about the localisation error. It nonetheless beat `none` by 7.0
points at p=0.0009.

That gain therefore cannot be uncertainty awareness. The mechanism is timing:
the believed position jumps every FIX_DT=0.5 s, and a clock lets the policy
anticipate those jumps.

**This also reconciles the two environments.** `ApproachEnv._obs` already
contains elapsed time (`obs[5]`), so a clock was redundant there and staleness
did nothing. The cage observation has no time feature, so staleness supplied a
missing clock. Both results now agree under one explanation.

**Consequence for the sigma claim:** sigma's +5.4 over `none` may likewise be an
"any extra structured input helps" effect rather than an uncertainty effect.
Sigma is a per-episode constant, which is genuinely informative about difficulty
-- but that is not established by this experiment.

## THE CONTROL THAT WOULD SETTLE IT (recommended before any writeup)

Add a fourth arm whose extra channel is a **random per-episode constant**,
drawn independently of sigma and of the error. Same shape as the sigma channel,
zero information.

  - if the random-constant arm ALSO beats `none`, then neither sigma nor
    staleness is being used as uncertainty, and the whole cage result is an
    input-shape artefact. That is the single most likely way this project
    produces a false positive.
  - if it does NOT beat `none`, sigma's advantage is real and specific.

Cost: 16 seeds x 3M = 48 runs-equivalent, ~1.3 h. Cheap relative to the risk of
publishing an artefact.

A second useful arm would be a bare clock (`t / T_MAX`), to confirm the
staleness gain is timing rather than anything else.

## Caveat on absolute performance

Success is 8-15% against a 0.25 m landing radius, with median miss 0.74-1.15 m.
The task is hard: the policy learns to fly from raw motor commands AND to land.
The repo's own trainer uses a stationary -> slow -> moderate pad curriculum.
Differences between arms are significant, but all arms are weak in absolute
terms, so these numbers should not be presented as landing performance.

# ============================================================
# CONTROL EXPERIMENT (19 Sept). THE CAGE SIGMA RESULT DOES NOT SURVIVE.
# ============================================================

56 runs (24 tuning + 32 ablation), 0 crashes. Control arms received the SAME
12-candidate tuning budget as the real arms, so they are not disadvantaged.

## All five arms, 16 seeds @ 3M steps, rho=0.707

```
arm          n   success%      median   channel carries
none        16   7.8 +/- 5.9     6.1    nothing
clock       16  10.1 +/- 8.1     7.6    CONTROL: bare elapsed time
random      16  11.5 +/- 5.5    13.2    CONTROL: zero info (corr +0.02 w/ error)
sigma       16  13.2 +/- 6.1    11.6    real error bar (corr +0.55 w/ error)
staleness   16  14.9 +/- 6.1    16.4    a clock (corr +0.003 w/ error)

sigma - none       +5.4  p=0.0109  d=+0.90  CI[+1.2,+9.6]  SIGNIFICANT
sigma - random     +1.7  p=0.3973  d=+0.30  CI[-2.3,+5.8]  n.s.   <- THE CONTROL
random - none      +3.7  p=0.0703  d=+0.64  CI[-0.3,+7.6]  n.s.
clock - none       +2.3  p=0.3668  d=+0.32  CI[-2.7,+7.2]  n.s.
staleness - clock  +4.8  p=0.0593  d=+0.67  CI[-0.2,+9.7]  n.s.
```

## Verdict

**Sigma is not distinguishable from a meaningless input.** The `random` channel
has the same distribution as the sigma channel (median 0.262, range
[0.17,0.55]) but is drawn from an independent bin and correlates +0.016 with the
episode's actual error, against sigma's +0.549. It scores 11.5% to sigma's
13.2%: +1.7 points, p=0.40.

**The best-performing arm carries no information at all.** `staleness` (14.9%)
is byte-identical across every episode and correlates +0.003 with the error.
The only informative channel, sigma, ranks THIRD of five.

So the earlier "sigma - none = +5.4, p=0.011" is real only in the sense that
sigma beats an EMPTY channel. It is not evidence that uncertainty is the active
ingredient. Any extra structured input produces a comparable lift.

**Do not claim the cage shows sigma helping.** The defensible statement is:
"a calibrated sigma gave no measurable benefit over an uninformative input of
the same shape (+1.7 points, p=0.40, 95% CI [-2.3,+5.8], 16 seeds)."

**Power limit, state it:** the CI admits a sigma advantage up to ~5.8 points.
This is "not demonstrated", NOT "proven absent". Ruling out a 2-point effect
would need roughly 170 seeds per arm.

## Why this control mattered

Without it the cage would have been written up as "sigma significantly improves
landing success, p=0.011". That claim would not have survived a single question
from a careful reader, and the answer would have been embarrassing: the
best-performing channel in the whole experiment is a deterministic sawtooth.

Running the control arm cost 3 hours. It is now the template: **any time an
extra observation channel appears to help, test a same-shaped channel with the
information removed before believing it.**

## Standing lesson, now six for six

Every wrong conclusion in this project came from an unequal comparison or an
unchecked assumption, never from a bug in the physics:
  1. scripted beat PPO by 11.7 -> tuning asymmetry
  2. sigma beat staleness p=0.007 -> only 8 seeds
  3. sigma variance advantage -> undertuned hyperparameters
  4. rho=0.363 -> measured at the wrong time scale
  5. cage runs all 17% -> action space never normalised
  6. cage sigma p=0.011 -> input-shape artefact, caught by the control arm

# ============================================================
# DIAGNOSTIC (19 Sept): WHY sigma gave no benefit. My hypothesis was WRONG.
# ============================================================

`sensitivity.py`. No new training: take the already-trained policies, feed each
one real observations twice -- identical except the uncertainty channel is set
to the bottom and the top of its range -- and measure how far the commanded
action moves. Two references make the number readable: the same swing applied
to a channel the policy MUST use (altitude, distance to pad), and the same
measurement on the `random` arm, whose channel is meaningless by construction.

## Result

```
POINT-MASS, 10 trained sigma policies
  uncertainty channel    action change 0.2841
  altitude (must use)    action change 0.3159   (sigma is 0.90x this)
  dist to pad (must use) action change 0.3669   (sigma is 0.77x this)

CAGE, 16 trained policies per arm
  sigma  channel (real error bar, corr +0.55)  0.1507 +/- 0.0426
  random channel (zero info,      corr +0.02)  0.1733 +/- 0.0407
  difference -0.0227, p=0.124  ->  NO DIFFERENCE
```

## The hypothesis was that the policy learned to ignore sigma. It did not.

The policy is as sensitive to sigma as it is to ALTITUDE -- a channel it cannot
possibly do the task without. So "the network ignored the input" is ruled out.

But it is exactly as sensitive to a channel known to carry zero information
(p=0.124, and the point estimate actually favours the meaningless one). So the
network does not learn to discount uninformative inputs either. It conditions on
whatever it is given.

**The correct explanation is therefore: the policy READ sigma and acted on it.
Conditioning on sigma changed its behaviour without improving its outcomes.**

That is a much more specific finding than "sigma did not help", and it relocates
the failure. It is not a perception failure (the channel is read) and not a
capacity failure (the network conditions on it fine). Either sigma is not
actionable in this task as posed, or the reward does not reward acting on it.

Note `approach_env._reward` computes its blind-descent gate from `self.sigma`
REGARDLESS of arm, so all three arms are already shaped toward sigma-aware
behaviour. That is a plausible reason the sigma OBSERVATION adds nothing: the
behaviour it would buy is already being trained in through the reward for every
arm. Worth testing in the varying-sigma follow-up below.

## Methodological note

A first run of this diagnostic reported only 3 policies per arm because the glob
`{arm}_s*` matched the sibling `.log` files as well as the run directories, and
the loader was silently swallowing the resulting exceptions. Fixed to glob the
model file and to print load failures. Silent `except: continue` in analysis
code is how a sample quietly halves itself.

# ============================================================
# NEXT, AS DECIDED 19 Sept: varying-sigma follow-up (SEPARATE, not a replacement)
# ============================================================

Current results are kept as-is and reported as the "constant sigma" condition.
A second, clearly-labelled experiment makes sigma vary WITHIN an episode, which
is both more realistic (uncertainty changes as altitude and viewpoint change)
and repairs a known structural flaw: with sigma constant, PerchRL's staleness
channel is a deterministic sawtooth that CANNOT carry error information, so the
"staleness does not replicate" result is partly an artefact of the test design.

This gives sigma a genuine job -- track changing uncertainty rather than read a
per-episode constant -- and it makes the staleness comparison fair for the first
time. Report both conditions.
