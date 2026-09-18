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
