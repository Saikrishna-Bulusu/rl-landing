#!/usr/bin/env python3
"""
baselines.py -- scripted baselines for the approach task, each given a fair
tuning budget, plus the evaluation harness the RL policy is scored against.

WHY THIS FILE EXISTS BEFORE train.py

"Leveling the Playing Field" (arXiv 2506.17832) re-ran classical-vs-learned
quadrotor control with the RL-favouring biases removed -- asymmetric tuning
effort, task-specific data, feedforward access -- and most of the published gaps
vanished. WaveLander's own baseline is constant descent at -0.3 m/s, a strawman
it beats 66% to 33%.

So the baselines here get a tuning sweep, on the same episodes, before the
policy is allowed to compete with them. Three of them:

    dive          straight down, no search                  (the strawman)
    fixed_spiral  descend to the ceiling, sweep a fixed disk (sigma-blind)
    sigma_spiral  sweep a disk scaled by sigma               (sigma-aware)

The third one matters most. If a hand-tuned sigma_spiral already matches a
learned sigma-conditioned policy, THAT is the finding, and it is the same shape
as GeoAnchor's finding 1 (a raw inlier threshold beats everything learned) and
MUSE's note that the empirical covariance baseline is "quite competitive".
Publish it either way -- but know it before three months are spent.

    python3 baselines.py --pool sigma_pool.npz --tune
    python3 baselines.py --pool sigma_pool.npz --episodes 400
"""

import argparse
import json

import numpy as np

from approach_env import ApproachEnv


# --- scripted policies -------------------------------------------------------
# Each takes (obs, env, params) and returns an action. They read `env` for the
# quantities a real aircraft would genuinely have -- altitude, sigma, whether
# the marker is in view -- and never for env.pad, which is ground truth.

def dive(obs, env, p):
    """Maximum descent, straight at the believed position. No search."""
    return np.array([-1.0, -1.0])


def _descend_aligned(env):
    """
    Descent rate gated on how well centred the marker is.

    The camera footprint shrinks as you descend (radius = h * tan(fov/2)), so
    diving the moment the marker appears drives it straight back out of frame.
    Descend fast when well centred, hold when near the edge. This is
    WaveLander's alignment gate, and without it every policy here scores ~2%.
    """
    reach = env.h * env.fov_tan
    dist = float(np.linalg.norm(env.pad - env.xy))
    align = np.clip(1.0 - dist / max(reach, 1e-6), 0.0, 1.0)
    return -align


def fixed_spiral(obs, env, p):
    """
    Descend to just under the detection ceiling, then sweep a disk of fixed
    radius that grows with time. Knows nothing about sigma.
    """
    return _spiral(env, p, radius_target=p["r_fixed"])


def sigma_spiral(obs, env, p):
    """Same, but the disk is scaled by the calibrated sigma."""
    return _spiral(env, p, radius_target=p["k_sigma"] * env.sigma)


def _spiral(env, p, radius_target):
    h_hold = p["h_hold"]

    if env._acquired():
        return np.array([_descend_aligned(env), -1.0])

    if env.h > h_hold:
        vz = -1.0                            # still above the search altitude
        r = 0.0
    else:
        # Hold altitude and sweep. Radius ramps from 0 outward over the search,
        # so the near field is covered before the far field.
        vz = np.clip((h_hold - env.h) / 5.0, -1.0, 1.0)
        frac = np.clip(env.t / (env.T_MAX * p["sweep_frac"]), 0.0, 1.0)
        r = radius_target * frac
    # action[1] in [-1,1] maps to radius 0..25 m
    return np.array([vz, np.clip(r / 12.5 - 1.0, -1.0, 1.0)])


POLICIES = {"dive": dive, "fixed_spiral": fixed_spiral,
            "sigma_spiral": sigma_spiral}

DEFAULTS = {"h_hold": 28.0, "r_fixed": 18.0, "k_sigma": 1.5, "sweep_frac": 0.75}


# --- evaluation --------------------------------------------------------------
def rollout(env, policy, params, seeds):
    """Run one policy over a fixed seed list. Same seeds -> same episodes."""
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        sigma0 = env.sigma
        while True:
            obs, r, term, trunc, info = env.step(policy(obs, env, params))
            if term or trunc:
                out.append({"success": info["success"], "miss_m": info["miss_m"],
                            "t_s": info["t_s"], "sigma_m": sigma0,
                            "ever_acquired": info["ever_acquired"],
                            "timeout": bool(trunc)})
                break
    return out


def summarise(rows):
    """Median and p90. Never the mean -- a miss can be 40 m and skew everything."""
    miss = np.array([r["miss_m"] for r in rows])
    sig = np.array([r["sigma_m"] for r in rows])
    ok = np.array([r["success"] for r in rows])
    s = {
        "n": len(rows),
        "success_%": 100 * ok.mean(),
        "acquired_%": 100 * np.mean([r["ever_acquired"] for r in rows]),
        "timeout_%": 100 * np.mean([r["timeout"] for r in rows]),
        "miss_median_m": float(np.median(miss)),
        "miss_p90_m": float(np.percentile(miss, 90)),
        "t_median_s": float(np.median([r["t_s"] for r in rows])),
    }
    # Success split by uncertainty. This is where a sigma-aware method should
    # separate from a blind one; an overall number can hide it entirely.
    lo, hi = np.percentile(sig, [33, 67])
    for name, m in (("low", sig <= lo), ("mid", (sig > lo) & (sig <= hi)),
                    ("high", sig > hi)):
        s[f"success_{name}_sigma_%"] = 100 * ok[m].mean() if m.any() else float("nan")
    return s


def tune(pool, name, episodes, rng, env_kw=None):
    """
    Fair tuning: random search over that policy's free parameters, scored on the
    same episodes every candidate sees. Equal budget for every baseline.
    """
    grid = {
        "dive": [{}],
        "fixed_spiral": [{"h_hold": h, "r_fixed": r, "sweep_frac": f}
                         for h in (26, 30) for r in (8, 16, 24)
                         for f in (0.5, 0.8)],
        "sigma_spiral": [{"h_hold": h, "k_sigma": k, "sweep_frac": f}
                         for h in (26, 30) for k in (0.75, 1.5, 2.5)
                         for f in (0.5, 0.8)],
    }[name]   # equal budget: 12 candidates each

    seeds = rng.integers(0, 2**31, episodes)
    env = ApproachEnv(pool=pool, arm="sigma", seed=0, **(env_kw or {}))
    best, best_p = -1.0, None
    for cand in grid:
        p = {**DEFAULTS, **cand}
        sc = summarise(rollout(env, POLICIES[name], p, seeds))["success_%"]
        if sc > best:
            best, best_p = sc, p
    return best_p, best, len(grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sigma_pool.npz")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--tune", action="store_true",
                    help="random-search each baseline's parameters first")
    ap.add_argument("--tune-episodes", type=int, default=120)
    ap.add_argument("--out", default="baselines.json")
    # The two honest difficulty knobs, per approach_env's docstring: loiter time
    # is a battery reserve and lateral speed is an airframe limit. Tighten these
    # if the baselines saturate -- never the camera constants, which are physics.
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--vxy-max", type=float, default=None)
    a = ap.parse_args()

    env_kw = {k: v for k, v in (("t_max", a.t_max), ("vxy_max", a.vxy_max))
              if v is not None}
    if env_kw:
        print(f"env overrides: {env_kw}\n")

    rng = np.random.default_rng(0)
    params = {k: dict(DEFAULTS) for k in POLICIES}

    if a.tune:
        print("tuning (equal budget per baseline)\n")
        for name in POLICIES:
            p, sc, n = tune(a.pool, name, a.tune_episodes,
                            np.random.default_rng(1), env_kw)
            params[name] = p
            extra = {k: v for k, v in p.items() if k in
                     ("h_hold", "r_fixed", "k_sigma", "sweep_frac")}
            print(f"  {name:<14} {n:>3} candidates  best {sc:5.1f}%  {extra}")
        print()

    # Held-out seeds, disjoint from tuning.
    seeds = rng.integers(2**31, 2**32 - 1, a.episodes)
    env = ApproachEnv(pool=a.pool, arm="sigma", seed=0, **env_kw)

    results = {}
    hdr = (f"{'policy':<14} {'succ%':>6} {'acq%':>6} {'miss50':>8} {'miss90':>8} "
           f"{'t50':>6} {'lo-sig':>7} {'mid':>7} {'hi-sig':>7}")
    print(hdr)
    print("-" * len(hdr))
    for name, fn in POLICIES.items():
        s = summarise(rollout(env, fn, params[name], seeds))
        results[name] = {"params": params[name], **s}
        print(f"{name:<14} {s['success_%']:6.1f} {s['acquired_%']:6.1f} "
              f"{s['miss_median_m']:7.2f}m {s['miss_p90_m']:7.2f}m "
              f"{s['t_median_s']:5.1f}s {s['success_low_sigma_%']:6.1f}% "
              f"{s['success_mid_sigma_%']:6.1f}% {s['success_high_sigma_%']:6.1f}%")

    with open(a.out, "w") as fh:
        json.dump({"episodes": a.episodes, "tuned": a.tune, "env": env_kw,
                   "results": results}, fh, indent=2)
    print(f"\nwrote {a.out}")
    print("\nRead the last three columns. If sigma_spiral only matches "
          "fixed_spiral\nat high sigma, the uncertainty channel is not "
          "carrying usable signal and\nthe RL result will not either -- "
          "investigate before training.")


if __name__ == "__main__":
    main()
