#!/usr/bin/env python3
"""
train.py -- PPO on the approach task, one run per ablation arm.

    python3 train.py --pool sigma_pool.npz --arm sigma --steps 2000000
    python3 train.py --pool sigma_pool.npz --all --steps 2000000

THE EXPERIMENT

Three arms, identical in every respect except what the policy is told about how
wrong its position estimate is:

    none        position estimate only
    staleness   + exp(-lambda * t_since_fix)      PerchRL's rho_t proxy
    sigma       + the calibrated sigma, in metres GeoAnchor's estimator

PerchRL (arXiv 2606.03441) already showed a reliability input helps: removing
its rho_t "forc[es] the policy to fully trust the estimate even when it has
significantly deviated from the ground truth, leading to degraded performance."
So the `none` vs `staleness` gap is the replication. The claim under test is the
`staleness` vs `sigma` gap -- a stopwatch tells you the estimate is stale, a
calibrated sigma tells you it is wrong by twenty-one metres.

Report success rate SPLIT BY SIGMA. An overall number hides the entire effect:
at low sigma every arm should commit and succeed, and the arms can only separate
where the uncertainty is large.

Asymmetric actor-critic is deliberately NOT used here. WaveLander adds
observation noise to the actor and not the critic, and DPRL (arXiv 2412.06313)
gives the critic full privileged state. Both would help -- but privileged
information given to one arm and not another is exactly the kind of asymmetry
"Leveling the Playing Field" shows produces misleading comparisons. Add it to
all three arms at once or not at all.
"""

import argparse
import json
import os

import numpy as np

from approach_env import ApproachEnv


def make_env(pool, arm, seed, t_max, vxy_max):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(ApproachEnv(pool=pool, arm=arm, seed=seed,
                                   t_max=t_max, vxy_max=vxy_max))
    return _f


def evaluate(model, pool, arm, episodes, t_max, vxy_max, seed0=10_000_000):
    """Fixed seeds, so every arm sees the identical episodes and error draws."""
    env = ApproachEnv(pool=pool, arm=arm, seed=0, t_max=t_max, vxy_max=vxy_max)
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        sigma0 = env.sigma
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                rows.append({"success": info["success"], "miss_m": info["miss_m"],
                             "sigma_m": sigma0, "t_s": info["t_s"],
                             "ever_acquired": info["ever_acquired"]})
                break

    sig = np.array([r["sigma_m"] for r in rows])
    ok = np.array([r["success"] for r in rows], float)
    miss = np.array([r["miss_m"] for r in rows])
    lo, hi = np.percentile(sig, [33, 67])
    out = {
        "n": len(rows),
        "success_%": float(100 * ok.mean()),
        "acquired_%": float(100 * np.mean([r["ever_acquired"] for r in rows])),
        "miss_median_m": float(np.median(miss)),      # never the mean
        "miss_p90_m": float(np.percentile(miss, 90)),
        "t_median_s": float(np.median([r["t_s"] for r in rows])),
    }
    for name, m in (("low", sig <= lo), ("mid", (sig > lo) & (sig <= hi)),
                    ("high", sig > hi)):
        out[f"success_{name}_sigma_%"] = float(100 * ok[m].mean()) if m.any() else None
    return out


def run(arm, a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([make_env(a.pool, arm, a.seed + i, a.t_max, a.vxy_max)
                        for i in range(a.n_envs)])
    # Observations are already hand-normalised to roughly unit scale in _obs;
    # VecNormalize is here for the reward, whose magnitude swings by 100 at
    # touchdown and destabilises the value function without it.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=20.0)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024, batch_size=256, learning_rate=3e-4,
        gamma=0.995,          # 600-step episodes; 0.99 discounts touchdown away
        gae_lambda=0.97,
        ent_coef=0.008,       # WaveLander's value; the task needs exploration
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[128, 64]),
        # "" disables logging; SB3 treats any non-None value as a request and
        # then dies if tensorboard is not installed.
        tensorboard_log=a.tb or None, verbose=1, seed=a.seed,
    )
    model.learn(total_timesteps=a.steps, progress_bar=False)

    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, f"ppo_{arm}"))
    venv.save(os.path.join(a.out, f"vecnorm_{arm}.pkl"))

    return evaluate(model, a.pool, arm, a.eval_episodes, a.t_max, a.vxy_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sigma_pool.npz")
    ap.add_argument("--arm", default="sigma",
                    choices=["none", "staleness", "sigma"])
    ap.add_argument("--all", action="store_true", help="run all three arms")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--vxy-max", type=float, default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tb", default="runs/tb")
    a = ap.parse_args()

    arms = ["none", "staleness", "sigma"] if a.all else [a.arm]
    results = {}
    for arm in arms:
        print(f"\n{'=' * 70}\n  arm: {arm}\n{'=' * 70}")
        results[arm] = run(arm, a)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ablation.json"), "w") as fh:
        json.dump({"steps": a.steps, "pool": a.pool, "results": results}, fh, indent=2)

    hdr = (f"\n{'arm':<11} {'succ%':>6} {'acq%':>6} {'miss50':>8} {'miss90':>8} "
           f"{'lo-sig':>7} {'mid':>7} {'hi-sig':>7}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for arm, s in results.items():
        f = lambda k: "   n/a" if s[k] is None else f"{s[k]:6.1f}"
        print(f"{arm:<11} {s['success_%']:6.1f} {s['acquired_%']:6.1f} "
              f"{s['miss_median_m']:7.2f}m {s['miss_p90_m']:7.2f}m "
              f"{f('success_low_sigma_%')}% {f('success_mid_sigma_%')}% "
              f"{f('success_high_sigma_%')}%")
    print(f"\nwrote {a.out}/ablation.json")
    print("\nThe result is the hi-sigma column, not the overall one. Compare "
          "against\nbaselines.json -- if tuned sigma_spiral matches the sigma "
          "arm there, the\nhonest finding is that a scripted rule suffices, "
          "and that is publishable.")


if __name__ == "__main__":
    main()
