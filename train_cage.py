#!/usr/bin/env python3
"""
train_cage.py -- STAGE 2. The same three-arm sigma ablation, in the MuJoCo cage.

Mirrors train.py deliberately: same arms, same equal-tuning-budget discipline,
same sigma-split reporting. The only differences are the environment and that
success is defined by the aviary's own landing test rather than a miss radius.

    python3 train_cage.py --arm sigma --steps 1000000
"""

import argparse
import json
import os

import numpy as np

from cage_env import CageLandingEnv


def make_env(a, arm, seed):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(CageLandingEnv(pool=a.pool, arm=arm, rho=a.rho,
                                      sigma_scale=a.sigma_scale,
                                      render_mode=None))
    return _f


def evaluate(model, a, arm, episodes, seed0=10_000_000):
    """Fixed seeds, so every arm sees identical pad placements and error draws."""
    env = CageLandingEnv(pool=a.pool, arm=arm, rho=a.rho,
                         sigma_scale=a.sigma_scale, render_mode=None)
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        sigma0 = env.sigma
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                rows.append({"success": bool(info.get("landed", False)),
                             "h_dist": float(info.get("h_dist", np.nan)),
                             "sigma_m": sigma0})
                break
    sig = np.array([r["sigma_m"] for r in rows])
    ok = np.array([r["success"] for r in rows], float)
    d = np.array([r["h_dist"] for r in rows])
    lo, hi = np.percentile(sig, [33, 67])
    out = {"n": len(rows), "success_%": float(100 * ok.mean()),
           "h_dist_median_m": float(np.median(d)),       # never the mean
           "h_dist_p90_m": float(np.percentile(d, 90))}
    for name, m in (("low", sig <= lo), ("mid", (sig > lo) & (sig <= hi)),
                    ("high", sig > hi)):
        out[f"success_{name}_sigma_%"] = float(100 * ok[m].mean()) if m.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sigma_pool.npz")
    ap.add_argument("--arm", default="sigma",
                    choices=["none", "staleness", "sigma"])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--sigma-scale", type=float, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.008)
    ap.add_argument("--net", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--out", default="runs_cage")
    a = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([make_env(a, a.arm, a.seed + i) for i in range(a.n_envs)])
    # Unlike the point-mass env, the aviary's observation is raw MuJoCo state
    # spanning very different magnitudes, so normalise observations too.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=20.0)

    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=256,
                learning_rate=a.lr, gamma=0.995, gae_lambda=0.97,
                ent_coef=a.ent_coef, clip_range=0.2,
                policy_kwargs=dict(net_arch=a.net),
                tensorboard_log=None, verbose=1, seed=a.seed)
    model.learn(total_timesteps=a.steps, progress_bar=False)

    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, f"ppo_{a.arm}"))
    venv.save(os.path.join(a.out, f"vecnorm_{a.arm}.pkl"))

    res = evaluate(model, a, a.arm, a.eval_episodes)
    with open(os.path.join(a.out, "ablation.json"), "w") as fh:
        json.dump({"steps": a.steps, "rho": a.rho, "arm": a.arm,
                   "sigma_scale": a.sigma_scale, "results": {a.arm: res}},
                  fh, indent=2)
    print(f"\n{a.arm}: success {res['success_%']:.1f}%  "
          f"lo/mid/hi-sigma "
          f"{res['success_low_sigma_%']}/{res['success_mid_sigma_%']}/"
          f"{res['success_high_sigma_%']}")


if __name__ == "__main__":
    main()
