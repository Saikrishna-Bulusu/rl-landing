#!/usr/bin/env python3
"""
train_v2.py -- PPO on the varying-sigma (scene-based) task.

Differs from train.py in three ways, each forced by a measurement:

  --sigma-scale   multiplies the replayed error AND its sigma. At the real
                  GeoAnchor accuracy (scale 1.0) the task is saturated: 75.9%
                  of fixes already land inside the 17.7 m camera footprint and a
                  dive policy scores 96.4%. The discriminating band is 1.5-2.0.

  evaluation is BY SCENE, not by first-fix sigma. Once sigma varies within an
  episode, labelling an episode by its sigma at reset is meaningless --
  corr(first-fix sigma, episode-mean sigma) = +0.179. Scenes have real,
  reportable difficulty (Scene_10 and Scene_22 contain no easy fixes at all).

  a `random` control arm. In the cage a meaningless channel moved the policy as
  much as sigma did (p=0.124), so no positive result is believable without it.
"""

import argparse
import collections
import json
import os

import numpy as np

from approach_env_v5 import ApproachEnvV5 as ApproachEnvV2


def _tuned(a):
    """Difficulty is set by class constants, so bind them per run."""
    from approach_env_v5 import ApproachEnvV5 as _B

    class E(_B):
        N_OBST = (a.n_obst, a.n_obst + 2)
        OBST_H = (a.obst_h * 0.6, a.obst_h)
        PAD_CLEAR = a.pad_clear
        OBST_R = (a.obst_r * 0.5, a.obst_r)
    return E


def make_env(a, arm, seed):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(_tuned(a)(pool=a.pool, arm=arm, seed=seed,
                                     t_max=a.t_max, rho=a.rho,
                                     sigma_scale=a.sigma_scale,
                                     id_check=a.id_check, energy=a.energy,
                                     history=a.history, steer=a.steer,
                                     obstacles=a.obstacles))
    return _f


def evaluate(model, a, arm, episodes, seed0=10_000_000):
    """Fixed seeds: every arm sees identical scenes and identical error draws."""
    env = _tuned(a)(pool=a.pool, arm=arm, seed=0, t_max=a.t_max,
                        rho=a.rho, sigma_scale=a.sigma_scale,
                        id_check=a.id_check, energy=a.energy,
                        history=a.history, steer=a.steer,
                        obstacles=a.obstacles)
    rows = []
    for i in range(episodes):
        obs, info = env.reset(seed=seed0 + i)
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                rows.append({"success": info["success"], "miss_m": info["miss_m"],
                             "scene": info["scene"], "t_s": info["t_s"],
                             "ever_acquired": info["ever_acquired"],
                             "outcome": info["outcome"],
                             "battery": info["battery"],
                             "saw_decoy": info["saw_decoy"],
                             "decoy_present": info["decoy_present"]})
                break
    ok = np.array([r["success"] for r in rows], float)
    miss = np.array([r["miss_m"] for r in rows])
    out = {
        "n": len(rows),
        "success_%": float(100 * ok.mean()),
        "acquired_%": float(100 * np.mean([r["ever_acquired"] for r in rows])),
        "miss_median_m": float(np.median(miss)),     # never the mean
        "miss_p90_m": float(np.percentile(miss, 90)),
        "t_median_s": float(np.median([r["t_s"] for r in rows])),
    }
    # THE GUARDRAIL METRICS. Success rate alone cannot distinguish "aborted
    # safely" from "landed in the wrong garden", and that distinction is the
    # entire point of v3.
    oc = collections.Counter(r["outcome"] for r in rows)
    n = max(len(rows), 1)
    out["outcomes_%"] = {k: round(100 * v / n, 2) for k, v in oc.items()}
    risky = oc.get("risky_landing_had_options", 0)
    out["reckless_landing_%"] = round(100 * risky / n, 2)
    out["rtl_%"] = round(100 * oc.get("rtl", 0) / n, 2)
    # Deliveries made while a neighbour's marker had been seen: with the safety
    # layer on this should be impossible to reach via the decoy.
    dec = [r for r in rows if r["decoy_present"]]
    out["decoy_episodes"] = len(dec)
    out["decoy_seen_%"] = round(100 * np.mean([r["saw_decoy"] for r in dec]), 2) if dec else None
    out["decoy_ep_success_%"] = round(100 * np.mean([r["success"] for r in dec]), 2) if dec else None

    by = collections.defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r["success"])
    out["by_scene"] = {k: {"n": len(v), "success_%": float(100 * np.mean(v))}
                       for k, v in sorted(by.items())}
    # The headline: scenes with NO easy fixes. Measured, not assumed.
    hard = [k for k in by if k in ("Scene_10", "Scene_22")]
    if hard:
        h = np.concatenate([by[k] for k in hard])
        out["success_hard_scenes_%"] = float(100 * h.mean())
        out["n_hard"] = int(len(h))
    return out


def run(arm, a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([make_env(a, arm, a.seed + i) for i in range(a.n_envs)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=20.0)
    model = PPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=a.batch_size,
                learning_rate=a.lr, gamma=0.995, gae_lambda=0.97,
                ent_coef=a.ent_coef, clip_range=0.2,
                policy_kwargs=dict(net_arch=a.net),
                tensorboard_log=a.tb or None, verbose=1, seed=a.seed)
    model.learn(total_timesteps=a.steps, progress_bar=False)
    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, f"ppo_{arm}"))
    venv.save(os.path.join(a.out, f"vecnorm_{arm}.pkl"))
    return evaluate(model, a, arm, a.eval_episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sigma_pool.npz")
    ap.add_argument("--arm", default="sigma",
                    )
    ap.add_argument("--steps", type=int, default=2_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=45.0)
    ap.add_argument("--rho", type=float, default=0.707)
    ap.add_argument("--sigma-scale", type=float, default=1.75)
    ap.add_argument("--obstacles", type=int, default=1,
                    help="trees: occlusion + no-fly, sensed not mapped")
    ap.add_argument("--n-obst", type=int, default=5)
    ap.add_argument("--obst-h", type=float, default=14.0)
    ap.add_argument("--pad-clear", type=float, default=0.5)
    ap.add_argument("--obst-r", type=float, default=5.0,
                    help="max canopy radius; min is 0.5x this")
    ap.add_argument("--energy", type=int, default=1,
                    help="charge per-step energy and stop paying to loiter")
    ap.add_argument("--history", type=int, default=1,
                    help="show the running mean of fixes, not just the latest")
    ap.add_argument("--steer", type=int, default=1,
                    help="0 fixed spiral, 1 polar bearing, 2 cartesian offset")
    ap.add_argument("--id-check", type=int, default=1,
                    help="1 = ArUco identity safety layer ON (a wrong marker "
                         "reads as no marker); 0 = OFF, a decoy can fool it")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--net", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--out", default="runs_v3")
    ap.add_argument("--tb", default="")
    a = ap.parse_args()

    res = {a.arm: run(a.arm, a)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ablation.json"), "w") as fh:
        json.dump({"steps": a.steps, "rho": a.rho, "sigma_scale": a.sigma_scale,
                   "results": res}, fh, indent=2)
    s = res[a.arm]
    per = ", ".join("{} {:.0f}%".format(k, v["success_%"])
                    for k, v in s["by_scene"].items())
    print("\n{}: overall {:.1f}%  hard {:.1f}%  | RTL {:.1f}%  RECKLESS {:.1f}%  ({})".format(
        a.arm, s["success_%"], s.get("success_hard_scenes_%", float("nan")),
        s["rtl_%"], s["reckless_landing_%"], per))


if __name__ == "__main__":
    main()
