#!/usr/bin/env python3
"""
compare_physics.py -- does the measured aircraft change what the policy achieves?

The point-mass environment used to guess its own physics: a velocity lag drawn
uniformly from 0.30-0.80 s for BOTH axes, a 60 degree camera, and a
detection-ceiling formula that treated the field of view as a small angle.
Every one of those has now been replaced by a measurement taken off the PX4
airframe we actually fly (px4_response.json, camera_info).

This runs ONE trained policy against both, on the SAME episode seeds, and
reports what moved. It is the honest version of "does closing the sim-to-real
gap matter": if the numbers barely move, the gap was never the problem and
saying so is worth more than assuming it was.
"""

import argparse
import json

import numpy as np

from approach_env_v3 import ApproachEnvV3

# What the environment used to assume, before anything was measured.
OLD = dict(TAU_LAT=0.55, TAU_VER=0.55,      # mean of the old uniform draw
           FOV_DEG=60.0, PX=640, MIN_PX=20.0, PAD_SIZE=1.0)
NEW = dict(TAU_LAT=0.69, TAU_VER=0.23,
           FOV_DEG=99.7, PX=1280, MIN_PX=34.0, PAD_SIZE=1.20)


def make(cfg, small_angle, seed, arm="sigma"):
    class E(ApproachEnvV3):
        pass
    for k, v in cfg.items():
        setattr(E, k, v)
    env = E(pool="sigma_pool.npz", arm=arm, seed=seed)
    if small_angle:            # reproduce the old approximation exactly
        env.h_detect = env.PAD_SIZE * env.PX / (np.radians(env.FOV_DEG)
                                                * env.MIN_PX)
    return env


def evaluate(model, vecnorm, cfg, small_angle, n, seed0, arm="sigma"):
    outs, misses, rewards = [], [], []
    for i in range(n):
        env = make(cfg, small_angle, seed0 + i, arm)
        obs, _ = env.reset(seed=seed0 + i)
        total, done = 0.0, False
        while not done:
            o = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            a, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        outs.append(info.get("outcome", "timeout"))
        misses.append(info.get("miss_m", np.nan))
        rewards.append(total)
    return outs, np.array(misses, float), np.array(rewards)


def summarise(label, outs, misses, rewards, ceiling):
    n = len(outs)
    kinds = sorted(set(outs))
    line = "  ".join(f"{k} {100*outs.count(k)/n:5.1f}%" for k in kinds)
    ok = [m for o, m in zip(outs, misses) if o == "delivered" and np.isfinite(m)]
    print(f"  {label:22} ceiling {ceiling:5.1f} m   reward {rewards.mean():+7.1f}"
          f" +/- {rewards.std()/np.sqrt(n):4.1f}")
    print(f"  {'':22} {line}")
    if ok:
        print(f"  {'':22} median miss when delivered {np.median(ok):.2f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3_final/sigma_s0/ppo_sigma.zip")
    ap.add_argument("--vecnorm", default="v3_final/sigma_s0/vecnorm_sigma.pkl")
    ap.add_argument("--arm", default="sigma")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=90000)
    a = ap.parse_args()

    import pickle
    from stable_baselines3 import PPO
    model = PPO.load(a.model, device="cpu")
    # These runs saved a VecNormalize, but checking it rather than assuming:
    # norm_obs is False, so only the REWARD was normalised and the policy was
    # trained on raw observations. Feeding it raw observations here is
    # therefore correct -- worth verifying, because silently skipping an
    # observation normaliser the policy did expect would produce exactly the
    # kind of collapse this script is meant to detect.
    vn = None
    try:
        d = pickle.load(open(a.vecnorm, "rb")).__dict__
        assert not d.get("norm_obs"), "policy expects normalised observations"
    except FileNotFoundError:
        pass

    print(f"\n{a.n} episodes, identical seeds, policy {a.model}\n")
    # One factor at a time, so the collapse can be attributed instead of just
    # observed.
    steps = [("ASSUMED physics", dict(OLD), True)]
    c = dict(OLD); steps.append(("+ tangent ceiling", dict(c), False))
    c.update(FOV_DEG=99.7, PX=1280); steps.append(("+ real camera", dict(c), False))
    c.update(MIN_PX=34.0); steps.append(("+ reliable-decode px", dict(c), False))
    c.update(PAD_SIZE=1.20); steps.append(("+ real marker size", dict(c), False))
    c.update(TAU_LAT=0.69, TAU_VER=0.23); steps.append(("+ measured lag (ALL)", dict(c), False))
    for label, cfg, sa in steps:
        env = make(cfg, sa, 0, a.arm)
        o, m, r = evaluate(model, vn, cfg, sa, a.n, a.seed0, a.arm)
        summarise(label, o, m, r, env.h_detect)
        print()


if __name__ == "__main__":
    main()
