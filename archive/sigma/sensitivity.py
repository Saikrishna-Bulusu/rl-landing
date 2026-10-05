#!/usr/bin/env python3
"""
sensitivity.py -- does the trained policy actually USE the uncertainty channel?

    python3 sensitivity.py

WHY

Three experiments now agree that a calibrated sigma gives a landing policy no
measurable advantage. But "sigma did not help" and "the policy ignored sigma"
are different claims, and only the second explains anything. This measures the
second directly, with no new training.

METHOD

Collect real observations from a rollout. Feed each one to the trained policy
twice, identical except that the uncertainty channel is set to the bottom and
top of its range. Measure how much the commanded action moves.

Two reference points make the number interpretable:
  - the same swing applied to a channel the policy MUST use (altitude, or
    distance to the pad). This is what "the policy uses this input" looks like.
  - the same measurement on the `random` arm, whose channel is meaningless by
    construction. This is what "the policy ignores this input" looks like.

A sigma sensitivity near the random arm's means the channel was learned to be
noise. A sigma sensitivity near the altitude reference means it was used.
"""

import glob
import os
import sys

import numpy as np


def load(run_dir, arm, cage):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    m = PPO.load(os.path.join(run_dir, f"ppo_{arm}.zip"), device="cpu")
    vn = None
    if cage:
        p = os.path.join(run_dir, f"vecnorm_{arm}.pkl")
        if os.path.exists(p):
            vn = VecNormalize.load(p, venv=None) if False else None
            import pickle
            with open(p, "rb") as fh:
                vn = pickle.load(fh)
    return m, vn


def norm(obs, vn):
    """Apply the saved observation normalisation, if the run used one."""
    if vn is None:
        return obs
    return np.clip((obs - vn.obs_rms.mean) /
                   np.sqrt(vn.obs_rms.var + vn.epsilon),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)


def collect(env, n=600):
    """Real observations, from a policy-free rollout."""
    out = []
    s = 0
    while len(out) < n:
        o, _ = env.reset(seed=90000 + s); s += 1
        for _ in range(40):
            out.append(o.copy())
            o, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
    return np.asarray(out[:n], np.float32)


def swing(model, vn, obs, idx, lo, hi):
    """Mean absolute change in the commanded action when channel idx goes lo->hi."""
    a = obs.copy(); a[:, idx] = lo
    b = obs.copy(); b[:, idx] = hi
    pa, _ = model.predict(norm(a, vn), deterministic=True)
    pb, _ = model.predict(norm(b, vn), deterministic=True)
    return float(np.mean(np.abs(pa - pb)))


def report(title, envfn, run_glob, unc_idx, unc_lo, unc_hi,
           refs, cage, arms=("sigma", "random")):
    print(f"\n{'='*66}\n{title}\n{'='*66}")
    rows = {}
    for arm in arms:
        # glob the MODEL FILE, not the directory pattern: "{arm}_s*" also
        # matches the sibling .log files and silently halves the sample.
        dirs = [os.path.dirname(x) for x in
                sorted(glob.glob(os.path.join(run_glob.format(arm=arm),
                                              f"ppo_{arm}.zip")))][:10]
        if not dirs:
            continue
        unc, ref = [], {k: [] for k in refs}
        for d in dirs:
            try:
                m, vn = load(d, arm, cage)
            except Exception as ex:
                print(f"    !! could not load {d}: {type(ex).__name__}: {ex}")
                continue
            if cage and vn is None:
                print(f"    !! {d}: cage run has no saved normaliser -- "
                      "observations would be fed unscaled")
            env = envfn(arm)
            obs = collect(env)
            unc.append(swing(m, vn, obs, unc_idx, unc_lo, unc_hi))
            for name, (i, lo, hi) in refs.items():
                ref[name].append(swing(m, vn, obs, i, lo, hi))
        if not unc:
            continue
        rows[arm] = (np.mean(unc), {k: np.mean(v) for k, v in ref.items()},
                     len(unc), np.asarray(unc))
    for arm, (u, r, n, raw) in rows.items():
        label = "uncertainty channel" if arm == "sigma" else "MEANINGLESS channel"
        print(f"\n  arm '{arm}'  ({n} trained policies)")
        print(f"    {label:<24} swing -> action change {u:.4f}")
        for name, v in r.items():
            ratio = u / v if v > 1e-9 else float("inf")
            print(f"    {name:<24} swing -> action change {v:.4f}"
                  f"   (channel is {ratio:5.2f}x this)")
    if "sigma" in rows and "random" in rows:
        s, rnd = rows["sigma"][0], rows["random"][0]
        print(f"\n  VERDICT: sigma channel moves the action {s:.4f}; a channel "
              f"KNOWN to be\n           meaningless moves it {rnd:.4f}. "
              f"Ratio {s/rnd if rnd>1e-9 else float('inf'):.2f}x.")
    return rows


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tello"))
    import approach_env, cage_env

    report(
        "POINT-MASS (approach_env), rho=0.707, 20-seed runs",
        lambda arm: approach_env.ApproachEnv(pool="sigma_pool.npz", arm=arm,
                                             seed=0, t_max=45, rho=0.707),
        "final07/{arm}_s*",
        unc_idx=3, unc_lo=8.75 / 50.0, unc_hi=80.0 / 50.0,
        refs={"altitude (must use)": (0, 10 / 60.0, 60 / 60.0),
              "dist to pad (must use)": (2, 0.0, 1.0)},
        cage=False, arms=("sigma",))

    report(
        "CAGE (cage_env), rho=0.707, 16-seed runs",
        lambda arm: cage_env.CageLandingEnv(pool="sigma_pool.npz", arm=arm,
                                            rho=0.707, render_mode=None),
        "{d}/{{arm}}_s*".format(d="cage_final2"),
        unc_idx=26, unc_lo=0.18, unc_hi=0.59,
        refs={"altitude (must use)": (22, 0.1, 2.5),
              "dist to pad (must use)": (25, 0.0, 1.5)},
        cage=True, arms=("sigma",))

    # The random arm lives in a different directory.
    report(
        "CAGE control arm, for reference",
        lambda arm: cage_env.CageLandingEnv(pool="sigma_pool.npz", arm=arm,
                                            rho=0.707, render_mode=None),
        "ctrl_final/{arm}_s*",
        unc_idx=26, unc_lo=0.18, unc_hi=0.59,
        refs={"altitude (must use)": (22, 0.1, 2.5),
              "dist to pad (must use)": (25, 0.0, 1.5)},
        cage=True, arms=("random",))


if __name__ == "__main__":
    main()
