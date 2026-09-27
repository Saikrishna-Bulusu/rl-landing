#!/usr/bin/env python3
"""
behaviour.py -- what the wind learner does that the rule doesn't: the same MJX flights (evaluation seed 10000) flown
by both, logged step by step. Height against time, the vertical option by height, and whether its feed-forward
pushes against the gusts (its correlation with the wind's deviation from the mean, across the flight).
    python wind/behaviour.py runs/both95_s0.npz --w10 15 18       -> gazebo_wind/figs/behaviour.png, printed table
"""
import argparse
import json
import math
import os

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp

import land_env as E
import ppo as P

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)


def fly(policy_fn, w10, fly_, decide_, shield=0.0, n=512, seed=10_000):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    st = jax.vmap(lambda k: E.reset(k, w10))(keys)

    def one(st, k):
        ks = jax.random.split(k, n * 2).reshape(n, 2, 2)
        obs = jax.vmap(E.obs_of)(st, ks[:, 0])
        act = policy_fn(obs, st)
        new, r, done = jax.vmap(lambda s, a, kk: E.step(s, a, kk, fly_, decide_, shield))(st, act, ks[:, 1])
        keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        nxt = jax.tree_util.tree_map(keep, st, new)
        p = st["sim"]["d"].qpos[:, :3]
        wind_dev = st["wind"]["uvw"][:, :2]                          # turbulence (along, across the mean wind)
        a_ff = jnp.clip(act[:, :2], -1, 1) * E.A_FF * fly_
        return nxt, (p, act[:, 2], a_ff, st["wind"]["th"], st["done"], new["sim"]["d"].qvel[:, :2])
    run = jax.jit(lambda s, ks: jax.lax.scan(one, s, ks))
    st, tr = run(st, jax.random.split(jax.random.PRNGKey(seed + 1), int(E.T_MAX / E.DT_C) + 1))
    return st, [np.array(x) for x in tr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--w10", type=float, nargs="+", default=[15.0, 18.0])
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, rms, meta = P.load(a.ckpt)
    learner, fly_, decide_, shield_ = P.load_policy(a.ckpt)
    meta = dict(meta, fly=fly_, decide=decide_, shield=shield_)
    rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, len(a.w10) + 1, figsize=(5.2 * (len(a.w10) + 1), 4.4))
    table = []
    for j, w in enumerate(a.w10):
        for name, pol, f_, d_, sh, col in (("rule", rule, False, False, 0.0, "#b8620a"),
                                            ("learner", learner, meta["fly"], meta["decide"], meta.get("shield", 0.0), "#1f7a4d")):
            st, (pos, mode, a_ff, th, done, vel) = fly(pol, w, f_, d_, sh)
            live = ~done
            t = np.arange(pos.shape[0]) * E.DT_C
            h = np.where(live, pos[..., 2], np.nan)
            med = np.nanmedian(h, axis=1)
            ax[j].plot(t, med, color=col, lw=2.2, label=name)
            ax[j].fill_between(t, np.nanpercentile(h, 25, axis=1), np.nanpercentile(h, 75, axis=1), color=col, alpha=0.15)
            out = np.array(st["outcome"])
            row = dict(w10=w, who=name, landed=float((out == 1).mean()), t_landed=float(np.array(st["t"])[out == 1].mean()) if (out == 1).any() else None)
            if name == "learner":
                bands = [(20, 40), (8, 20), (2, 8), (0, 2)]
                row["modes_by_height"] = {f"{lo}-{hi} m": np.round(np.bincount(mode[live & (pos[..., 2] >= lo) & (pos[..., 2] < hi)].astype(int), minlength=5)
                                                            / max(1, (live & (pos[..., 2] >= lo) & (pos[..., 2] < hi)).sum()), 2).tolist() for lo, hi in bands}
                # feed-forward against the wind: its component along the mean wind (negative = into the wind)
                ea = np.stack([np.cos(th), np.sin(th)], -1)
                along = (a_ff * ea).sum(-1)
                row["ff_along_wind_mean"] = float(np.nanmean(np.where(live, along, np.nan)))
                row["ff_rms"] = float(np.sqrt(np.nanmean(np.where(live[..., None], a_ff, np.nan) ** 2)))
            table.append(row)
        ax[j].set_title(f"wind {w:.0f} m/s at 10 m"); ax[j].set_xlabel("seconds"); ax[j].set_ylabel("height (m), median and middle half")
        ax[j].set_xlim(0, 90); ax[j].legend(frameon=False)
    # the vertical option by height at the strongest wind
    last = [r for r in table if r["who"] == "learner"][-1]
    labels = ["rule's own", "down 1.5", "down 0.6", "hold", "up 1.0"]
    bands = list(last["modes_by_height"].keys())
    bottom = np.zeros(len(bands))
    colours = ["#b8620a", "#1f7a4d", "#7fb99b", "#9aa0a6", "#2b6cb0"]
    for k in range(5):
        v = np.array([last["modes_by_height"][b][k] for b in bands]) * 100
        ax[-1].barh(bands, v, left=bottom, color=colours[k], label=labels[k], edgecolor="white")
        bottom += v
    ax[-1].set_xlabel("share of decisions (%)"); ax[-1].set_title(f"learner's vertical choice, {last['w10']:.0f} m/s")
    ax[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    os.makedirs(f"{R}/gazebo_wind/figs", exist_ok=True)
    fig.savefig(f"{R}/gazebo_wind/figs/behaviour.png", dpi=110)
    json.dump(table, open(f"{R}/gazebo_wind/figs/behaviour.json", "w"), indent=1)
    for r in table:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
