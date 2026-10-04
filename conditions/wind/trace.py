#!/usr/bin/env python3
"""
trace.py -- one flight, rule against learner, the same yard, start and gusts (MJX, evaluation seed 10000, episode i):
height, distance from the pad against the rule's gate, and the learner's push against the gust that moves the drone.
    python wind/trace.py runs/fly_s0_best.npz --w10 15 --episode 3     -> gazebo_wind/figs/trace_w15_e3.png
"""
import argparse
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


def run(policy_fn, w10, i, fly, decide, shield, n=512, seed=10_000):
    """Episode i of the evaluation batch (the batch size fixes the keys), stepped and recorded."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    st = jax.vmap(lambda k: E.reset(k, w10))(keys)
    step_keys = jax.random.split(jax.random.PRNGKey(seed + 1), int(E.T_MAX / E.DT_C) + 1)

    def one(st, k):
        ks = jax.random.split(k, n * 2).reshape(n, 2, 2)
        obs = jax.vmap(E.obs_of)(st, ks[:, 0])
        act = policy_fn(obs, st)
        new, r, done = jax.vmap(lambda s, a, kk: E.step(s, a, kk, fly, decide, shield))(st, act, ks[:, 1])
        keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        w = st["wind"]
        th = w["th"]
        amp = jnp.where(w["gt"] >= 0, w["gA"] * 0.5 * (1 - jnp.cos(2 * math.pi * w["gt"] / w["gT"])), 0.0)
        # the wind's push beyond its mean, along the mean wind direction: turbulence + gust
        dev_along = w["uvw"][:, 0] + amp * jnp.cos(w["gth"] - th)
        a_ff = jnp.clip(act[:, :2], -1, 1) * E.A_FF * fly
        ff_along = a_ff[:, 0] * jnp.cos(th) + a_ff[:, 1] * jnp.sin(th)
        p = st["sim"]["d"].qpos[:, :3]
        return jax.tree_util.tree_map(keep, st, new), (p[i], ff_along[i], dev_along[i], st["done"][i], act[i, 2])
    st, tr = jax.jit(lambda s, ks: jax.lax.scan(one, s, ks))(st, step_keys)
    return int(st["outcome"][i]), float(st["t"][i]), [np.array(x) for x in tr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--w10", type=float, default=15.0)
    ap.add_argument("--episode", type=int, default=0)
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, rms, meta = P.load(a.ckpt)
    learner, fly_, decide_, shield_ = P.load_policy(a.ckpt)
    meta = dict(meta, fly=fly_, decide=decide_, shield=shield_)
    names = {1: "landed", 2: "crashed", 3: "hard/off pad", 4: "gave up", 0: "gave up"}
    rule = lambda obs, st: jnp.zeros((obs.shape[0], 3))
    out = {}
    for who, pol, f_, d_, sh in (("rule", rule, False, False, 0.0),
                                 ("learner", learner, meta["fly"], meta["decide"], meta.get("shield", 0.0))):
        out[who] = run(pol, a.w10, a.episode, f_, d_, sh)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    cols = {"rule": "#b8620a", "learner": "#1f7a4d"}
    for who, (oc, t_end, (pos, ff, dev, done, mode)) in out.items():
        live = ~done
        t = np.arange(len(pos))[live] * E.DT_C
        ax[0].plot(t, pos[live, 2], color=cols[who], lw=2.2, label=f"{who}: {names[oc]} at {t_end:.0f} s")
        d = np.hypot(pos[live, 0], pos[live, 1])
        ax[1].plot(t, d, color=cols[who], lw=1.8, label=who)
        if who == "learner":
            ax[2].plot(t, dev[live], color="#9aa0a6", lw=1.6, label="the wind beyond its mean (m/s, along it)")
            ax[2].plot(t, ff[live], color=cols[who], lw=1.8, label="learner's push (m/s$^2$, along the wind)")
    h = np.linspace(0, 6, 50)
    ax[1].axhline(0.75, color="#555", lw=1, ls=":", label="pad edge (0.75 m)")
    ax[0].set_xlabel("seconds"); ax[0].set_ylabel("height (m)"); ax[0].legend(frameon=False, fontsize=10)
    ax[0].set_title(f"wind {a.w10:.0f} m/s at 10 m, yard {a.episode}")
    ax[1].set_xlabel("seconds"); ax[1].set_ylabel("distance from pad centre (m)"); ax[1].set_ylim(0, 3)
    ax[1].legend(frameon=False, fontsize=10); ax[1].set_title("centring (the rule waits below 0.3 m + 0.1/m)")
    ax[2].axhline(0, color="#ccc", lw=1)
    ax[2].set_xlabel("seconds"); ax[2].legend(frameon=False, fontsize=10); ax[2].set_title("pushing against the gusts")
    fig.tight_layout()
    os.makedirs(f"{R}/gazebo_wind/figs", exist_ok=True)
    path = f"{R}/gazebo_wind/figs/trace_w{a.w10:g}_e{a.episode}.png"
    fig.savefig(path, dpi=110)
    corr = np.corrcoef(out["learner"][2][2][~out["learner"][2][3]], out["learner"][2][1][~out["learner"][2][3]])[0, 1]
    print(path, {w: (names[o], round(t_, 1)) for w, (o, t_, _) in out.items()}, f"push vs gust correlation {corr:+.2f}")


if __name__ == "__main__":
    main()
