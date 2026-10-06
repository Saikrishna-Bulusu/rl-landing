#!/usr/bin/env python3
"""
stray_check.py -- does MJX fly the Gazebo ladder's own test yards the way Gazebo did (PLAN.md Phase 2a)? The first 20
yards of the test flights (seed 10000), rule and paper decider s0, W10 15 and 18, both simulators: highest tilt, time
past 48 deg, and the worst stray from the pad after first reaching it. Gazebo's side: outputs/gazebo_wind/ladder_rule
(30 Sept). 6 Oct: before the twin's CoM / body-drag fix MJX's worst stray was 20 m where Gazebo's was 286 m.

    python conditions/wind/stray_check.py      (~3 min on the RTX 5070)
"""
import csv
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
import jax
import jax.numpy as jnp
import numpy as np

import land_env as E
import ppo
import x500_mjx as X

HERE = os.path.dirname(os.path.abspath(__file__))
GZ = f"{X.R}/outputs/gazebo_wind/ladder_rule"


def metrics(t, x, y, tilt):
    d = np.hypot(x, y); reach = np.argmax(d < 2.0) if (d < 2.0).any() else None
    drift = float(d[reach:].max()) if reach is not None else float("nan")      # worst stray after first reaching the pad
    return float(tilt.max()), float((tilt > 48).mean()), drift


def gz(tag, w):
    out = []
    for e in range(20):
        f = f"{GZ}/e{e}_w{w:g}_{tag}.csv"
        if not os.path.exists(f): continue
        c = np.array([[float(r[k]) for k in ("t", "x", "y", "tilt")] for r in csv.DictReader(open(f))])
        if len(c) > 10: out.append(metrics(c[:, 0], c[:, 1], c[:, 2], c[:, 3]))
    return np.array(out)


def mjx(ck, w, n=1024, seed=10_000, k=20):
    pol, fly, dec, sh = ppo.load_policy(ck) if ck else ((lambda o, s: jnp.zeros((o.shape[0], 3))), False, False, 0.0)
    st = jax.vmap(lambda kk: E.reset(kk, w))(jax.random.split(jax.random.PRNGKey(seed), n))
    def one(st, kk):
        ks = jax.random.split(kk, n * 2).reshape(n, 2, 2)
        obs = jax.vmap(E.obs_of)(st, ks[:, 0]); act = pol(obs, st)
        new, r, done = jax.vmap(lambda s, a, q: E.step(s, a, q, fly, dec, sh, 0.0))(st, act, ks[:, 1])
        up = jax.vmap(lambda q: X.qrot(q)[2, 2])(new["sim"]["d"].qpos[:k, 3:7])
        rec = jnp.concatenate([new["sim"]["d"].qpos[:k, :2], jnp.degrees(jnp.arccos(jnp.clip(up, -1, 1)))[:, None], st["done"][:k, None]], 1)
        keep = lambda a, b: jnp.where(st["done"].reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
        return jax.tree_util.tree_map(keep, st, new), rec
    _, rec = jax.jit(lambda s, ks: jax.lax.scan(one, s, ks))(st, jax.random.split(jax.random.PRNGKey(seed + 1), int(E.T_MAX / E.DT_C) + 1))
    rec = np.array(rec); out = []
    for i in range(k):
        live = rec[:, i, 3] < 0.5
        out.append(metrics(None, rec[live, i, 0], rec[live, i, 1], rec[live, i, 2]))
    return np.array(out)


def main():
    print("per flight on yards 0-19: tilt max (median / highest), share of flying time past 48 deg, worst stray after first reaching the pad (median / highest)")
    for tag, ck in (("rule", None), ("stage2_c100h100g_s0_best", f"{HERE}/runs/stage2_c100h100g_s0_best.npz")):
        for w in (15.0, 18.0):
            for sim, m in (("Gazebo", gz(tag, w)), ("MJX   ", mjx(ck, w))):
                if len(m) == 0: print(f"{tag:26s} W{w:g} {sim}: no flights"); continue
                print(f"{tag:26s} W{w:g} {sim} n={len(m):2d}: tilt {np.median(m[:,0]):4.1f} / {m[:,0].max():4.1f}  past 48: {100*m[:,1].mean():5.2f}%  stray {np.nanmedian(m[:,2]):6.2f} / {np.nanmax(m[:,2]):6.2f} m")


if __name__ == "__main__":
    main()
