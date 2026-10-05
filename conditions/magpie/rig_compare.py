#!/usr/bin/env python3
"""
rig_compare.py -- does the fast simulator's rig (rig_sense.Rig: a calibrated detection model +
the tracker) behave like the real thing in Gazebo (the detector on rendered pictures + the same
tracker)? Tracked share, position error and direction error by the magpie's distance, from Gazebo
rig flights (*_rig.jsonl, gz_learner.py --sense rig) beside the fast simulator flying the planner
against the hunter with the rig, same world.

    python rig_compare.py GAZEBO_RIG_JSONL [...] [--n 512]
"""
import argparse
import json
import sys

import numpy as np

BANDS = (0, 10, 20, 40, 60, 100, 1e9)


def table(rows):
    """rows: (distance, tracked, pos_err or nan, dir_err or nan) -> per-band summary."""
    a = np.array(rows, float)
    out = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() == 0:
            continue
        t = m & (a[:, 1] > 0)
        out.append((lo, hi, int(m.sum()), a[m, 1].mean(), np.nanmedian(a[t, 2]) if t.any() else np.nan,
                    np.nanmedian(a[t, 3]) if t.any() else np.nan))
    return out


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9), -1, 1))))


def gazebo(paths):
    rows = []
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            rel = np.array(r["bird"]) - np.array(r["drone"])
            o = r["out"]
            est = np.array(o["rel"])
            rows.append((np.linalg.norm(rel), float(o["seen"]), np.linalg.norm(est - rel) if o["seen"] else np.nan,
                         ang(est, rel) if o["seen"] else np.nan))
    return rows


def fast(n):
    import torch
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "fast_sim"))
    import urban_agents as A
    import urban_env as U
    env = U.UrbanEnv(n, "corridor", seed=2026, map_path=f"{U.HERE}/logan_map_lidar.npz", nav="gps", p_swoop=1.0,
                     bird_sense="rig", magpie="hunter", device="cpu")
    agent, rows = A.MPC("obs"), []
    o = env.reset(); agent.reset(env)
    while not bool(env.done.all()) and env.t < U.T_MAX - 1e-6:
        o = env.step(agent.act(o))
        live = ~env.done & env.bird["on"]
        rel = (env.bird["pos"] - env.pos)[live]
        seen, est = o["bird_seen"][live], o["bird_rel"][live]
        d = rel.norm(dim=-1)
        pe = (est - rel).norm(dim=-1)
        de = torch.rad2deg(torch.acos(((est * rel).sum(-1) / (est.norm(dim=-1) * d).clamp(min=1e-9)).clamp(-1, 1)))
        rows += list(zip(d.tolist(), seen.float().tolist(), torch.where(seen, pe, torch.nan).tolist(),
                         torch.where(seen, de, torch.nan).tolist()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gazebo", nargs="+")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    g, f = table(gazebo(a.gazebo)), table(fast(a.n))
    print("magpie distance | Gazebo (real detector): samples, tracked, pos err, dir err | fast sim (model): samples, tracked, pos err, dir err")
    fd = {(lo, hi): r for lo, hi, *r in f}
    for lo, hi, n, t, pe, de in g:
        r = fd.get((lo, hi))
        fs = f"{r[0]:6d} {100 * r[1]:6.1f}% {r[2]:6.1f} m {r[3]:5.1f} deg" if r else "   --"
        print(f"  {lo:3.0f}-{hi if hi < 1e8 else 999:3.0f} m | {n:5d} {100 * t:6.1f}% {pe:6.1f} m {de:5.1f} deg | {fs}")


if __name__ == "__main__":
    main()
