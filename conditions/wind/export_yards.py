#!/usr/bin/env python3
"""
export_yards.py -- the first N flights of an MJX evaluation seed (land_env.evaluate: episode i uses
split(PRNGKey(seed), n)[i]) as plain numbers, so Gazebo flies the same yards, starts and wind directions:
outputs/gazebo_wind/yards_s<seed>.json.   python conditions/wind/export_yards.py --seed 10000 --n 512 --first 40
"""
import argparse
import json
import math

import jax
import numpy as np

import land_env as E
import x500_mjx as X

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=10_000)
ap.add_argument("--n", type=int, default=512, help="the evaluation's batch size (the keys depend on it)")
ap.add_argument("--first", type=int, default=40)
a = ap.parse_args()
keys = jax.random.split(jax.random.PRNGKey(a.seed), a.n)
out = []
for i in range(a.first):
    st = E.reset(keys[i], 0.0)
    y = {k: np.array(v).tolist() for k, v in st["yard"].items()}
    p = np.array(st["sim"]["d"].qpos[:3])
    out.append(dict(i=i, start=p.tolist(), yaw=float(st["yaw"]), wind_dir=float(st["wind"]["th"]),
                    bias=np.array(st["bias"]).tolist(), yard=y))
json.dump(out, open(f"{X.R}/outputs/gazebo_wind/yards_s{a.seed}.json", "w"), indent=1)
print(len(out), "yards ->", f"{X.R}/outputs/gazebo_wind/yards_s{a.seed}.json")
