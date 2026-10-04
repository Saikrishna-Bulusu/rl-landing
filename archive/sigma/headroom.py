#!/usr/bin/env python3
"""
headroom.py -- how much could ANY better decision be worth, before training one.

The lesson that cost an hour today: at a 28 s budget an agent told the exact
pad position scored 60.5% and a one-line rule scored 60.2%. The gap was 0.0
points, so every hour of training in that regime was guaranteed wasted -- the
failures were aircraft that knew exactly where to go and still ran out of
flight time. Nothing about the policy could have fixed that.

So: measure the ceiling first, in minutes, and only spend the hours where the
ceiling is above the floor. Two ceilings are reported because they answer
different questions -- see rule_agent for why.
"""

import argparse
import itertools
import json

import numpy as np

from approach_env_v5 import ApproachEnvV5
import rule_agent as R


def score(agent_fn, env, n, seed0=7_000_000):
    rows = [R.rollout(agent_fn(), env, seed0 + i) for i in range(n)]
    ok = sum(r["success"] for r in rows)
    crash = sum(r["outcome"] == "crash" for r in rows)
    return 100.0 * ok / n, 100.0 * crash / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--t-max", type=float, nargs="+", default=[45.0, 60.0])
    ap.add_argument("--sigma-scale", type=float, nargs="+", default=[1.75, 2.5])
    ap.add_argument("--n-obst", type=int, nargs="+", default=[4, 7])
    ap.add_argument("--obst-h", type=float, nargs="+", default=[10.0, 16.0])
    ap.add_argument("--pad-clear", type=float, nargs="+", default=[3.0])
    ap.add_argument("--out", default="headroom.json")
    a = ap.parse_args()

    print(f"{'t_max':>6} {'sig':>5} {'trees':>6} {'ht':>5} {'clr':>5} "
          f"{'rule':>7} {'crash':>6} {'obstOracle':>11} {'padOracle':>10} "
          f"{'headroom':>9}")
    out, best = [], (None, -1.0)
    grid = itertools.product(a.t_max, a.sigma_scale, a.n_obst, a.obst_h,
                             a.pad_clear)
    for t, sc, nob, oh, pc in grid:
        class E(ApproachEnvV5):
            N_OBST = (nob, nob + 2)
            OBST_H = (oh * 0.6, oh)
            PAD_CLEAR = pc
        env = E(pool="sigma_pool.npz", arm="sigma", seed=0, t_max=t,
                rho=0.707, sigma_scale=sc)
        rule, crash = score(lambda: R.AvoidRule(env), env, a.n)
        oo, _ = score(lambda: R.ObstacleOracle(env), env, a.n)
        po, _ = score(lambda: R.PadOracle(env), env, a.n)
        gap = oo - rule
        print(f"{t:6.0f} {sc:5.2f} {nob:6d} {oh:5.1f} {pc:5.1f} "
              f"{rule:6.1f}% {crash:5.1f}% {oo:10.1f}% {po:9.1f}% {gap:8.1f}pp")
        out.append(dict(t_max=t, sigma_scale=sc, n_obst=nob, obst_h=oh,
                        pad_clear=pc, rule=rule, crash=crash,
                        obstacle_oracle=oo, pad_oracle=po, headroom=gap))
        if gap > best[1]:
            best = ((t, sc, nob, oh, pc), gap)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nlargest obstacle-decision headroom: {best[1]:.1f} pp at "
          f"t_max={best[0][0]:.0f}s sigma x{best[0][1]} trees={best[0][2]} "
          f"height={best[0][3]:.0f}m clear={best[0][4]:.0f}m")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
