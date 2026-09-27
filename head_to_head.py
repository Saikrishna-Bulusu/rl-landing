#!/usr/bin/env python3
"""
head_to_head.py -- the trained policy against the analytical rules, at n large
enough to mean something.

Every agent sees identical episodes: same seeds, same scenes, same error draws.
Reports delivery rate with a binomial confidence interval, flight time, and the
outcome mix, because "delivered more often" and "delivered faster" are
different claims and the rule beat the policy on the second one last time.
"""

import argparse
import json

import numpy as np

from approach_env_v4 import ApproachEnvV4
import rule_agent as R


def ci(k, n):
    """Wilson interval: honest at the ceiling, where normal approximation is not."""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--k", type=float, default=2.5)
    ap.add_argument("--t-max", type=float, default=60.0)
    ap.add_argument("--sigma-scale", type=float, default=1.75)
    ap.add_argument("--policies", nargs="*", default=[])
    ap.add_argument("--out", default="head_to_head.json")
    a = ap.parse_args()

    kw = dict(pool="sigma_pool.npz", arm="sigma", seed=0, t_max=a.t_max,
              rho=0.707, sigma_scale=a.sigma_scale)
    env = ApproachEnvV4(**kw)

    agents = {
        f"rule  k={a.k} (latest fix)": R.SigmaRule(env, a.k),
        f"rule  k={a.k} (running mean)": R.SigmaRuleMean(env, a.k),
        "rule  fixed 25 m (sigma-blind)": R.FixedRule(env, 25.0),
    }
    for p in a.policies:
        agents[f"policy {p.split('/')[-2]}"] = R.PolicyAgent(p)

    out = {}
    print(f"\n{a.n} episodes, identical seeds for every agent "
          f"(budget {a.t_max:.0f} s, sigma x{a.sigma_scale})\n")
    print(f"{'agent':<32} {'delivered':>20} {'median s':>9} {'miss m':>7} {'reward':>8}")
    for name, ag in agents.items():
        rows = [R.rollout(ag, env, 5_000_000 + i) for i in range(a.n)]
        ok = sum(r["success"] for r in rows)
        lo, hi = ci(ok, len(rows))
        land = [r for r in rows if r["success"]]
        t = float(np.median([r["t_s"] for r in land])) if land else float("nan")
        mm = float(np.median([r["miss_m"] for r in land])) if land else float("nan")
        rw = float(np.mean([r["reward"] for r in rows]))
        print(f"{name:<32} {100*ok/len(rows):6.1f}% [{100*lo:4.1f},{100*hi:5.1f}] "
              f"{t:9.1f} {mm:7.2f} {rw:8.1f}")
        out[name] = dict(delivered=ok / len(rows), ci=[lo, hi], n=len(rows),
                         median_t_s=t, median_miss_m=mm, mean_reward=rw)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
