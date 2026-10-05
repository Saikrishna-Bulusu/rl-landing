#!/usr/bin/env python3
"""
sweep_search_k.py -- how big should the search box be, in units of sigma?

THE FIRST PLACE SIGMA CHANGES WHAT THE AIRCRAFT DOES

Every earlier result in this project used sigma as an OBSERVATION and asked
whether a policy given it did better. This asks something more direct and more
defensible: the search box is sized at k * sigma, so k is a dial with an
obvious cost on each side. Too small and the aircraft gives up on deliveries
that were findable. Too large and it burns the battery it needed to get home,
sweeping ground the pad was never on.

If sigma is calibrated, there is a k where that trade-off is best, and it
should be near the value that covers a two-dimensional error -- about 2.5. If
sigma were uninformative, every k would be equally bad and the curve would be
flat. The shape of the curve is the result.
"""

import argparse
import json
import time

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "px4_gazebo"))
import px4_mission as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=float, nargs="+", default=[1.0, 2.5, 4.0])
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", default="search_k_sweep.json")
    a = ap.parse_args()

    results = {}
    for k in a.ks:
        rows = []
        for i in range(a.runs):
            print(f"--- k={k}  mission {i+1}/{a.runs} ---", flush=True)
            mi = M.Mission(seed=i, search_k=k, expect_ids=M.PAD_IDS,
                           policy=a.policy)
            r = mi.run(verbose=False)
            mi.close()
            rows.append(r)
            print(f"  {r['outcome']:<8} miss "
                  f"{(r['miss_m'] if r['miss_m'] is not None else -1):5.2f} m  "
                  f"{r['t_s']:5.0f} s  sigma {r['sigma_m']:.1f}", flush=True)
            time.sleep(2)
        land = [r for r in rows if r["outcome"] == "landed"]
        results[str(k)] = dict(
            n=len(rows),
            delivered=len(land) / len(rows),
            aborted=sum(r["outcome"] == "abort" for r in rows) / len(rows),
            median_time_s=float(np.median([r["t_s"] for r in rows])),
            median_miss_m=(float(np.median([r["miss_m"] for r in land]))
                           if land else None),
            rows=[{kk: r[kk] for kk in
                   ("outcome", "miss_m", "t_s", "sigma_m", "scene")} for r in rows])
        json.dump(results, open(a.out, "w"), indent=2)

    print(f"\n{'k':>5} {'delivered':>10} {'aborted':>8} {'median s':>9} {'miss m':>8}")
    for k, v in results.items():
        mm = "  --  " if v["median_miss_m"] is None else f"{v['median_miss_m']:.2f}"
        print(f"{k:>5} {100*v['delivered']:9.0f}% {100*v['aborted']:7.0f}% "
              f"{v['median_time_s']:9.0f} {mm:>8}")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
