#!/usr/bin/env python3
"""
morning_report.py -- what the overnight run found, against the right baselines.

Run this when Sai wakes up. It reads every completed run and puts each number
next to the thing it has to beat, because a delivery rate on its own has never
once been the interesting part of this project.
"""

import glob
import json
import os

import numpy as np

BASE = {  # measured before the run, same env config, 200 episodes each
    "obst": dict(rule=80.5, oracle=99.0, label="35 s, sigma x3.5, 8 trees"),
    "noobst": dict(rule=91.0, oracle=100.0, label="35 s, sigma x3.5, no trees"),
}


def load(pattern):
    out = {}
    for f in sorted(glob.glob(f"overnight/{pattern}/ablation.json")):
        name = os.path.basename(os.path.dirname(f))
        cfg = name.rsplit("_s", 1)[0]
        try:
            r = json.load(open(f))["results"]
            v = r[list(r)[0]]
            out.setdefault(cfg, []).append(v["success_%"])
        except Exception:
            pass
    return out


def show(title, rows, base=None):
    if not rows:
        print(f"\n{title}: nothing finished")
        return
    print(f"\n{title}")
    for cfg, vals in sorted(rows.items()):
        a = np.array(vals, float)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        mark = ""
        if base:
            mark = "  BEATS THE RULE" if a.mean() > base["rule"] else ""
        print(f"  {cfg:<18} {a.mean():5.1f}% +/- {se:4.1f}  n={len(a)}  "
              f"{np.round(a, 1).tolist()}{mark}")
    if base:
        print(f"  {'-- best rule':<18} {base['rule']:5.1f}%        ({base['label']})")
        print(f"  {'-- oracle':<18} {base['oracle']:5.1f}%        ceiling, knows the pad")


def main():
    done = len(glob.glob("overnight/*/ablation.json"))
    total = sum(1 for _ in open("overnight/jobs_all.txt"))
    print(f"OVERNIGHT RUN: {done}/{total} runs complete")

    obst = {k: v for k, v in load("obst_*").items() if k != "obst_noobst"}
    show("OBSTACLES -- does a policy beat the best hand-written rule?",
         obst, BASE["obst"])
    show("NO TREES -- what the trees actually cost the policy",
         load("obst_noobst_*"), BASE["noobst"])
    show("CONSOLIDATION -- sigma vs sigma-blind, finally powered",
         load("cons_*"))

    cons = load("cons_*")
    if "cons_sigma" in cons and "cons_none" in cons:
        s, n = np.array(cons["cons_sigma"]), np.array(cons["cons_none"])
        if len(s) > 1 and len(n) > 1:
            from scipy import stats
            t, p = stats.ttest_ind(s, n, equal_var=False)
            print(f"\n  sigma - none = {s.mean()-n.mean():+.1f} pp, "
                  f"Welch p = {p:.3f} (n={len(s)} vs {len(n)})")
            if p > 0.05:
                print("  -> not significant. Do not claim it.")

    if os.path.exists("overnight/watchdog.log"):
        print("\nWATCHDOG (anything that went wrong overnight):")
        for line in open("overnight/watchdog.log"):
            if any(w in line for w in ("LOW", "RAM", "NOTHING", "restart", "killing")):
                print("  " + line.rstrip())


if __name__ == "__main__":
    main()
