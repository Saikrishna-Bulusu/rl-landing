#!/usr/bin/env python3
"""
plot_sensing.py -- what knowing the magpie is worth: the headroom ladders against the hunter magpie
(from behind, steering) under each kind of bird sensing, rule vs planner vs all-knowing planner.

    python plot_sensing.py [--world corridor] [--out sensing.png]

Bars: magpie flights only. "All-knowing" = exact present of everything (the bird included, at any
range), extrapolated -- for a bird that re-aims, that is perfect SENSING, not a known future.
"""
import argparse
import json
import os

MODES = [("", "Handed the truth\n(the old assumption)", "#7f8c8d"),
         ("_rig", "360-degree rig\n(detector + tracker)", "#16a085"),
         ("_rig_half", "Rig, half the\ndetection range", "#76b7a6"),
         ("_cameras", "Own two cameras\n(ahead + below)", "#c0392b")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="corridor")
    ap.add_argument("--out", default="sensing.png")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = []
    for suf, label, col in MODES:
        f = ("headroom_lidar_gps_birds_cameras_hunter.json" if suf == "_cameras" else
             f"headroom_lidar_gps_birds_hunter{'_rig' if suf.startswith('_rig') else ''}{'_half' if suf == '_rig_half' else ''}.json")
        if os.path.exists(f):
            w = json.load(open(f))[a.world]
            rows.append((label, col, w["rule"], w[w["best_obs"]], w[w["best_full"]]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    who = ["Rule", "Planner", "All-knowing planner"]
    for ax, key, title in ((axes[0], "struck_magpie", "Struck by the magpie (%)"),
                           (axes[1], "clean_magpie", "Clean deliveries (%)")):
        x = np.arange(len(rows))
        for j, alpha in enumerate((0.45, 0.75, 1.0)):
            v = [r[2 + j][key] for r in rows]
            b = ax.bar(x + (j - 1) * 0.27, v, 0.25, color=[r[1] for r in rows], alpha=alpha,
                       label=who[j], edgecolor="black", linewidth=0.4)
            for xi, vi in zip(x + (j - 1) * 0.27, v):
                ax.text(xi, vi + 0.4, f"{vi:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x, [r[0] for r in rows], fontsize=9)
        ax.set_title(f"{title}, flights with a magpie", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(title="decider (bar shade)", fontsize=9)
    fig.suptitle(f"The hunter magpie (attacks from behind, keeps steering), {a.world} world, 4,096 flights, GPS: "
                 f"what each way of knowing the bird is worth", fontsize=11)
    fig.savefig(a.out, dpi=150)
    for r in rows:
        print(f"{r[0].replace(chr(10), ' '):45s} struck: rule {r[2]['struck_magpie']:5.1f}  planner {r[3]['struck_magpie']:5.1f}  "
              f"all-knowing {r[4]['struck_magpie']:5.1f}   clean: {r[2]['clean_magpie']:5.1f} {r[3]['clean_magpie']:5.1f} {r[4]['clean_magpie']:5.1f}")
    print(a.out)


if __name__ == "__main__":
    main()
