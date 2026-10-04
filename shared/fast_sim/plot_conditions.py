#!/usr/bin/env python3
"""
plot_conditions.py -- success vs training episodes, one learner per condition (the layout of
Benali & Hamoud's Figure 6: baseline vs harder conditions), plus what learning added.

    python plot_conditions.py [--world corridor] [--prefix rl_eyes_lidar_gps] [--out conditions.png]

Left: each line is 3 seeds, the dice read-out scored every 10 rounds on the same 2,048
validation flights (mean line, seed range shaded). Round 0 is the learner right after copying
the planner -- the copy costs 6,144 supervised flights, not counted on the x-axis. Dashed: the
planner it copied, in the same condition. The star marks the round each seed kept (best on
validation; late rounds drift down), labelled with the kept rounds' score on 4,096 untouched
test flights. Right: that score minus the planner's on the same test flights, paired 95% CI.
"""
import argparse
import glob
import json

import numpy as np

COND = [("", "Clear", "#2e86c1"), ("_birds", "Birds (old: straight dive, truth given)", "#8e3b8e"),
        ("_fog", "Fog", "#e67e22"), ("_birds_fog", "Birds + fog (old birds)", "#c0392b"),
        ("_unmapped", "Unmapped structures", "#16a085"),
        ("_birds_fog_unmapped", "Everything (old birds)", "#2c3e50"),
        ("_birds_cameras", "Birds (own cameras only)", "#d4ac0d"),
        ("_birds_hunter_rig", "Hunter magpie, 360-deg camera rig", "#27ae60")]
READ = [("greedy", "top choice"), ("sampled", "dice"), ("mean", "average")]


def load(paths):
    """Every seed saved for one condition, across files (a lost seed re-run lands in its own file)."""
    runs, planner = {}, None
    for path in sorted(paths):
        d = json.load(open(path))
        planner = d["baselines"]["planner"] if planner is None else planner
        runs.update({k: v for k, v in d.items() if isinstance(v, dict) and "log" in v})
    runs = [runs[k] for k in sorted(runs)]
    curves = [[(x["episodes"], x["eval_sampled"]) for x in r["log"] if "eval_sampled" in x] for r in runs]
    kept = [r["log"][-1]["kept_sampled"]["gen"] for r in runs]
    return curves, kept, runs, planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="corridor")
    ap.add_argument("--prefix", default="rl_eyes_lidar_gps")
    ap.add_argument("--out", default="conditions.png")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[1.6, 1], layout="constrained")
    lo, done, placed = 100.0, [], []
    for suf, name, col in COND:
        paths = glob.glob(f"{a.prefix}{suf}_{a.world}_*.json")
        if not paths:
            continue
        curves, kept, runs, planner = load(paths)
        n = min(len(c) for c in curves)
        x = np.array([e for e, _ in curves[0][:n]]) / 1000
        y = np.array([[v for _, v in c[:n]] for c in curves])
        test = np.mean([r["eval"]["sampled"]["clean"] for r in runs])
        ax.fill_between(x, y.min(0), y.max(0), color=col, alpha=0.13, lw=0)
        ax.plot(x, y.mean(0), color=col, lw=2.2, label=f"{name}: {test:.1f}% on test flights")
        ax.axhline(planner, color=col, lw=1, ls="--", alpha=0.75)
        gx = np.mean([2.048 * (g + 1) for g in kept])          # 2,048 episodes a round
        gy = np.interp(gx, x, y.mean(0))
        ax.plot(gx, gy, marker="*", ms=15, color=col, mec="white", zorder=5)
        near = sum(abs(gy - py) < 2.0 and abs(gx - px) < 60 for px, py in placed)     # stagger labels that would collide
        ax.annotate(f"{test:.1f}%", (gx, gy), xytext=(0, 12 + 14 * near), textcoords="offset points",
                    color=col, fontweight="bold", ha="center", fontsize=9)
        placed.append((gx, gy))
        lo = min(lo, y.min(), planner)
        done.append((name, col, runs))
    ax.set_ylim(max(0, 5 * np.floor((lo - 2) / 5)), 100)
    ax.set_xlabel("Training episodes (thousands), after copying the planner")
    ax.set_ylabel("Clean deliveries, 2,048 validation flights (%)")
    ax.set_title(f"Success vs training episodes, one learner per condition\n"
                 f"({a.world} world, 3 seeds each; dashed = the planner it copied; star = round kept)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    w = 0.26
    for k, (name, col, runs) in enumerate(done):
        for j, (mode, label) in enumerate(READ):
            g = np.array([r["eval"][mode]["vs_planner"] for r in runs])       # (seeds, [mean, ci])
            bx.bar(k + (j - 1) * w, g[:, 0].mean(), w * 0.9, color=col, alpha=[0.45, 0.75, 1.0][j],
                   yerr=g[:, 1].mean(), capsize=3, label=label if k == 0 else None)
            bx.scatter(np.full(len(g), k + (j - 1) * w), g[:, 0], s=9, color="black", zorder=3)
    bx.axhline(0, color="black", lw=0.8)
    bx.set_xticks(range(len(done)), [d[0] for d in done], rotation=30, ha="right", fontsize=9)
    bx.set_ylabel("Learner minus planner, 4,096 test flights (points)")
    bx.set_title("What practice added over the planner it copied\n(bars: mean of 3 seeds; dots: seeds; "
                 "whiskers: paired 95% CI)", fontsize=11)
    bx.legend(title="read-out", fontsize=9)
    bx.grid(axis="y", alpha=0.3)
    fig.savefig(a.out, dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()
