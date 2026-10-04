#!/usr/bin/env python3
"""
Slide-sized versions of the checkpoint-2 figures, in the deck's palette, with
type large enough to read at 1920x1080 (every label >= 24 px on the slide).
  slide_scratch.png  PPO from scratch, corridor: the top choice swings, the dice stay under the rule
  slide_curves.png   planner's eyes + practice: three read-outs vs planner/oracle, both worlds
"""
import glob
import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, ORANGE, GREEN, PALE = "#14181f", "#6a7179", "#b8620a", "#1f7a4d", "#8fbfa3"
plt.rcParams.update({"font.size": 18, "axes.edgecolor": "#c9c4b8", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})


def lines(ax, base, keys=(("rule", "#b0aaa0", "--"), ("planner", ORANGE, "-."), ("oracle", MUTED, ":"))):
    for k, c, ls in keys:
        ax.axhline(base[k], color=c, ls=ls, lw=2.2)
        ax.text(1.01, base[k], f"{k} {base[k]:.1f}%", color=c, transform=ax.get_yaxis_transform(),
                va="center", fontsize=18)


# ---- from scratch: corridor ----
d = json.load(open("rl_corridor_0.json")); r = d["corridor_s0"]; base = d["baselines"]
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200, layout="constrained")
ev = np.array([(row["episodes"], row["eval_clean"]) for row in r["log"] if "eval_clean" in row])
tr = np.array([(row["episodes"], row["clean"]) for row in r["log"]])
sm = np.convolve(tr[:, 1], np.ones(10) / 10, "valid")
ax.plot(tr[9:, 0] / 1e3, sm, color=GREEN, ls="--", lw=2.6, label="rolling its dice")
ax.plot(ev[:, 0] / 1e3, ev[:, 1], color=GREEN, lw=2.6, marker="o", ms=5, label="top choice")
for k, c, ls in (("planner", ORANGE, "-."), ("rule", "#b0aaa0", "--")):     # too close to label at the edge
    ax.axhline(base[k], color=c, ls=ls, lw=2.2, label=f"{k} {base[k]:.1f}%")
ax.set_ylim(0, 100); ax.set_xlabel("training flights (thousands)"); ax.set_ylabel("clean deliveries (%)")
ax.legend(loc="lower right", frameon=False); ax.grid(alpha=0.2)
fig.savefig("slide_scratch.png", facecolor="#ffffff"); plt.close(fig)

# ---- planner's eyes + practice: both worlds ----
READ = [("eval_mean", "average command", GREEN, "-", 3.0),
        ("eval_sampled", "rolling its dice", GREEN, "--", 2.2),
        ("eval_clean", "top choice", PALE, ":", 2.6)]
fig, axes = plt.subplots(1, 2, figsize=(16.64, 4.3), dpi=200, layout="constrained", sharey=True)
for ax, world in zip(axes, ("corridor", "random")):
    logs, base = [], None
    for p in sorted(glob.glob(f"rl_eyes_{world}_*.json")):
        dd = json.load(open(p)); base = dd["baselines"]
        logs += [v["log"] for k, v in dd.items() if k != "baselines"]
    for key, name, c, ls, lw in READ:
        runs = [np.array([(row["episodes"] + 6144, row[key]) for row in lg if key in row]) for lg in logs]
        L = min(len(x) for x in runs); Y = np.stack([x[:L, 1] for x in runs]); X = runs[0][:L, 0] / 1e3
        ax.plot(X, Y.mean(0), color=c, ls=ls, lw=lw, label=name)
    lines(ax, base, (("planner", ORANGE, "-."), ("oracle", MUTED, ":")))
    ax.set_title(f"{world} world", color=INK, fontsize=20, loc="left")
    ax.set_xlabel("training flights (thousands)"); ax.grid(alpha=0.2); ax.set_ylim(86, 98.5)
axes[0].set_ylabel(f"clean %, validation\n(mean of {len(logs)} seeds)")
axes[0].legend(loc="lower left", frameon=False, fontsize=17)
fig.savefig("slide_curves.png", facecolor="#ffffff"); plt.close(fig)
print("slide_scratch.png slide_curves.png")
