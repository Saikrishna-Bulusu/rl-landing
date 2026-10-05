#!/usr/bin/env python3
"""Checkpoint-1 figure from headroom.json: where the failures are, per decider."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
r = json.load(open(os.path.join(R, "shared/fast_sim/headroom.json")))
groups = [("helicopter (NMAC, helipad duty)", ["near mid-air (NMAC)", "helipad duty"], "#c0392b"),
          ("other drones", ["drone separation"], "#8e3b8e"),
          ("people / vehicles / overhead", ["people 30 m", "vehicle 30 m", "over a person"], "#e08a00"),
          ("timeout: people stay at pad", ["timeout, people at pad"], "#95a5a6"),
          ("timeout: helipad duty / other", ["timeout, helipad duty", "timeout, other", "landed off pad", "crash"], "#bdc3c7")]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]; w = r["corridor"]
rows = [("rule", w["fail_rule"]), (f"same sensors,\nbetter decisions", w["fail_best_obs"]), ("oracle\n(knows the future)", w["fail_oracle"])]
for i, (lab, fl) in enumerate(rows):
    x = 0
    for g, keys, c in groups:
        v = sum(fl.get(k, 0.0) for k in keys)
        ax.barh(i, v, left=x, color=c, edgecolor="white", label=g if i == 0 else None); x += v
    ax.text(x + 0.08, i, f"{x:.1f}% fail", va="center", fontsize=10)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r_[0] for r_ in rows]); ax.invert_yaxis()
ax.set_xlabel("% of deliveries that are not clean (corridor world, 4,096 paired episodes)")
ax.set_xlim(0, 8.5); ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.25)
ax.set_title("Where the failures are: violations shrink, the timeouts stay")
ax = axes[1]
names = [("rule", "rule"), (r["corridor"]["best_obs"], "same sensors"), (r["corridor"]["best_full"], "oracle")]
for j, world in enumerate(("corridor", "random")):
    vals = [r[world][k]["violation"] for k, _ in names]
    xs = [i + (j - 0.5) * 0.36 for i in range(3)]
    ax.bar(xs, vals, width=0.34, color=["#2b7bba", "#7fb3d5"][j], label=f"{world} world")
    for x_, v in zip(xs, vals): ax.text(x_, v + 0.06, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels([lab for _, lab in names])
ax.set_ylabel("% of deliveries with any CASA/ASTM\nviolation, near mid-air, or crash"); ax.legend(fontsize=9)
ax.set_title("Regulatory violations: ~10x fewer with foresight"); ax.grid(axis="y", alpha=0.25)
fig.suptitle("Checkpoint 1, Logan QLD: the room above the rule is violations, and 2/3 of it needs no new sensor", fontsize=13)
fig.tight_layout(); fig.savefig(out := os.path.join(R, "outputs/headroom.png"), dpi=130); print(out)
