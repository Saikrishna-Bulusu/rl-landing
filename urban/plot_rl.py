#!/usr/bin/env python3
"""
Learning curves: greedy clean-delivery rate vs training episodes, per world,
one thin line per seed, mean with a min-max band, and the rule, the best
same-sensor planner and the oracle drawn as lines scored on the SAME episodes.
"""
import glob
import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, world in zip(axes, ("corridor", "random")):
    runs, base = [], None
    for path in sorted(glob.glob(f"rl_{world}_*.json")):
        d = json.load(open(path))
        base = d.get("baselines", base)
        for tag, r in d.items():
            if tag == "baselines":
                continue
            pts = [(row["episodes"], row["eval_clean"]) for row in r["log"] if "eval_clean" in row]
            if pts:
                runs.append(np.array(pts))
            tr = np.array([(row["episodes"], row["clean"]) for row in r["log"]])
            k = 10                                           # smooth the noisy per-batch rate
            sm = np.convolve(tr[:, 1], np.ones(k) / k, "valid")
            ax.plot(tr[k - 1:, 0] / 1e3, sm, color="#2b7bba", ls="--", lw=1.6,
                    label="RL sampled (training batches, 10-gen mean)" if not ax.get_legend_handles_labels()[1].count(
                        "RL sampled (training batches, 10-gen mean)") else None)
    for r in runs:
        ax.plot(r[:, 0] / 1e3, r[:, 1], color="#2b7bba", lw=0.8, alpha=0.35)
    if runs:
        L = min(len(r) for r in runs)
        Y = np.stack([r[:L, 1] for r in runs]); X = runs[0][:L, 0] / 1e3
        ax.plot(X, Y.mean(0), color="#2b7bba", lw=2.4, label=f"RL greedy (argmax), {len(runs)} seed(s)")
        ax.fill_between(X, Y.min(0), Y.max(0), color="#2b7bba", alpha=0.15)
    if base:
        for k, c, ls in (("rule", "#555555", "--"), ("planner", "#e08a00", "-."), ("oracle", "#c0392b", ":")):
            ax.axhline(base[k], color=c, ls=ls, lw=1.6, label=f"{k} {base[k]:.1f}%")
    ax.set_title(f"{world} world"); ax.set_xlabel("training episodes (thousands)"); ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("clean deliveries (%)\n(solid lines: same 1,024 evaluation episodes)")
fig.suptitle("Learning a decision layer for last-mile delivery in Logan, QLD", fontsize=13)
fig.tight_layout(); fig.savefig("rl_curves.png", dpi=130); print("rl_curves.png")

# ---- the ladder: every decider on the SAME 4,096 episodes (the headroom ladder's) ----
head, clone = json.load(open("headroom.json")), json.load(open("clone_check.json"))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for ax, world in zip(axes, ("corridor", "random")):
    ev = json.load(open(f"rl_{world}_0.json"))[f"{world}_s0"]["eval"]
    rows = [("PPO from scratch, greedy", ev["greedy"]["clean"], "#2b7bba"),
            ("PPO from scratch, sampled", ev["sampled"]["clean"], "#2b7bba"),
            ("rule", head[world]["rule"]["clean"], "#555555"),
            ("copy of the planner\n(same network, same inputs)", clone[world]["clean"], "#2e9e5b"),
            ("planner (same sensors)", head[world]["shield_obs"]["clean"], "#e08a00"),
            ("oracle (knows the future)", head[world]["safeplan_full"]["clean"], "#c0392b")]
    for i, (name, v, c) in enumerate(rows):
        ax.hlines(i, 60, v, color=c, lw=1, alpha=0.4); ax.plot(v, i, "o", color=c, ms=10)
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_xlim(60, 101)
    ax.set_title(f"{world} world"); ax.set_xlabel("clean deliveries (%), same 4,096 episodes"); ax.grid(axis="x", alpha=0.25)
fig.suptitle("Same network, same inputs: copying the planner beats the rule; PPO from scratch never found it", fontsize=12)
fig.tight_layout(); fig.savefig("rl_ladder.png", dpi=130); print("rl_ladder.png")

# ---- planner's eyes + practice: validation curves (top) and the ladder's episodes (bottom) ----
eyes = {w: sorted(glob.glob(f"rl_eyes_{w}_*.json")) for w in ("corridor", "random")}
READ = [("mean", "eval_mean", "average command", "#1b6e3c", "-"),      # the three read-outs of one network
        ("sampled", "eval_sampled", "rolling its dice", "#2e9e5b", "--"),
        ("greedy", "eval_clean", "top choice", "#8fcfa6", ":")]
if any(eyes.values()):
    COPY = 3 * 2048                      # flights spent building the copy before any practice
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw=dict(height_ratios=[3, 2.6]), layout="constrained")
    for col, world in enumerate(("corridor", "random")):
        ax, logs, evs, base = axes[0, col], [], [], None
        for path in eyes[world]:
            d = json.load(open(path)); base = d["baselines"]
            logs += [r["log"] for t, r in d.items() if t != "baselines"]
            evs += [r["eval"] for t, r in d.items() if t != "baselines"]
        for k, key, name, c, ls in READ:
            runs = [np.array([(row["episodes"] + COPY, row[key]) for row in lg if key in row]) for lg in logs]
            L = min(len(r) for r in runs); Y = np.stack([r[:L, 1] for r in runs]); X = runs[0][:L, 0] / 1e3
            ax.plot(X, Y.mean(0), color=c, ls=ls, lw=2.2, label=f"{name} (mean of {len(runs)} seed{'s' * (len(runs) > 1)}, band = range)")
            ax.fill_between(X, Y.min(0), Y.max(0), color=c, alpha=0.12)
        for k, c, ls in (("rule", "#555555", "--"), ("planner", "#e08a00", "-."), ("oracle", "#c0392b", ":")):
            ax.axhline(base[k], color=c, ls=ls, lw=1.4)
            ax.text(1.005, base[k], f"{k} {base[k]:.1f}%", color=c, transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.set_ylim(86, 98.5); ax.set_title(f"{world} world: planner's eyes + practice")
        ax.set_ylabel("clean deliveries (%), 2,048 validation flights"); ax.grid(alpha=0.25)
        ax.set_xlabel(f"training flights (thousands; the first {COPY:,} build the copy)")
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

        ax = axes[1, col]
        ev0 = json.load(open(f"rl_{world}_0.json"))[f"{world}_s0"]["eval"]
        rows = [("PPO from scratch (dice)", [ev0["sampled"]["clean"]], "#2b7bba", None),
                ("rule", [head[world]["rule"]["clean"]], "#555555", None),
                ("copy, raw inputs", [clone[world]["clean"]], "#b7dcc4", None),
                ("planner (same sensors)", [head[world]["shield_obs"]["clean"]], "#e08a00", None)]
        rows += [(f"eyes+practice, {name}", [e[k]["clean"] for e in evs], c, [e[k]["vs_planner"] for e in evs])
                 for k, key, name, c, ls in READ[::-1]]
        rows += [("oracle (knows the future)", [head[world]["safeplan_full"]["clean"]], "#c0392b", None)]
        for i, (name, v, c, vp) in enumerate(rows):
            ax.plot(v, [i] * len(v), "o", color=c, ms=6, alpha=0.5 if len(v) > 1 else 1)
            ax.plot(np.mean(v), i, "o", color=c, ms=11)
            note = f"{np.mean(v):.1f}%"
            if vp:
                note += f"  ({np.mean([x[0] for x in vp]):+.2f} vs planner)"
            ax.text(max(v) + 0.2, i, note, va="center", fontsize=9)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_xlim(89, 102)
        ax.set_xlabel("clean deliveries (%), the ladder's 4,096 test flights (paired)"); ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Handing the network the planner's 20-second forecast, then letting it practise", fontsize=13)
    fig.savefig("rl_eyes.png", dpi=130); print("rl_eyes.png")

# ---- the numbers behind rl_eyes.png: gain over the planner across seeds ----
if any(eyes.values()):
    T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}     # Student t, two-sided 95%, by degrees of freedom
    for world in ("corridor", "random"):
        runs = [r for p in eyes[world] for t, r in json.load(open(p)).items() if t != "baselines"]
        print(f"\n{world}: gain over the planner, 4,096 paired test flights, {len(runs)} seeds")
        for k, key, name, c, ls in READ:
            d = np.array([r["eval"][k]["vs_planner"][0] for r in runs])
            half = T975.get(len(d) - 1, 2.0) * d.std(ddof=1) / len(d) ** 0.5 if len(d) > 1 else float("nan")
            print(f"  {name:16s} {d.mean():+.2f} pp  (seeds {', '.join(f'{x:+.2f}' for x in d)}; "
                  f"95% CI across seeds +/-{half:.2f})")
        # the fair headline: each seed's read-out chosen on VALIDATION, scored on the test flights
        pick = [max(READ, key=lambda rd: r["log"][-1][f"kept_{rd[0]}"]["validation"])[0] for r in runs]
        d = np.array([r["eval"][k]["vs_planner"][0] for r, k in zip(runs, pick)])
        print(f"  {'chosen on validation':16s} {d.mean():+.2f} pp  (picked {', '.join(pick)}; "
              f"seeds {', '.join(f'{x:+.2f}' for x in d)})")
