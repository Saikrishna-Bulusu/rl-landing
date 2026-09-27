#!/usr/bin/env python3
"""
figs.py -- the wind study's figures (gazebo_wind/figs/):
  mjx_vs_gazebo.png   PX4 in Gazebo against the MJX drone: velocity steps (truth, sim time) and hover tilt in wind
  ladder.png          landed / failure kinds against the 10 m wind: the rule and the learners, MJX (paired seeds)
    python wind/figs.py [--ladder runs/both_s0.npz ...]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
OUT = f"{R}/gazebo_wind/figs"
sys.path.insert(0, HERE)


def com(tr, h=0.224):
    """Gazebo truth rows (t, x, y, z, qw, qx, qy, qz) -> the CoM: the x500 model's origin is at its feet, and they
    swing back as it pitches to accelerate (a false 0.5 m/s dip in the origin's velocity)."""
    w, x, y, z = tr[:, 4:8].T
    up = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return tr[:, 1:4] + h * up


def mjx_vs_gazebo():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    import x500_mjx as X
    from px4_steps import STEPS
    g = np.load(f"{R}/gazebo_wind/steps/steps.npz")
    tr, c = g["truth"], g["cmds"]
    tg = tr[:, 0]
    vg = np.gradient(com(tr), tg, axis=0)                   # the CoM, as MJX: the model origin sits at the feet
    chg = np.where(np.any(np.abs(np.diff(c[:, 1:4], axis=0)) > 0.01, axis=1))[0] + 1
    t_chg = list(c[chg, 0])[1:]
    step = jax.jit(X.control_step)
    s = X.init_state((0.0, 0.0, 30.0))
    ts, vs, t = [], [], 0.0
    for secs, vn, ve, vu in STEPS:
        for _ in range(int(round(secs / (X.DT * X.N_SUB)))):
            s = step(s, jnp.array([ve, vn, vu], jnp.float32), jnp.zeros(3), 0.0, jnp.zeros(3))
            t += X.DT * X.N_SUB
            ts.append(t); vs.append(np.array(s["d"].qvel[:3]))
    ts, vs = np.array(ts), np.array(vs)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    show = [(1, 1, "north 0 -> 3 m/s"), (9, 1, "north 0 -> 8 m/s"), (5, 2, "down 0 -> 1.5 m/s")]
    tm, starts, prev = 0.0, {}, (0.0, 0.0, 0.0)
    for k, (secs, vn, ve, vu) in enumerate(STEPS):
        tgt = (ve, vn, vu)
        if tgt != prev:
            starts[k] = (t_chg.pop(0), tm)
        prev = tgt
        tm += secs
    for a_, (k, axis, title) in zip(ax[:2], show[:2]):
        t0g, t0m = starts[k]
        sg, sm = (tg >= t0g - 0.5) & (tg < t0g + 4), (ts >= t0m - 0.5) & (ts < t0m + 4)
        a_.plot(tg[sg] - t0g, vg[sg, axis], color="#b8620a", lw=2.2, label="PX4 in Gazebo (truth)")
        a_.plot(ts[sm] - t0m, vs[sm, axis], color="#1f7a4d", lw=2.2, ls="--", label="MJX drone + PX4 emulation")
        a_.set_title(title); a_.set_xlabel("seconds after the command"); a_.set_ylabel("velocity (m/s)")
    ax[0].legend(frameon=False, fontsize=10, loc="lower right")
    # tilt against wind at the drone: Gazebo hovers with WindDrag, MJX's steady-wind curve
    speeds = np.linspace(0, 20, 21)
    for cda, col, lab in ((0.0, "#9aa0a6", "propeller drag only"), (X.CDA_H, "#1f7a4d", "+ body drag (CdA 0.04 m$^2$)")):
        tl = X.tilt_in_wind(speeds=tuple(speeds), cda=cda)
        ax[2].plot([q[0] for q in tl], [q[1] for q in tl], color=col, lw=2, label=f"MJX, {lab}")
    for name, col in (("w7_nodrag", "#9aa0a6"), ("w7_steady", "#1f7a4d"), ("w12_steady", "#1f7a4d")):
        d = np.load(f"{R}/gazebo_wind/hover_{name}/hover.npz")
        t_ = d["truth"][:, 0]
        sel = t_ > t_[-1] - 10
        x, y = d["truth"][sel, 5], d["truth"][sel, 6]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))).mean()
        w = d["wind"]
        ws = np.linalg.norm(w[w[:, 0] > t_[-1] - 10, 1:4], axis=1).mean()
        ax[2].plot(ws, tilt, "o", ms=9, mfc="white", mec=col, mew=2.5)
    ax[2].plot([], [], "o", ms=9, mfc="white", mec="#555555", mew=2.5, label="PX4 in Gazebo + WindDrag")
    ax[2].set_title("hovering in steady wind"); ax[2].set_xlabel("wind at the drone (m/s)"); ax[2].set_ylabel("tilt (deg)")
    ax[2].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/mjx_vs_gazebo.png", dpi=110)
    print(f"{OUT}/mjx_vs_gazebo.png")


def ladder(ckpts, gz_dirs=()):
    """Landed / how it failed / time against the 10 m wind: the rule and each learner on the same 1024 MJX flights per
    wind (ppo.py --ladder), and Gazebo flights (gz_ladder.sh) where flown, with Wilson 95% intervals."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"rule": "#b8620a", "fly": "#1f7a4d", "fly_hp": "#1f7a4d", "stage2": "#7b4fa0", "stage2_c100": "#2b6cb0",
            "both": "#9aa0a6", "both_sh2": "#c0392b"}
    names = {"rule": "today's rule", "fly": "learner flies, rule decides", "fly_hp": "learner flies, rule decides",
             "stage2": "+ learner decides (crash -15)", "stage2_c100": "+ learner decides (crash -100)",
             "both": "flies + decides, trained together", "both_sh2": "... with the safety layer"}
    runs = [("rule", json.load(open(f"{HERE}/runs/rule_ladder.json")))]
    for c in ckpts:
        runs.append((os.path.basename(c).split("_s")[0], json.load(open(c[:-4] + "_ladder.json"))))
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for tag, res in runs:
        w = [r["w10"] for r in res]
        ax[0].plot(w, [100 * r["landed"] for r in res], "-o", color=cols.get(tag, "#555"), lw=2.2, ms=5, label=names.get(tag, tag))
        ax[2].plot(w, [r["t_mean"] for r in res], "-o", color=cols.get(tag, "#555"), lw=2.2, ms=5)
    # Gazebo: landed share per wind, per decider
    for d in gz_dirs:
        rows = [json.loads(l) for l in open(f"{R}/gazebo_wind/{d}/flights.jsonl")]
        for dec in sorted({r["decider"] for r in rows}):
            tag = "rule" if dec == "rule" else dec.split("_s")[0]
            for w in sorted({r["w10"] for r in rows if r["decider"] == dec}):
                sel = [r for r in rows if r["decider"] == dec and r["w10"] == w]
                n, k = len(sel), sum(r["outcome"] == "landed" for r in sel)
                p, z = k / n, 1.96
                lo = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
                hi = (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
                ax[0].errorbar(w + (0.25 if tag != "rule" else -0.25), 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="s",
                               ms=8, mfc="white", mec=cols.get(tag, "#555"), ecolor=cols.get(tag, "#555"), mew=2, capsize=3)
    if gz_dirs:
        ax[0].plot([], [], "s", ms=8, mfc="white", mec="#555", mew=2, label="Gazebo + PX4 (20 flights, 95% CI)")
    ax[0].set_xlabel("wind at 10 m (m/s)"); ax[0].set_ylabel("landed on the pad (%)"); ax[0].set_ylim(-3, 103)
    ax[0].legend(frameon=False, fontsize=10, loc="lower left"); ax[0].set_title("deliveries landed")
    # failure kinds at the strong winds: rule vs the first learner
    kinds = [("hard", "hard / off the pad", "#d4a373"), ("timeout", "gave up (90 s)", "#9aa0a6"), ("crashed", "crashed", "#c0392b")]
    ws = [15.0, 18.0, 20.0]
    x = np.arange(len(ws))
    pick = [runs[0]] + [r for r in runs[1:] if r[0].startswith("fly")][:1] + [r for r in runs[1:] if r[0].startswith("stage2")][-1:]
    width = 0.8 / len(pick)
    for j, (tag, res) in enumerate(pick):
        bottom = np.zeros(len(ws))
        for k, lab, col in kinds:
            v = np.array([100 * next(r[k] for r in res if r["w10"] == w) for w in ws])
            ax[1].bar(x + (j - (len(pick) - 1) / 2) * width, v, width * 0.92, bottom=bottom, color=col, edgecolor="white", label=lab if j == 0 else None)
            bottom += v
        for xi in x:
            ax[1].text(xi + (j - (len(pick) - 1) / 2) * width, 101, {"rule": "rule"}.get(tag, "flies" if tag.startswith("fly") else "decides"),
                       ha="center", fontsize=9, color="#555")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{w:.0f} m/s" for w in ws]); ax[1].set_ylabel("flights (%)")
    ax[1].set_ylim(0, 110); ax[1].legend(frameon=False, fontsize=10, loc="upper left"); ax[1].set_title("how the rest failed")
    ax[2].set_xlabel("wind at 10 m (m/s)"); ax[2].set_ylabel("mean flight time (s)"); ax[2].set_title("time to land (90 s = gave up)")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/ladder.png", dpi=110)
    print(f"{OUT}/ladder.png")


def pairs(gz_dir, ladder_json, decider="rule"):
    """The same flights in both simulators (episode i = yard i, same start, wind speed and direction; gusts differ):
    how often MJX and Gazebo agree on the outcome, per wind."""
    names = {1: "landed", 2: "crashed", 3: "hard", 4: "timeout", 0: "timeout"}
    mj = {r["w10"]: r["first40"] for r in json.load(open(ladder_json))}
    rows = [json.loads(l) for l in open(f"{R}/gazebo_wind/{gz_dir}/flights.jsonl") if json.loads(l)["decider"] == decider]
    print(f"{'wind':>5s} {'n':>3s}  landed MJX  landed Gazebo  same outcome  miss (landed, median) MJX / Gazebo")
    for w in sorted({r["w10"] for r in rows}):
        sel = sorted([r for r in rows if r["w10"] == w], key=lambda r: r["episode"])
        m = [names[mj[w]["outcome"][r["episode"]]] for r in sel]
        g = [r["outcome"] for r in sel]
        mm = [mj[w]["miss"][r["episode"]] for r, o in zip(sel, m) if o == "landed"]
        gm = [r["miss"] for r in sel if r["outcome"] == "landed"]
        print(f"{w:5.0f} {len(sel):3d}  {m.count('landed'):4d}/{len(sel):<3d}    {g.count('landed'):4d}/{len(sel):<3d}      "
              f"{sum(a == b for a, b in zip(m, g)):4d}/{len(sel):<3d}      {np.median(mm) if mm else float('nan'):.2f} / "
              f"{np.median(gm) if gm else float('nan'):.2f} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", action="store_true")
    ap.add_argument("--ladder", nargs="*", help="checkpoints (runs/*.npz) whose _ladder.json to plot beside the rule")
    ap.add_argument("--gz", nargs="*", default=[], help="gazebo_wind/<dir> with flights.jsonl")
    ap.add_argument("--pairs", nargs=2, metavar=("GZ_DIR", "LADDER_JSON"), help="MJX vs Gazebo outcome agreement, flight by flight")
    a = ap.parse_args()
    if a.valid:
        mjx_vs_gazebo()
    if a.ladder is not None:
        ladder(a.ladder, a.gz)
    if a.pairs:
        pairs(*a.pairs)
