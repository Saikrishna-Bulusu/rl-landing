#!/usr/bin/env python3
"""
paper_figs.py -- the ACRA paper's figures in one style, drawn from saved results only (nothing re-simulated).

    python paper_figs.py [world headroom learning birds rig gazebo]   -> ../paper/figs/<name>.pdf (+ .png preview)

Text is set by LaTeX in Computer Modern, like the paper; widths are the ACRA column (3.375 in) and page
(7 in). Each decider keeps one colour throughout, and the learner is always orange. The learner is scored
with the dice read-out (sampling its policy) everywhere, the one its training curves and kept round use.
"Best planner" = the better of the two same-sensor planners (shield, safe-plan) in that condition;
"all-knowing" = the better of the two told every intruder's true present state (urban_headroom.py).
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE)
OUT = f"{R}/paper/figs"
COL, PAGE = 3.375, 7.0
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.4, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.titlepad": 4, "savefig.dpi": 300})   # dpi: rasters only
C = dict(rule="#99a3a4", planner="#2e86c1", learner="#d35400", oracle="#17202a")
LAB = dict(rule="Rule", planner="Best planner, same sensors", learner="Learner", oracle="All-knowing planner")
CONDS = [("", "Clear"), ("_unmapped", "Unmapped structures"), ("_fog", "Fog"), ("_birds_hunter_rig", "Magpie, camera rig")]
CCOL = {"": "#2e86c1", "_unmapped": "#16a085", "_fog": "#e67e22", "_birds_hunter_rig": "#8e44ad"}
READ = "sampled"


def ladder(suf, world="corridor"):
    return json.load(open(f"{HERE}/headroom_lidar_gps{suf}.json"))[world]


def learners(suf, world="corridor"):
    """Every seed of one condition (a re-run seed lands in its own file); the copied planner's validation score."""
    runs, base = {}, None
    for p in sorted(glob.glob(f"{HERE}/rl_eyes_lidar_gps{suf}_{world}_*.json")):
        d = json.load(open(p)); base = base or d["baselines"]
        runs.update({k: v for k, v in d.items() if isinstance(v, dict) and "log" in v})
    return [runs[k] for k in sorted(runs)], base


def pct(x):
    return rf"{x:.1f}\%"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf"); plt.close(fig)
    subprocess.run(["pdftoppm", "-r", "250", "-png", "-singlefile", f"{OUT}/{name}.pdf", f"{OUT}/{name}"], check=True)
    print(f"{OUT}/{name}.pdf")


def headroom():
    """Where decisions have room: rule, best same-sensor planner, learner (3 seeds), all-knowing planner."""
    rows = [(suf, name) for suf, name in CONDS]
    rows.insert(3, ("_birds_hunter", "Magpie, handed the truth"))
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    for i, (suf, name) in enumerate(rows):
        h = ladder(suf)
        v = dict(rule=h["rule"]["clean"], planner=h[h["best_obs"]]["clean"], oracle=h[h["best_full"]]["clean"])
        runs, _ = learners(suf)
        if runs:
            s = [r["eval"][READ]["clean"] for r in runs]
            v["learner"] = np.mean(s)
            ax.plot([min(s), max(s)], [i, i], color=C["learner"], lw=2.2, alpha=0.35, solid_capstyle="round")
        ax.plot([min(v.values()), max(v.values())], [i, i], color="#d5d8dc", lw=0.8, zorder=0)
        for k, m in (("rule", "o"), ("planner", "s"), ("oracle", "^"), ("learner", "D")):
            if k in v:
                ax.plot(v[k], i, m, color=C[k], ms=4.2 if k != "learner" else 4.6, mec="white", mew=0.4,
                        label=LAB[k] if i == 0 else None, zorder=3)
        if "learner" in v:
            ax.annotate(pct(v["learner"]), (v["learner"], i), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=6.5, color=C["learner"])
    ax.set_yticks(range(len(rows)), [n for _, n in rows]); ax.invert_yaxis()
    ax.set_xlabel(r"Clean deliveries (\%), 4,096 test flights")
    ax.set_xlim(66, 100); ax.grid(axis="x"); ax.tick_params(axis="y", length=0)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.2, columnspacing=0.8)
    save(fig, "headroom")


def learning(h=2.1, wa=0.6 * PAGE, wb=0.38 * PAGE):
    """Validation curves per condition (learning_a) and the kept learners' gain over the planner they copied on the
    test flights (learning_b), as two figures: each panel its own diagram, its label below it in the paper (Sai,
    30 Sept). The legend sits above (a), clear of the curves, and names the band (the seeds' range) and the dashes
    (the copied planner). (b)'s bars carry the paper's combined 95% interval: the widest per-seed interval and a t
    interval over the three seed gains, root-sum-square, as Table 2's."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fa, ax = plt.subplots(figsize=(wa, h), layout="constrained")
    fb, bx = plt.subplots(figsize=(wb, h), layout="constrained")
    gains, hs = [], []
    for suf, name in CONDS:
        runs, base = learners(suf)
        curves = [[(x["episodes"], x[f"eval_{READ}"]) for x in r["log"] if f"eval_{READ}" in x] for r in runs]
        n = min(len(c) for c in curves)
        x = np.array([e for e, _ in curves[0][:n]]) / 1000
        y = np.array([[v for _, v in c[:n]] for c in curves])
        ax.fill_between(x, y.min(0), y.max(0), color=CCOL[suf], alpha=0.15, lw=0)
        hs += ax.plot(x, y.mean(0), color=CCOL[suf], lw=1.3)
        ax.axhline(base["planner"], color=CCOL[suf], lw=0.8, ls=(0, (4, 2)), alpha=0.9)
        gains.append((name, CCOL[suf], np.array([r["eval"][READ]["vs_planner"] for r in runs])))
    hs += [Line2D([], [], color="0.35", lw=0.8, ls=(0, (4, 2))), Patch(color="0.35", alpha=0.25, lw=0)]
    ax.legend(hs, [n for _, n in CONDS] + ["copied planner", "range of 3 seeds"], loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=3, handlelength=1.6, columnspacing=1.0, borderaxespad=0, frameon=False)
    ax.set_xlabel("Training flights after copying the planner (thousands)")
    ax.set_ylabel(r"Clean deliveries (\%), validation")
    ax.set_xlim(0, None); ax.grid(alpha=0.4)
    t975 = 4.302652729911275                  # Student's t, 2 degrees of freedom
    for k, (name, col, g) in enumerate(gains):
        ci = (g[:, 1].max() ** 2 + (t975 * g[:, 0].std(ddof=1) / np.sqrt(len(g))) ** 2) ** 0.5
        bx.bar(k, g[:, 0].mean(), 0.62, color=col, alpha=0.85, yerr=ci, capsize=2.5, error_kw=dict(lw=0.7, capthick=0.7))
        bx.scatter(np.full(len(g), k) + np.linspace(-0.12, 0.12, len(g)), g[:, 0], s=5, color="black", zorder=3, lw=0)
        bx.text(k, max(g[:, 0].mean() + ci, g[:, 0].max()) + 0.4, f"{g[:, 0].mean():+.1f}", ha="center", fontsize=6.5)
    eb = bx.errorbar([np.nan], [np.nan], yerr=[1.0], fmt="none", ecolor="black", capsize=2.5, lw=0.7, capthick=0.7)
    bx.legend([Line2D([], [], marker="o", ls="", color="black", ms=2.2), eb], ["one seed", r"95\% interval"],
              loc="upper left", handlelength=1.2, borderaxespad=0.3)
    bx.axhline(0, color="black", lw=0.6)
    bx.set_xticks(range(len(gains)), [n.replace(" ", "\n", 1) if len(n) > 12 else n for n, _, _ in gains])
    bx.set_ylabel("Gain over copied planner (points)")
    bx.grid(axis="y", alpha=0.4); bx.tick_params(axis="x", length=0)
    save(fa, "learning_a"); save(fb, "learning_b")


def birds():
    """(a) Can a drone dodge a magpie that keeps re-aiming (per-dive contact vs how late it stops correcting);
    (b) what each way of knowing the bird costs in strikes, per decider."""
    sw = json.load(open(f"{HERE}/hunter_sweep.json"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(PAGE, 2.1), width_ratios=[1, 1.35], layout="constrained")
    for sense, lab, col in (("omni", "Handed the truth", "#566573"), ("rig", "Camera rig (detector + tracker)", "#8e44ad")):
        r = sorted((x for x in sw if x["sense"] == sense), key=lambda x: x["t_commit"])
        ax.plot([x["t_commit"] for x in r], [x["per_dive"] for x in r], "-o", color=col, ms=3, label=lab)
    ax.axhline(12.0, color="#c0392b", ls=(0, (4, 2)), lw=0.8)        # the hunter's calibration: no dodging at all
    ax.text(1.22, 12.25, "no dodging", ha="right", va="bottom", color="#c0392b", fontsize=6.5)
    ax.axvline(0.3, color="black", ls=":", lw=0.7)
    ax.text(0.32, 1.0, "model", fontsize=6.5, va="bottom")
    ax.set(xlabel="Magpie stops correcting its dive (s before contact)", ylabel=r"Dives that touch the drone (\%)", ylim=(0, 14))
    ax.grid(alpha=0.4); ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8))
    ax.text(0.01, 0.99, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
    modes = [("headroom_lidar_gps_birds_hunter.json", "Handed\nthe truth"), ("headroom_lidar_gps_birds_hunter_rig.json", "Camera rig"),
             ("headroom_lidar_gps_birds_hunter_rig_half.json", "Rig, half\nthe range"), ("headroom_lidar_gps_birds_cameras_hunter.json", "Own two\ncameras")]
    w = 0.26
    for k, (f, lab) in enumerate(modes):
        h = json.load(open(f"{HERE}/{f}"))["corridor"]
        for j, key in enumerate(("rule", "planner", "oracle")):
            d = h[{"rule": "rule", "planner": h["best_obs"], "oracle": h["best_full"]}[key]]["struck_magpie"]
            bx.bar(k + (j - 1) * w, d, w * 0.92, color=C[key], label=LAB[key] if k == 0 else None)
            bx.text(k + (j - 1) * w, d + 0.6, f"{d:.0f}", ha="center", fontsize=6)
    bx.set_xticks(range(len(modes)), [m[1] for m in modes]); bx.tick_params(axis="x", length=0)
    bx.set(ylabel=r"Magpie flights struck (\%)", ylim=(0, 58))
    bx.grid(axis="y", alpha=0.4); bx.legend(loc="upper right", ncol=3, handlelength=1.0, columnspacing=0.8)
    bx.text(0.01, 0.99, "(b)", transform=bx.transAxes, va="top", fontweight="bold")
    save(fig, "birds")


def rig(yolo=True):
    """(a) how big the magpie looks to the rig; (b) how often each detector finds it in one frame, by that size,
    on drone places it never trained on. (Coverage -- some camera sees 100% of directions above the drone,
    96-100% below -- is one sentence in the text, not a panel.)"""
    d = json.load(open(f"{R}/gz_logan/rig/cover2/cover.json"))
    size = lambda row: max((max(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]) + 1
                            for c in row["cams"].values() if c["px"]), default=0)
    fig, (ax, cx) = plt.subplots(1, 2, figsize=(PAGE, 1.9), width_ratios=[1, 1.5], layout="constrained")
    ranges = sorted({r["range"] for r in d["rows"]})
    for pose, col in (("broadside", "#2e86c1"), ("head-on", "#c0392b")):
        med = [np.median([size(r) for r in d["rows"] if r["pose"] == pose and r["range"] == g and size(r)]) for g in ranges]
        ax.plot(ranges, med, "-o", color=col, ms=3, label=pose)
    ax.set(xscale="log", yscale="log", xlabel="Distance (m)", ylabel="Magpie's longest side (px)")
    ax.set_xticks(ranges, [str(r) for r in ranges]); ax.set_yticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.minorticks_off(); ax.grid(alpha=0.4); ax.legend(loc="lower left")
    ax.text(0.97, 0.97, "(a)", transform=ax.transAxes, va="top", ha="right", fontweight="bold")
    dets = [("score.json", "Small CNN (ours)", "#566573")]
    if yolo and os.path.exists(f"{R}/gz_logan/rig/data2/score_yolo.json"):
        dets.append(("score_yolo.json", "YOLO26n, fine-tuned", "#d35400"))
    bins, bw = None, 0.8 / len(dets)
    for j, (f, lab, col) in enumerate(dets):
        sc = json.load(open(f"{R}/gz_logan/rig/data2/{f}"))
        bins = [(k, v) for k, v in sc["found_by_size_px"].items() if v["n"] >= 10]
        fa = sc["by_threshold"][str(sc["operating_threshold"])]["false_alarms_per_image"]
        cx.bar(np.arange(len(bins)) + (j - (len(dets) - 1) / 2) * bw, [100 * v["found"] for _, v in bins], bw * 0.92,
               color=col, label=f"{lab}: {fa:.3f} false alarms per picture")
    cx.set_xticks(range(len(bins)), [k.replace("50-2000", "50+").replace("-", "--") for k, _ in bins])
    cx.tick_params(axis="x", length=0)
    cx.set(xlabel="Magpie's longest side (px)", ylabel=r"Found in one frame (\%)", ylim=(0, 100))
    cx.grid(axis="y", alpha=0.4); cx.legend(loc="lower right", framealpha=0.92, frameon=True, edgecolor="none")
    cx.text(0.02, 0.97, "(b)", transform=cx.transAxes, va="top", fontweight="bold")
    save(fig, "rig")


def gazebo():
    """PX4 + Gazebo: ten landings per sky, landed on the pad vs aborted, median miss."""
    w = json.load(open(f"{R}/weather_report.json"))
    skies = ["calm", "breeze", "gusty", "mist", "fog"]
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    on = [w[s]["on_pad"] for s in skies]; ab = [w[s]["outcomes"].get("abort", 0) for s in skies]
    other = [w[s]["flights"] - a - b for s, a, b in zip(skies, on, ab)]
    ax.bar(range(5), on, 0.6, color="#27ae60", label="landed on the pad")
    ax.bar(range(5), ab, 0.6, bottom=on, color="#f5b041", label="aborted safely")
    if any(other):
        ax.bar(range(5), other, 0.6, bottom=np.add(on, ab), color="#c0392b", label="other")
    for i, s in enumerate(skies):
        ax.text(i, 10.3, f"{100 * w[s]['miss_m_median']:.0f} cm", ha="center", fontsize=6.5)
    ax.set_xticks(range(5), [s.capitalize() for s in skies]); ax.tick_params(axis="x", length=0)
    ax.set(ylabel="Flights (of 10)", ylim=(0, 12.8), yticks=[0, 5, 10])
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.28), handlelength=1.0)
    save(fig, "gazebo")


def world():
    """The Logan map rebuilt from the LiDAR survey: obstacle heights, the 1,324 pads, hub, helipad, corridor."""
    import math
    from matplotlib.lines import Line2D
    m = np.load(f"{HERE}/logan_map_lidar.npz")
    half, k = float(m["half"]), 5          # 1 m cells -> 5 m, tallest obstacle kept: ~800 px, what 300 dpi prints
    H = m["H"].reshape(m["H"].shape[0] // k, k, -1, k).max(axis=(1, 3))
    road = m["road"].reshape(m["road"].shape[0] // k, k, -1, k).any(axis=(1, 3))
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    ext = (-half / 1000, half / 1000, -half / 1000, half / 1000)
    ax.imshow(np.where(road, 1.0, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=5, interpolation="nearest")
    im = ax.imshow(np.where(H > 2.0, H, np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=25,
                   interpolation="nearest")
    ax.scatter(*(m["pads"] / 1000).T, s=0.8, c="#e67e22", lw=0)
    ax.scatter(*(m["hub"] / 1000), s=28, marker="s", c="#c0392b", edgecolors="white", linewidths=0.5, zorder=4)
    ax.scatter(*(m["helipad"] / 1000), s=45, marker="P", c="#e74c3c", edgecolors="white", linewidths=0.5, zorder=4)
    a, hp = math.radians(135.0), m["helipad"] / 1000                   # urban_env.CORRIDOR: arrivals from the north-west
    tail = (hp[0] + 1.3 * math.cos(a), hp[1] + 1.3 * math.sin(a))
    ax.annotate("", xy=hp, xytext=tail, arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.0, ls="--"))
    ax.text(*tail, "air ambulance", color="#c0392b", fontsize=6.5, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    handles = [Line2D([], [], ls="", marker="o", ms=2.5, color="#e67e22", label=f"pads ({len(m['pads']):,})"),
               Line2D([], [], ls="", marker="s", ms=4, color="#c0392b", label="depot"),
               Line2D([], [], ls="", marker="P", ms=5, color="#e74c3c", label="hospital helipad")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, borderpad=0.3, handletextpad=0.2, framealpha=0.9,
              frameon=True, edgecolor="none")
    ax.set(xlabel="East (km)", ylabel="North (km)", aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("Obstacle height (m)"); cb.outline.set_linewidth(0.5)
    save(fig, "world")


def why():
    """What practice fixed: first failure reason, the planner the learner copied vs the learner (3 seeds), on
    the same 4,096 test flights per condition (why_cond.py)."""
    d = json.load(open(f"{HERE}/why_cond.json"))
    groups = [("Crash", ["crash, unmapped", "crash"], "#17202a"), ("Bird strike", ["bird strike"], "#8e44ad"),
              ("People \\& vehicles", ["people 30 m", "vehicle 30 m", "over a person"], "#e67e22"),
              ("Helicopter", ["near mid-air (NMAC)", "helipad duty"], "#c0392b"), ("Other drones", ["drone separation"], "#5dade2"),
              ("Timeout", ["timeout, people at pad", "timeout, helipad duty", "timeout, other"], "#aab7b8"),
              ("Off the pad", ["landed off pad", "120 m ceiling"], "#d5d8dc")]
    rows = []
    for c, name in (("clear", "Clear"), ("unmapped", "Unmapped"), ("fog", "Fog"), ("rig", "Magpie, rig")):
        if c not in d:
            continue
        ls = [v["fail"] for k, v in d[c].items() if k.startswith("learner")]
        rows += [(f"{name}: planner", d[c]["planner"]["fail"]), ("learner", {k: np.mean([l[k] for l in ls]) for k in ls[0]})]
    fig, ax = plt.subplots(figsize=(COL, 0.24 * len(rows) + 0.7))
    for i, (lab, f) in enumerate(rows):
        x = 0.0
        for g, keys, col in groups:
            v = sum(f.get(k, 0.0) for k in keys)
            ax.barh(i, v, left=x, color=col, height=0.72, label=g if i == 0 else None, lw=0)
            x += v
        ax.text(x + 0.4, i, f"{x:.1f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"Flights not clean (\%), by first reason; 4,096 test flights")
    ax.grid(axis="x", alpha=0.4)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.35, 1.0), handlelength=1.0, columnspacing=0.8)
    save(fig, "why")


def tello():
    """(a) One cage flight with the unmapped pole in the way: the rule flies into it, the planner round it;
    (b) the cage ladder's outcomes on 4,096 test flights (+ the learner once trained)."""
    from matplotlib.patches import Circle, Rectangle
    d = json.load(open(f"{R}/tello/cage_fig.json"))
    f = d["flight"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(PAGE, 2.3), width_ratios=[1.5, 1], layout="constrained")
    ax.add_patch(Rectangle((0, 0), 3, 2, fill=False, lw=0.8, ec="black"))
    for cx, cy, hx, hy, h in f["houses"]:
        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="#aab7b8", lw=0))
    for cx, cy, r, h in f["trees"]:
        ax.add_patch(Circle((cx, cy), r, color="#58d68d", lw=0))
    for j, (px, py) in enumerate(f["pads"]):
        ax.add_patch(Rectangle((px - 0.075, py - 0.075), 0.15, 0.15, color="#e67e22" if j == 0 else "#566573", lw=0))
    ax.add_patch(Rectangle((f["hub"][0] - 0.1, f["hub"][1] - 0.1), 0.2, 0.2, fill=False, ec="#2e86c1", lw=0.8))
    ax.plot(*f["pole"][:2], "o", color="#c0392b", ms=4)
    for key, col, ls in (("rule", C["rule"], (0, (3, 2))), ("planner", C["planner"], "-")):
        p = np.array(f[key])
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.1, ls=ls, label=f"{key.capitalize()} ({f['outcome'][key]})")
    rp = np.array(f["rule"])[-1]
    ax.plot(rp[0], rp[1], "x", color="black", ms=5, mew=1.2)
    ax.text(f["hub"][0], f["hub"][1] - 0.17, "hub", ha="center", va="top", fontsize=6.5, color="#2e86c1")
    ax.text(f["pole"][0], f["pole"][1] - 0.08, "pole, not\non the map", ha="center", va="top", fontsize=6.5, color="#c0392b")
    ax.text(f["pads"][0][0], f["pads"][0][1] + 0.1, "customer", ha="center", va="bottom", fontsize=6.5, color="#e67e22")
    ax.set(xlim=(-0.02, 3.02), ylim=(-0.02, 2.02), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="lower left", fontsize=6.5)
    ax.text(0.01, 0.99, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
    lad = d["ladder"]                                   # every decider on the same 4,096 flights (tello/figdata.py)
    order = [k for k in ("rule", "planner", "learner", "all_knowing") if k in lad]
    parts = [("delivered", "Delivered", "#27ae60"), ("off pad", "Off the pad", "#f5b041"), ("crash", "Crash", "#17202a"),
             ("timeout", "Timeout", "#aab7b8"), ("net", "Net", "#c0392b"), ("wrong pad", "Wrong pad", "#8e44ad")]
    for i, k in enumerate(order):
        x = 0.0
        for o, lab, col in parts:
            v = lad[k].get(o, 0.0)
            bx.barh(i, v, left=x, color=col, height=0.66, lw=0,
                    label=lab if i == 0 and max(lad[q].get(o, 0.0) for q in order) > 0.05 else None)
            x += v
        bx.text(1.5, i, pct(lad[k]["delivered"]), va="center", fontsize=6.5, color="white")
    bx.set_yticks(range(len(order)), [LAB.get(k, "All-knowing planner") if k != "planner" else "Planner" for k in order])
    bx.invert_yaxis(); bx.tick_params(axis="y", length=0); bx.set_xlim(0, 100)
    bx.set_xlabel(r"Test flights (\%), 4,096 per decider")
    bx.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.0, columnspacing=0.8)
    bx.text(-0.55, 1.3, "(b)", transform=bx.transAxes, va="top", fontweight="bold")
    save(fig, "tello")


FIGS = dict(world=world, headroom=headroom, learning=learning, why=why, birds=birds, rig=rig, gazebo=gazebo, tello=tello)

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        if name == "why" and not os.path.exists(f"{HERE}/why_cond.json"):
            print("why: why_cond.json not written yet"); continue
        FIGS[name]()
