#!/usr/bin/env python3
"""
deck_figs.py -- the graphs for the deck's wind and GeoAnchor parts (27 Sept), in the deck's colours:
  wind_mjx_ladder.png   MJX: landed and crashed vs wind -- rule, flying learner (8 seeds), crash-priced decider (3 seeds)
  wind_gz_ladder.png    Gazebo: landed of 20 per wind -- rule, two flyer seeds, the decider; crashes marked
  geo_matcher.png       GeoAnchor on the Logan sweep: fixes accepted with a plain / block / draped floor; heading error
  geo_readheight.png    the pad's 1.80 m marker decoded, by height
  geo_navtrace.png      one GNSS-free delivery: PX4's position error and height, GPS -> GeoAnchor -> the pad marker
  pad_markers.png       the pad's three markers (make_pad.py's texture) and the heights each can be read from
None of them shows the Esri imagery (numbers only), so they can go anywhere.
    python3 papers/deck/deck_figs.py -> outputs/gazebo_geo/deck/
"""
import collections
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
OUT = f"{R}/outputs/gazebo_geo/deck"
RULE, FLY, DEC, GREY, INK = "#b8620a", "#1f7a4d", "#2b6cb0", "#8a9099", "#14181f"
plt.rcParams.update({"font.size": 19, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6a7179",
                     "axes.labelcolor": INK, "xtick.color": "#3d444d", "ytick.color": "#3d444d", "font.family": "DejaVu Sans"})


def ladders(recipe):
    out = {}
    for f in sorted(glob.glob(f"{R}/conditions/wind/runs/{recipe}_s*_best_ladder.json")):
        d = json.load(open(f))
        d = d if isinstance(d, list) else d.get("rows", [])
        out[f] = {float(r["w10"]): r for r in d}
    return list(out.values())


def wind_mjx():
    rule = json.load(open(f"{R}/conditions/wind/runs/rule_ladder.json"))
    rule = {float(r["w10"]): r for r in (rule if isinstance(rule, list) else rule.get("rows", []))}
    fig, ax = plt.subplots(1, 2, figsize=(16.6, 5.4))
    ws = sorted(rule)
    for k, key in enumerate(("landed", "crashed")):
        a = ax[k]
        a.plot(ws, [100 * rule[w][key] for w in ws], color=RULE, lw=3.2, marker="o", ms=8, label="rule")
        for recipe, col, name in (("fly_hp", FLY, "flying learner"), ("stage2_c100", DEC, "crash-priced decider")):
            L = ladders(recipe)
            m = np.array([[100 * l[w][key] for w in ws] for l in L])
            a.fill_between(ws, m.min(0), m.max(0), color=col, alpha=0.18, lw=0)
            a.plot(ws, np.median(m, 0), color=col, lw=3.2, marker="o", ms=8, label=f"{name} ({len(L)} seeds)")
        a.set_xlabel("mean wind at 10 m (m/s)")
        a.set_ylabel("% landed on the pad" if key == "landed" else "% crashed")
        a.set_xticks(ws); a.grid(axis="y", color="#e4e1d8", lw=1)
        if key == "landed":
            a.set_ylim(-3, 103); a.legend(frameon=False, fontsize=17, loc="lower left")
        else:
            a.set_ylim(-0.5, 18)
    fig.tight_layout()
    fig.savefig(f"{OUT}/wind_mjx_ladder.png", dpi=100)


def wind_gz():
    fl = [json.loads(l) for l in open(f"{R}/outputs/gazebo_wind/ladder_rule/flights.jsonl")]
    by = collections.defaultdict(list)
    for r in fl:
        by[(r["decider"], r["w10"])].append(r["outcome"])
    ws = [9.0, 12.0, 15.0, 18.0]
    who = [("rule", RULE, "rule"), ("fly_hp_s0_best", FLY, "flying learner, seed 0"),
           ("fly_hp_s1_best", "#5aa77f", "flying learner, seed 1"), ("stage2_c100_s0_best", DEC, "crash-priced decider")]
    fig, a = plt.subplots(figsize=(16.6, 5.2))
    wbar = 0.2
    for i, (d, col, name) in enumerate(who):
        x = np.arange(len(ws)) + (i - 1.5) * wbar
        landed = [by[(d, w)].count("landed") for w in ws]
        n = [len(by[(d, w)]) for w in ws]
        crashed = [by[(d, w)].count("crashed") for w in ws]
        a.bar(x, landed, wbar * 0.92, color=col, label=name)
        for xx, c, l_, nn in zip(x, crashed, landed, n):
            if c:
                a.text(xx, l_ + 0.4, f"{c}✖", ha="center", color="#b3261e", fontsize=17, fontweight="bold")
            a.text(xx, -1.6, f"{nn}", ha="center", color=GREY, fontsize=12)
    a.set_xticks(np.arange(len(ws))); a.set_xticklabels([f"{w:g} m/s" for w in ws])
    a.set_ylabel("landed (of 20 yards)"); a.set_ylim(-2.2, 22); a.grid(axis="y", color="#e4e1d8", lw=1)
    a.legend(frameon=False, fontsize=16, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(f"{OUT}/wind_gz_ladder.png", dpi=100)


def geo_matcher():
    runs = {}
    for tag in ("plain", "floor", "draped", "draped_yaw5", "draped_yaw10", "draped_yaw15"):
        f = sorted(glob.glob(f"{R}/outputs/gnss_denied/runs/{tag}_2*/score.json"))
        if f:
            runs[tag] = json.load(open(f[-1]))
    fig, ax = plt.subplots(1, 2, figsize=(16.6, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    names = [("plain", "plain ground,\nhouse blocks"), ("floor", "photo floor,\nflat-coloured blocks"),
             ("draped", "photo floor,\nroofs + trees draped")]
    vals = [runs[k]["accepted_pct"] for k, _ in names]
    bars = a.bar(range(3), vals, 0.6, color=[GREY, "#9fb7cf", DEC])
    for i, (k, _) in enumerate(names):
        med = runs[k].get("median_m")
        a.text(i, vals[i] + 1.2, f"{vals[i]:.0f}%" + (f"\nmedian {med:.1f} m" if med else "\nno fixes"),
               ha="center", fontsize=17, color=INK)
    a.set_xticks(range(3)); a.set_xticklabels([n for _, n in names], fontsize=16)
    a.set_ylabel("% of frames with a fix"); a.set_ylim(0, 52); a.grid(axis="y", color="#e4e1d8", lw=1)
    a = ax[1]
    hd = [0, 5, 10, 15]
    v = [runs[k]["accepted_pct"] for k in ("draped", "draped_yaw5", "draped_yaw10", "draped_yaw15")]
    p99 = [runs[k]["p99_m"] for k in ("draped", "draped_yaw5", "draped_yaw10", "draped_yaw15")]
    a.plot(hd, v, color=DEC, lw=3.2, marker="o", ms=9)
    for x, y, q in zip(hd, v, p99):
        a.text(x, y + 1.6, f"{y:.0f}%", ha="center", fontsize=17)
        a.text(x, 3, f"worst 1%:\n{q:.0f} m", ha="center", fontsize=13, color="#b3261e" if q > 20 else GREY)
    a.set_xticks(hd); a.set_xlabel("heading error given to GeoAnchor (deg)"); a.set_ylabel("% of frames with a fix")
    a.set_ylim(0, 46); a.grid(axis="y", color="#e4e1d8", lw=1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/geo_matcher.png", dpi=100)


def geo_read():
    rows = json.load(open(f"{R}/outputs/gazebo_geo/read_height.json"))
    h = [r["h"] for r in rows]
    p = [100 * np.mean([2 in f["ids"] for f in r["frames"]]) for r in rows]
    fig, a = plt.subplots(figsize=(8.2, 5.2))
    a.plot(h, p, color=FLY, lw=3.2, marker="o", ms=9)
    for x, y in zip(h, p):
        a.text(x, y + 3.5, f"{y:.0f}%", ha="center", fontsize=16)
    a.axvline(30, color=INK, lw=1.2, ls=":")
    a.set_xlabel("height over the pad (m)"); a.set_ylabel("% of frames: 1.80 m marker read")
    a.set_ylim(0, 105); a.invert_xaxis(); a.grid(axis="y", color="#e4e1d8", lw=1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/geo_readheight.png", dpi=100)


def geo_trace(src=f"{R}/outputs/gazebo_geo/geo_obst3/obstacles_geo.json"):
    d = json.load(open(src))
    rows = [r for r in d["rows"] if r["gz"]]
    t0 = rows[0]["t"]
    T = np.array([r["t"] - t0 for r in rows])
    err = np.array([np.hypot(r["est"][0] - r["gz"][0], r["est"][1] - r["gz"][1]) for r in rows])
    z = np.array([r["gz"][2] for r in rows])
    gnss = np.array([bool(r["gnss"]) for r in rows])
    wall = np.array([r["wall"] for r in rows]); ts = np.array([r["t"] for r in rows])
    mk = [np.interp(f["t_wall"], wall, ts) - t0 for f in d["fixes"] if f.get("source") == "marker"]
    t_mark = min(mk) if mk else T[-1]
    t_den = T[np.argmax(~gnss)] if (~gnss).any() else 0.0
    fig, a = plt.subplots(figsize=(16.6, 5.0))
    for lo, hi, col, lab in ((0, t_den, "#e9e6de", "GPS"), (t_den, t_mark, "#dbe7f3", "GeoAnchor (GPS denied)"),
                             (t_mark, T[-1], "#dcefe3", "the pad marker (GPS denied)")):
        a.axvspan(lo, hi, color=col, lw=0)
        a.text((lo + hi) / 2, 16.2, lab, ha="center", fontsize=16, color=INK)
    a.plot(T, err, color=DEC, lw=2.2, label="PX4's position error (m)")
    a.set_ylabel("PX4's error vs truth (m)", color=DEC); a.set_ylim(0, 17.5); a.set_xlabel("seconds")
    b = a.twinx()
    b.plot(T, z, color=INK, lw=1.6, alpha=0.7)
    b.set_ylabel("height (m)", color=INK); b.set_ylim(0, 90); b.spines["top"].set_visible(False)
    a.set_xlim(0, T[-1])
    fig.tight_layout()
    fig.savefig(f"{OUT}/geo_navtrace.png", dpi=100)


def pad_markers():
    """The pad as Gazebo paints it (make_pad.py's own texture, no imagery) and the three markers' height bands."""
    import cv2
    tex = cv2.cvtColor(cv2.imread("/home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/models/arucotag/arucotag.png"),
                       cv2.COLOR_BGR2RGB)
    import sys; sys.path.insert(0, f"{R}/shared/px4_gazebo")
    from make_pad import MARKERS, MAT                       # (id, size m, centre x m) on a MAT m square
    bands = {2: (0.76, 28.6), 0: (0.51, 19.0), 1: (0.06, 2.4)}   # make_pad.py's design table
    col = {2: RULE, 0: FLY, 1: DEC}
    fig, (a, b) = plt.subplots(1, 2, figsize=(16.6, 6.4), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a.imshow(tex, extent=(-MAT / 2, MAT / 2, -MAT / 2, MAT / 2))
    a.add_patch(plt.Rectangle((-MAT / 2, -MAT / 2), MAT, MAT, fill=False, ec=GREY, lw=1.5))
    for mid, size, x in MARKERS:
        a.add_patch(plt.Rectangle((x - size / 2 - 0.06, -size / 2 - 0.06), size + 0.12, size + 0.12, fill=False,
                                  ec=col[mid], lw=3.5))
        if mid == 1:                                         # the small one: labelled below, with a leader
            a.annotate("id 1, 0.15 m\nthe touchdown point", (x, -0.14), (x, -1.55), ha="center", va="top", fontsize=16,
                       color=col[mid], fontweight="bold", arrowprops=dict(arrowstyle="-", color=col[mid], lw=2))
        else:
            a.text(x, size / 2 + 0.22, f"id {mid}, {size:.2f} m", ha="center", va="bottom", color=col[mid], fontsize=16,
                   fontweight="bold")
    a.plot([-2.25, -1.25], [-2.2, -2.2], color=INK, lw=3)
    a.text(-1.75, -2.1, "1 m", ha="center", va="bottom", fontsize=15)
    a.set_xlim(-MAT / 2 - 0.05, MAT / 2 + 0.05); a.set_ylim(-MAT / 2 - 0.05, MAT / 2 + 0.05); a.axis("off")
    a.set_title("the 5 m mat, as Gazebo paints it", fontsize=18, color="#3d444d")
    sizes = {m: sz for m, sz, _ in MARKERS}
    for k, mid in enumerate((2, 0, 1)):
        lo, hi = bands[mid]
        b.barh(k, np.log10(hi) - np.log10(lo), left=np.log10(lo), height=0.55, color=col[mid], alpha=0.9)
        b.text(np.log10(hi) + 0.05, k, f"{hi:g} m", va="center", ha="right", fontsize=17, color=INK)
        b.text(np.log10(lo) - 0.05, k, f"{lo:g} m", va="center", ha="left", fontsize=17, color=INK)
        b.text(np.log10(hi) - 0.02, k - 0.42, f"id {mid}, {sizes[mid]:.2f} m", va="center", ha="left", fontsize=16,
               color=col[mid], fontweight="bold")
    b.axvline(np.log10(30), color=GREY, ls="--", lw=2)
    b.text(np.log10(30) - 0.05, 2.55, "30 m: the drop ends, the\nmarker takes over (88% read)", ha="left", va="center",
           fontsize=15, color="#3d444d")
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    b.set_xticks(np.log10(ticks)); b.set_xticklabels([f"{t:g}" for t in ticks])
    b.set_xlim(1.75, -1.6); b.set_ylim(2.9, -0.8); b.set_yticks([])
    b.set_xlabel("height above the pad (m), going down \u2192")
    b.spines["left"].set_visible(False)
    b.set_title("where each marker can be read", fontsize=18, color="#3d444d")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pad_markers.png", dpi=100)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (wind_mjx, wind_gz, geo_matcher, geo_read, geo_trace, pad_markers):
        f()
        print(f.__name__, "ok")
