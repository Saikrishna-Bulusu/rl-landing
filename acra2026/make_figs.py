"""The ACRA figures, drawn from saved results into acra2026/figs/ (PDF + PNG), in the style of urban/paper_figs.py.
    python3 acra2026/make_figs.py            all four data figures (the stack diagram is TikZ in the paper)
No render of the Esri photo floor appears anywhere: the GNSS-free figure is a plot of errors."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE)
sys.path.insert(0, f"{R}/urban")
import paper_figs as P                      # rcParams (LaTeX text, ACRA widths), colours, save()
import matplotlib.pyplot as plt
P.OUT = f"{HERE}/figs"
P.LAB["oracle"] = "Privileged planner"      # the paper's name for the planner told every intruder's true future


def magpie():
    """Magpie flights struck, by how the drone knows the bird (x) and by decider (bars): paper_figs.birds() panel (b)."""
    modes = [("headroom_lidar_gps_birds_hunter.json", "Handed\nthe truth"), ("headroom_lidar_gps_birds_hunter_rig.json", "Camera rig"),
             ("headroom_lidar_gps_birds_hunter_rig_half.json", "Rig, half\nthe range"), ("headroom_lidar_gps_birds_cameras_hunter.json", "Two own\ncameras")]
    fig, bx = plt.subplots(figsize=(P.COL, 1.75), layout="constrained")
    w = 0.26
    for k, (f, lab) in enumerate(modes):
        h = json.load(open(f"{R}/urban/{f}"))["corridor"]
        for j, key in enumerate(("rule", "planner", "oracle")):
            d = h[{"rule": "rule", "planner": h["best_obs"], "oracle": h["best_full"]}[key]]["struck_magpie"]
            bx.bar(k + (j - 1) * w, d, w * 0.92, color=P.C[key], label=P.LAB[key] if k == 0 else None)
            bx.text(k + (j - 1) * w, d + 0.6, f"{d:.0f}", ha="center", fontsize=6)
    bx.set_xticks(range(len(modes)), [m[1] for m in modes]); bx.tick_params(axis="x", length=0)
    bx.set(ylabel=r"Magpie flights struck (\%)", ylim=(0, 60))
    bx.grid(axis="y", alpha=0.4); bx.legend(loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.8, fontsize=6.5)
    P.save(fig, "magpie")


def gnssfree(run="geo_calm4"):
    """One GNSS-free flight: PX4's horizontal position error and each sent visual fix's error, both against Gazebo's
    truth, over the time since GNSS was denied (the same quantities as geo_cruise.report)."""
    d = json.load(open(f"{R}/gazebo_geo/{run}/geo_e0_w0_rule.json"))
    rows, fixes, rep = d["rows"], d["fixes"], d["report"]
    ts = np.array([r["t"] for r in rows]); wall = np.array([r["wall"] for r in rows])
    t0 = ts[-1] - rep["denied_s"]
    ok = [r for r in rows if r["gz"] is not None and r["t"] > t0]
    te = np.array([r["t"] for r in ok]) - t0
    ee = np.array([np.hypot(*(np.array(r["est"][:2]) - np.array(r["gz"][:2]))) for r in ok])
    gz = np.array([r["gz"][:2] for r in ok]); tg = np.array([r["t"] for r in ok])
    tf, fe = [], []
    for f in fixes:
        if not f["sent"] or f.get("source") == "marker":
            continue
        t_cap = np.interp(f["t_wall"] - f["latency_ms"] / 1e3, wall, ts)
        if t_cap <= t0:
            continue
        j = int(np.clip(np.searchsorted(tg, t_cap), 0, len(tg) - 1))
        tf.append(t_cap - t0); fe.append(float(np.hypot(f["gz"][0] - gz[j, 0], f["gz"][1] - gz[j, 1])))
    fig, ax = plt.subplots(figsize=(P.COL, 1.65), layout="constrained")
    ax.scatter(tf, fe, s=3, color="#e67e22", lw=0, alpha=0.7, label="Visual fix")
    ax.plot(te, ee, color="#2e86c1", lw=1.0, label="PX4 position")
    land = [r["t"] - t0 for r in ok if r["phase"].startswith("landing")]
    if land:
        ax.axvspan(land[0], te[-1], color="#d5d8dc", alpha=0.5, lw=0)
        ax.text(land[0] + 1, 7.6, "marker", fontsize=6.5, va="top")      # below the "fixes above 10 m" note
    ax.set(xlabel="Time since GNSS was denied (s)", ylabel="Horizontal error (m)", ylim=(0, 10), xlim=(0, te[-1]))
    off = int(np.sum(np.array(fe) > 10))
    if off:
        ax.text(0.99, 0.97, f"{off} fixes above 10 m not shown", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    ax.grid(alpha=0.4); ax.legend(loc="upper left", fontsize=6.5, markerscale=2)
    P.save(fig, "gnssfree")
    return dict(fixes_plotted=len(fe), fixes_over_10m=int(np.sum(np.array(fe) > 10)), est_median=round(float(np.median(ee)), 2))


def gazebo_frames():
    """Figure 5's four Gazebo frames as separate images (Sai, 30 Sept: every panel its own diagram, its label below
    it). figs/src/*_full.png are the frames grabbed from the flight videos on 29 Sept, all plain Gazebo worlds (no
    photo floor); the crops are the ones the earlier strip used."""
    import cv2
    box = {"fog": (40, 455, 25, 455), "crane": (56, 575, 0, 958), "magpie": (658, 1175, 0, 958), "pad": (56, 575, 0, 958)}
    for k, (n, (y0, y1, x0, x1)) in zip("abcd", box.items()):
        im = cv2.imread(f"{HERE}/figs/src/{n}_full.png")[y0:y1, x0:x1]
        im = cv2.resize(im, (int(im.shape[1] * 480 / im.shape[0]), 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{HERE}/figs/gazebo_{k}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"figs/gazebo_{k}.jpg", im.shape[1], "x", im.shape[0])


if __name__ == "__main__":
    os.makedirs(P.OUT, exist_ok=True)
    P.world(); P.learning(); gazebo_frames(); magpie(); print("gnssfree:", gnssfree())   # magpie and gnssfree: no longer in the paper (30 Sept)
