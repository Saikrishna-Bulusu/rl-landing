#!/usr/bin/env python3
"""
measure_detect.py -- detection rate against apparent marker size, measured.

WHY THIS ONE CONSTANT DESERVES A FLIGHT

compare_physics.py isolated the single assumption that decides whether the
trained policy delivers or aborts, and it is not the vehicle dynamics -- it is
MIN_PX, the apparent size at which an ArUco marker decodes reliably. Moving it
from 20 px to 34 px took delivery from 98% to 0%. Everything else together,
including the velocity lag, moved it by one percentage point.

A constant with that much leverage should not be my estimate. So: hold station
over the pad, descend slowly, and record for every frame how many pixels each
marker spans and whether it actually decoded. The result is a curve, not a
threshold, and the env can use the altitude where the rate crosses a stated
level rather than a number someone eyeballed.
"""

import argparse
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from px4_bridge import PX4Link
from px4_perception import ArucoEye

PAD = np.array([0.0, 120.0])
F_PX = 539.94
TAG = {2: 1.80, 0: 1.20, 1: 0.15}


def main():
    ap = argparse.ArgumentParser()
    # 26 m was the old default and it is WHY the curve had no data below
# 25 px: the descent simply never started high enough. Everything the
# env extrapolated below that floor was invention.
    ap.add_argument("--top", type=float, default=60.0)
    ap.add_argument("--bottom", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=0.6, help="descent m/s")
    ap.add_argument("--out", default=f"{HERE}/detect_curve.json")
    a = ap.parse_args()

    link = PX4Link()
    eye = ArucoEye(expect_ids=[2, 0, 1])
    assert link.start_offboard(), "could not arm"

    # climb and station-keep over the pad
    t0 = time.time()
    while time.time() - t0 < 70:
        s = link.state(timeout=0.5)
        if s is None:
            continue
        d = PAD - np.array([s["x"], s["y"]])
        v = np.clip(0.6 * d, -4, 4)
        link.send_velocity(v[0], v[1], 0.8 * (a.top - s["alt"]))
        if np.linalg.norm(d) < 1.0 and abs(s["alt"] - a.top) < 0.8:
            break
        time.sleep(0.05)

    print(f"  descending {a.top} -> {a.bottom} m at {a.rate} m/s")
    rows = []
    t0 = time.time()
    while time.time() - t0 < 200:
        s = link.state(timeout=0.5)
        if s is None:
            continue
        d = PAD - np.array([s["x"], s["y"]])
        v = np.clip(0.6 * d, -1.5, 1.5)
        link.send_velocity(v[0], v[1], -a.rate)
        alt = s["alt"]
        if alt <= a.bottom:
            break
        seen, off, ids, mid = eye.look()
        rows.append((alt, sorted(int(i) for i in ids), float(np.linalg.norm(d))))
        time.sleep(0.05)
    link.send_velocity(0, 0, 1.5)
    time.sleep(4)

    # detection rate per marker, bucketed by APPARENT SIZE not altitude
    out = {}
    for mid, size in TAG.items():
        pts = [(size * F_PX / r[0], mid in r[1]) for r in rows
               if r[0] > 0.2 and r[2] < 2.0]
        if not pts:
            continue
        px = np.array([p[0] for p in pts]); hit = np.array([p[1] for p in pts])
        edges = np.array([10, 15, 20, 25, 30, 35, 45, 60, 90, 150, 400, 2000])
        curve = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (px >= lo) & (px < hi)
            if m.sum() >= 4:
                curve.append(dict(px_lo=int(lo), px_hi=int(hi), n=int(m.sum()),
                                  rate=float(hit[m].mean())))
        out[str(mid)] = curve
        print(f"\n  marker id {mid} ({size} m)   {len(pts)} frames")
        print(f"    {'span px':>12} {'frames':>7} {'decoded':>9}")
        for c in curve:
            bar = "#" * int(round(c["rate"] * 20))
            print(f"    {c['px_lo']:5d}-{c['px_hi']:<6d} {c['n']:7d} "
                  f"{100*c['rate']:7.0f}%  {bar}")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n  saved {a.out}")


if __name__ == "__main__":
    main()
