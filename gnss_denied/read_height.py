#!/usr/bin/env python3
"""
read_height.py -- how high can the landing camera read the pad? (Sai, 26 Sept: "descend to the maximum height at
which it starts to see the landing pads ... I think 50 m" -- measured, not assumed.) Hover over our pad and step down,
decoding every gimbal frame with the landing's own detector (px4_perception's, DICT_4X4_50), per marker.

    OUT=geo_readh WORLD=geo_floor INSTANCE=5 ./wind/sitl.sh /usr/bin/python3 -u geo/read_height.py

Prints, per height: frames seen, % with any of our markers decoded, % per marker (2 = 1.80 m, 0 = 1.20 m), and the
1.80 m marker's side in pixels when read. Writes gazebo_geo/read_height.json.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path[:0] = [R, HERE, f"{R}/wind"]
import px4_mission as PM
from px4_perception import ArucoEye
from px4_steps import TruthLog

PAD = (120.0, 0.0)                                    # our pad, Gazebo ENU (aruco.sdf <pose>)
HEIGHTS = [60, 55, 50, 45, 40, 35, 30, 25]
SETTLE, SAMPLE = 4.0, 12.0                            # sim seconds at each height


def main():
    truth = TruthLog()
    eye = ArucoEye(expect_ids=[2, 0, 1], topic=PM.GIMBAL_TOPIC, hfov_deg=math.degrees(2.0))
    mi = PM.Mission(endpoint=f"udpin:0.0.0.0:{os.environ.get('MAV_PORT', 14540)}")
    assert mi.arm_offboard(), "could not arm"
    mi.yaw_hold = 0.0                                                  # nose north throughout

    def hold(z, secs, sample=None):
        t0 = None
        while True:
            p = mi.pos()
            if p is None:
                continue
            t0 = mi.t_boot if t0 is None else t0
            me = np.array([p[1], p[0], p[2]])                            # Gazebo ENU (GPS on)
            d = np.array(PAD) - me[:2]
            v = d / max(np.linalg.norm(d), 1e-6) * min(7.0, 0.8 * np.linalg.norm(d))
            mi.send(v[1], v[0], float(np.clip(0.8 * (z - me[2]), -2.0, 3.0))); mi.gimbal_down()
            if secs is None and np.linalg.norm(d) < 1.0 and abs(z - me[2]) < 0.5:
                return
            if secs is not None and mi.t_boot - t0 > secs:
                return
            if sample is not None:
                sample()
            time.sleep(0.02)

    rows, last = [], [None]

    def sample():
        fr = getattr(eye, "_last_frame", None)
        if fr is None or fr[1] == last[0]:
            return
        last[0] = fr[1]
        corners, ids = eye._detect(fr[0])
        found = {} if ids is None else {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.flatten())}
        with truth.lock:
            z = truth.rows[-1][3] if truth.rows else float("nan")
        side = float(np.mean(np.linalg.norm(found[2] - np.roll(found[2], 1, axis=0), axis=1))) if 2 in found else None
        rows[-1]["frames"].append(dict(z=round(float(z), 2), ids=sorted(found), side2=side))

    for h in HEIGHTS:
        hold(h, None)
        hold(h, SETTLE)
        rows.append(dict(h=h, frames=[]))
        hold(h, SAMPLE, sample)
        fs = rows[-1]["frames"]
        n = max(len(fs), 1)
        any_ = sum(bool(set(f["ids"]) & {0, 1, 2}) for f in fs) / n
        s2 = [f["side2"] for f in fs if f["side2"]]
        print(f"{h:3d} m (true {np.median([f['z'] for f in fs]) if fs else float('nan'):5.1f}): {len(fs):3d} frames  "
              f"any {100 * any_:5.1f}%  1.80 m {100 * sum(2 in f['ids'] for f in fs) / n:5.1f}%  "
              f"1.20 m {100 * sum(0 in f['ids'] for f in fs) / n:5.1f}%  side {np.median(s2) if s2 else 0:4.1f} px",
              flush=True)
    json.dump(rows, open(f"{R}/gazebo_geo/read_height.json", "w"))


if __name__ == "__main__":
    main()
