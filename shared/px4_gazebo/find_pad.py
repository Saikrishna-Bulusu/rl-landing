#!/usr/bin/env python3
"""
find_pad.py -- measure where the landing tag is in PX4's LOCAL_POSITION_NED.

WHY THIS IS NOT JUST A COORDINATE SWAP

Gazebo's world here is ENU and PX4 reports NED, so the obvious conversion is
north = ENU_y, east = ENU_x. That is wrong, and expensively so: it put the
aircraft in a perfect hover at the place the tag was supposed to be, seeing
nothing at all. PX4's local frame is not aligned to the Gazebo world -- the
EKF takes its heading from the simulated magnetometer, and with this world's
magnetic field the aircraft reports yaw 140 deg while physically pointing along
Gazebo +X. The two frames are rotated by that difference.

So the transform is measured rather than derived: fly to two known PX4-NED
positions, read the Gazebo pose at each, and solve the rotation and offset that
map one to the other. Two points fix a 2-D rigid transform exactly.
"""

import math
import re
import subprocess
import time

import numpy as np

from px4_bridge import PX4Link

ALT = 15.0


def gz_pose(model):
    """Gazebo world x, y of a model. Shelling out beats another subscription."""
    out = subprocess.run(["gz", "model", "-m", model, "-p"],
                         capture_output=True, text=True, timeout=20).stdout
    m = re.search(r"\[\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\]", out)
    if not m:
        raise RuntimeError(f"no pose for {model}:\n{out}")
    return np.array([float(m.group(1)), float(m.group(2))])


def goto(link, target_ne, alt, tmax=35):
    t0 = time.time()
    while time.time() - t0 < tmax:
        s = link.state(timeout=0.5)
        if s is None:
            continue
        d = np.array(target_ne) - np.array([s["x"], s["y"]])
        v = np.clip(0.8 * d, -3, 3)
        link.send_velocity(v[0], v[1], 0.6 * (alt - s["alt"]))
        if np.linalg.norm(d) < 0.4 and math.hypot(s["vx"], s["vy"]) < 0.2 \
                and abs(s["alt"] - alt) < 0.5:
            return np.array([s["x"], s["y"]])
        time.sleep(0.05)
    return None


def main():
    link = PX4Link()
    assert link.start_offboard(), "could not arm"
    pairs = []
    for tgt in [(0.0, 0.0), (15.0, 0.0), (0.0, 15.0)]:
        ned = goto(link, tgt, ALT)
        if ned is None:
            print(f"  could not reach {tgt}"); continue
        enu = gz_pose("x500_mono_cam_down_0")
        pairs.append((enu, ned))
        print(f"  NED ({ned[0]:+6.2f},{ned[1]:+6.2f})  <-  "
              f"ENU ({enu[0]:+6.2f},{enu[1]:+6.2f})")
    link.send_velocity(0, 0, -1.5)
    if len(pairs) < 2:
        print("need two points"); return

    # NED = R(theta) @ ENU + t, solved from the measured pairs
    E = np.array([p[0] for p in pairs]).T
    N = np.array([p[1] for p in pairs]).T
    Ec, Nc = E - E.mean(1, keepdims=True), N - N.mean(1, keepdims=True)
    u, _, vt = np.linalg.svd(Nc @ Ec.T)
    R = u @ vt
    t = N.mean(1) - R @ E.mean(1)
    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    resid = np.linalg.norm(N - (R @ E + t[:, None]), axis=0).max()

    pad = R @ gz_pose("arucotag") + t
    print(f"\n  rotation {theta:+.1f} deg, residual {resid:.3f} m")
    print(f"  TAG IN PX4 NED: north {pad[0]:+.2f}, east {pad[1]:+.2f}")
    print(f"  -> PAD_XY = ({pad[0]:.2f}, {pad[1]:.2f})")


if __name__ == "__main__":
    main()
