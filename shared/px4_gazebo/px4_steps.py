#!/usr/bin/env python3
"""
px4_steps.py -- how PX4 (x500, rig v2) answers velocity commands, measured on Gazebo's truth in sim time: the
target the MuJoCo drone (conditions/wind/x500_mjx.py) must reproduce before anything learned in it means anything.

measure_tau.py (the fast simulator's TAU_XY 0.69 / TAU_Z 0.23) timed PX4's own telemetry by the wall clock; this
logs Gazebo's true pose with its sim-time stamp next to every command, so a stale read or a slow loop cannot
pass for the aircraft's lag.

    OUT=steps ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u shared/px4_gazebo/px4_steps.py --out outputs/gazebo_wind/steps/steps.csv
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

import px4_mission as PM

# (seconds, north m/s, east m/s, up m/s) after the climb; held steps, then back to hover
STEPS = [(6, 0, 0, 0), (8, 3, 0, 0), (8, 0, 0, 0), (8, 0, 3, 0), (8, 0, 0, 0),
         (5, 0, 0, -1.5), (5, 0, 0, 0), (5, 0, 0, 1.5), (5, 0, 0, 0), (10, 8, 0, 0), (10, 0, 0, 0)]


class WindLog:
    """The wind WindDrag made (/wind_truth, 50 Hz), stamped with the newest truth time."""

    def __init__(self, truth):
        from gz.transport13 import Node
        from gz.msgs10.vector3d_pb2 import Vector3d
        self.truth, self.rows, self.node = truth, [], Node()
        self.node.subscribe(Vector3d, "/wind_truth", self._on)

    def _on(self, m):
        with self.truth.lock:
            t = self.truth.rows[-1][0] if self.truth.rows else 0.0
        self.rows.append((t, m.x, m.y, m.z))


class TruthLog:
    """Every pose Gazebo publishes for the drone: sim time, position, orientation (w, x, y, z)."""

    def __init__(self, name="x500_mono_cam_down_0"):
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V
        self.name, self.rows, self.lock, self.node = name, [], threading.Lock(), Node()
        self.node.subscribe(Pose_V, "/world/aruco/dynamic_pose/info", self._on)

    def _on(self, m):
        for q in m.pose:
            if q.name == self.name:
                st, p, o = m.header.stamp, q.position, q.orientation
                with self.lock:
                    self.rows.append((st.sec + st.nsec * 1e-9, p.x, p.y, p.z, o.w, o.x, o.y, o.z))
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt", type=float, default=30.0)
    ap.add_argument("--hover", type=float, default=0.0, help="instead of the steps: hover this many seconds at --alt")
    a = ap.parse_args()
    truth = TruthLog()
    wind = WindLog(truth)
    mi = PM.Mission(endpoint=f"udpin:0.0.0.0:{os.environ.get('MAV_PORT', 14540)}")
    assert mi.arm_offboard(), "could not arm"
    cmds = []                                                  # (sim time, north, east, up) as sent

    def hold(secs, vn, ve, vu, until_alt=None):
        t0 = None
        while True:
            p = mi.pos()
            if p is None:
                continue
            t0 = mi.t_boot if t0 is None else t0
            if (until_alt is not None and p[2] >= until_alt) or (until_alt is None and mi.t_boot - t0 >= secs):
                return
            mi.send(vn, ve, vu); mi.gimbal_down()
            cmds.append((mi.t_boot, vn, ve, vu, mi.roll, mi.pitch))
            time.sleep(0.02)

    hold(0, 0, 0, 2.0, until_alt=a.alt)                        # climb
    for secs, vn, ve, vu in ([(a.hover, 0, 0, 0)] if a.hover else STEPS):
        hold(secs, vn, ve, vu)
    mi.send(0, 0, -1.0)
    with truth.lock:
        tr = np.array(truth.rows)
    c = np.array(cmds)
    np.savez(a.out.replace(".csv", ".npz"), truth=tr, cmds=c, steps=np.array(STEPS, float), wind=np.array(wind.rows))
    print(f"{len(tr)} truth poses over {tr[-1, 0] - tr[0, 0]:.0f} s sim; {len(c)} commands -> {a.out.replace('.csv', '.npz')}")


if __name__ == "__main__":
    main()
