#!/usr/bin/env python3
"""
record_sweep.py -- a 75 m sweep over the Logan square with rig v2's landing camera held straight down by its gimbal,
recorded the way GeoAnchor replays a flight: the frames as a video, and a truth sidecar per frame (lat, lon, height,
the compass heading of the IMAGE TOP, camera tilt) -- so GeoAnchor's own pipeline can be run on it unchanged.

    OUT=geo_rec_floor WORLD=geo_floor INSTANCE=5 ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u gnss_denied/record_sweep.py --out outputs/gazebo_geo/floor
    OUT=geo_rec_plain WORLD=voxfast_v2 INSTANCE=5 ./shared/px4_gazebo/sitl.sh /usr/bin/python3 -u gnss_denied/record_sweep.py --out outputs/gazebo_geo/plain

The image top's heading is the body heading plus an offset (--top-offset, degrees) MEASURED from the frames against the
floor (gnss_denied/frame_check.py), not assumed: the gimbal model sits in the drone rotated 180 degrees.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path[:0] = [f"{R}/shared/px4_gazebo", HERE]
import px4_mission as PM
from px4_steps import TruthLog
from geo_world import gz_to_ll

ALT = 75.0
V = 7.0
LANES = [((-100.0, -120.0), (340.0, -120.0)), ((340.0, 0.0), (-100.0, 0.0)), ((-100.0, 120.0), (340.0, 120.0))]


class Frames:
    """The gimbal camera's newest picture and its sim-time stamp."""

    def __init__(self, topic=PM.GIMBAL_TOPIC):
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image
        self.lock, self.latest, self.node = threading.Lock(), None, Node()
        self.node.subscribe(Image, topic, self._on)

    def _on(self, m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        with self.lock:
            self.latest = (m.header.stamp.sec + m.header.stamp.nsec * 1e-9, img.copy())


def heading_deg(q):
    """Compass heading (0 north, 90 east) of the body's nose from a Gazebo ENU quaternion (w, x, y, z)."""
    w, x, y, z = q
    yaw_enu = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (90.0 - math.degrees(yaw_enu)) % 360.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-offset", type=float, default=0.0, help="image top heading = nose heading + this (deg); measured 0 (gnss_denied/frame_check.py)")
    a = ap.parse_args()
    os.makedirs(f"{R}/{a.out}/frames", exist_ok=True)
    truth, cam = TruthLog(), Frames()
    mi = PM.Mission(endpoint=f"udpin:0.0.0.0:{os.environ.get('MAV_PORT', 14540)}")
    assert mi.arm_offboard(), "could not arm"
    vw, side, k, t_last = None, open(f"{R}/{a.out}/flight_gt.jsonl", "w"), 0, -1.0

    def fly_to(x, y, z, record):
        nonlocal vw, k, t_last
        t0 = None
        while True:
            p = mi.pos()
            if p is None:
                continue
            t0 = mi.t_boot if t0 is None else t0
            me = np.array([p[1], p[0], p[2]])                       # PX4 estimate, Gazebo ENU (GPS on here)
            d = np.array([x, y]) - me[:2]
            if np.linalg.norm(d) < 2.0 and abs(z - me[2]) < 1.0:
                return
            if mi.t_boot - t0 > 400:
                raise RuntimeError("leg timed out")
            v = d / max(np.linalg.norm(d), 1e-6) * min(V, 0.8 * np.linalg.norm(d))
            if np.linalg.norm(d) > 5.0:
                mi.yaw_hold = math.atan2(d[0], d[1])                 # PX4 yaw (NED): nose along the leg
            mi.send(v[1], v[0], float(np.clip(0.8 * (z - me[2]), -2.0, 3.0))); mi.gimbal_down()
            with cam.lock:
                fr = cam.latest
            if record and fr is not None and fr[0] - t_last >= 0.5:
                with truth.lock:
                    rows = np.array(truth.rows[-400:])
                if len(rows):
                    j = int(np.argmin(np.abs(rows[:, 0] - fr[0])))
                    if abs(rows[j, 0] - fr[0]) < 0.05:
                        t_last = fr[0]
                        g = rows[j]
                        lat, lon = gz_to_ll(g[1], g[2])
                        hdg = heading_deg(g[4:8])
                        bgr = cv2.cvtColor(fr[1], cv2.COLOR_RGB2BGR)
                        if vw is None:
                            vw = cv2.VideoWriter(f"{R}/{a.out}/flight.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 2.0,
                                                 (bgr.shape[1], bgr.shape[0]))
                        vw.write(bgr)
                        if k % 10 == 0:
                            cv2.imwrite(f"{R}/{a.out}/frames/{k:04d}.jpg", bgr)
                        side.write(json.dumps(dict(t=round(fr[0], 3), frame_index=k, lat=round(lat, 8), lon=round(lon, 8),
                                                   alt_agl_m=round(float(g[3]) + 0.1, 2),
                                                   yaw_deg=round((hdg + a.top_offset) % 360.0, 2), roll_deg=0.0, pitch_deg=0.0,
                                                   gz=[round(float(v_), 3) for v_ in g[1:4]], nose_deg=round(hdg, 2))) + "\n")
                        k += 1
            time.sleep(0.02)

    p = mi.pos()
    while p is None:
        p = mi.pos()
    fly_to(0.0, 0.0, ALT, False)                                     # straight up
    for (xa, ya), (xb, yb) in LANES:
        fly_to(xa, ya, ALT, False)
        fly_to(xb, yb, ALT, True)
    side.close()
    if vw is not None:
        vw.release()
    print(f"{k} frames -> {R}/{a.out}")


if __name__ == "__main__":
    main()
