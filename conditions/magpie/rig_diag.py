#!/usr/bin/env python3
"""
rig_diag.py -- the rig drone as PX4 runs it, parked at the depot: put the magpie at known spots
around it and check each link the first live rig flight failed (25 Sept, 01:18: detections 122 deg
off the bird, 2 frame sets a second instead of 20).
  1. do all six cameras render in step (the frame-set assembly waits for stamps within 25 ms)?
  2. where does the magpie really appear (render difference) vs where gz_rig.project puts it --
     with Gazebo's pose of the drone, and with PX4's attitude through gz_learner.enu_rpy?
  3. does the detector find it there?
Needs the rigfly SITL up (rigfly.sh's environment); run with the system Python.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gz_learner as G
import gz_rig as GR


def main():
    from pymavlink import mavutil
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    last, stamps = {}, {n: [] for n in GR.RIG}

    def on(n, m):
        s = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        last[n] = (s, m); stamps[n].append(s)
    node = Node()
    for n in GR.RIG:
        node.subscribe(Image, f"/rig/{n}", lambda m, n=n: on(n, m))
    mv = mavutil.mavlink_connection("udpin:0.0.0.0:14540", source_system=255)
    mv.wait_heartbeat(timeout=30)
    att = mv.recv_match(type="ATTITUDE", blocking=True, timeout=10)
    rpy = G.enu_rpy(att.roll, att.pitch, att.yaw)
    print(f"PX4 attitude (NED, deg): roll {math.degrees(att.roll):.1f} pitch {math.degrees(att.pitch):.1f} yaw {math.degrees(att.yaw):.1f}"
          f" -> Gazebo rpy (deg) {[round(math.degrees(a), 1) for a in rpy]}  (the model stands at yaw 0)")
    time.sleep(8)
    # 1. are the six cameras in step?
    sets = sorted(set(round(s, 3) for s in stamps["front"]))[-20:]
    for n in GR.RIG:
        st = np.array(stamps[n][-20:])
        print(f"  {n:5s}: {len(stamps[n])} images in 8 s wall, last stamps {np.round(st[-3:], 3).tolist()}")
    spread = [max(last[n][0] for n in GR.RIG) - min(last[n][0] for n in GR.RIG) for _ in range(1)]
    print(f"  spread of the six newest stamps: {spread[0] * 1000:.0f} ms")

    def grab(k=6):
        """Six images all taken k frames after now (the drone stands still; the magpie was just moved)."""
        after = max(s for s, _ in last.values()) + k / GR.HZ
        while min(last[n][0] for n in GR.RIG) < after:
            time.sleep(0.01)
        return {n: np.frombuffer(last[n][1].data, np.uint8).reshape(GR.H, GR.W, 3).copy() for n in GR.RIG}

    mover = G.Mover()
    mover.put("magpie", [0, 0, -50]); mover.put("magpie_nest", [0, 0, -50])
    bg = grab()
    base = np.array([0.0, 0.0, GR.BASE_Z])                    # base_link: the model stands at the origin
    out = []
    for rel in ([12, 0, 3], [0, 12, 3], [-12, 0, 3], [0, -12, 3], [6, 6, 10], [-8, 4, 1]):
        p = base + rel
        to = -np.asarray(rel, float)
        mover.put("magpie", p, yaw=math.atan2(to[1], to[0]))
        img = grab()
        for n in GR.RIG:
            m = np.abs(img[n].astype(np.int16) - bg[n]).max(axis=2) > 12
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            seen = (float(xs.mean()), float(ys.mean()))
            q_gz = GR.project(p, base, np.eye(3), n)
            q_px4 = GR.project(p, base, GR.rot(*rpy), n)
            print(f"magpie at {rel}: {n:5s} camera sees it at ({seen[0]:6.1f},{seen[1]:6.1f}); predicted with Gazebo's pose "
                  f"{None if q_gz is None else tuple(round(v, 1) for v in q_gz)}, with PX4's attitude "
                  f"{None if q_px4 is None else tuple(round(v, 1) for v in q_px4)}")
            out.append(dict(rel=rel, cam=n, seen=seen, q_gz=q_gz, q_px4=q_px4))
            np.save(f"/tmp/claude-1000/-home-sai-RL-projects/128bb831-cfb2-4ef5-aef7-fdaa908caf06/scratchpad/rig_diag_{n}_{rel[0]}_{rel[1]}_{rel[2]}.npy", img[n])
    json.dump(out, open(f"{HERE}/gazebo_rig/diag.json", "w"), default=float)
    os._exit(0)


if __name__ == "__main__":
    main()
