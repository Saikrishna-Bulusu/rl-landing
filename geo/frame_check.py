#!/usr/bin/env python3
"""
frame_check.py -- which way does the gimbal camera's image top point? Measured, not assumed: each saved frame is
turned north-up for every candidate heading (and mirrored), scaled to the floor capture's 0.13 m pixels and
correlated with the floor under the drone's TRUE position. The best heading, minus the nose heading, is the offset
record_sweep.py must add (and the check fails loudly if the frames do not match the floor at all).

    <GeoAnchor venv python> geo/frame_check.py gazebo_geo/floor
"""
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from geo_world import EXTENT, RES

FX = 640.0 / math.tan(1.0)                         # PX4's gimbal camera: 1280 x 720 px, 2.0 rad across
C = 0.4                                            # m per pixel for the comparison: coarse, so roads and blocks
                                                   # line up even where 3D roofs lean and trees hide the floor


def north_up(img, top_heading, mirror=False):
    if mirror:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -top_heading, 1.0)       # GeoAnchor's rectify: rotate by -yaw
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh))


def main(folder, n=12):
    tex = cv2.cvtColor(cv2.imread(f"{R}/gz_logan/geo_floor/materials/textures/logan_2023.png"), cv2.COLOR_BGR2GRAY)
    tex = cv2.resize(tex, None, fx=RES / C, fy=RES / C, interpolation=cv2.INTER_AREA)
    x0, x1, y0, y1 = EXTENT
    rows = [json.loads(l) for l in open(f"{R}/{folder}/flight_gt.jsonl")]
    offs = []
    for r in rows[::max(1, len(rows) // (3 * n))]:
        cx, cy = (r["gz"][0] - x0) / C, (y1 - r["gz"][1]) / C
        gsd = r["alt_agl_m"] / FX
        side = int(0.7 * 720 * gsd / C)                  # the square inside the circle every rotation keeps
        half = side // 2 + int(15 / C)                   # +-15 m of search around the true position
        if min(cx, cy) - half < 0 or cx + half >= tex.shape[1] or cy + half >= tex.shape[0]:
            continue                                     # too near the floor's edge
        patch = tex[int(cy) - half:int(cy) + half, int(cx) - half:int(cx) + half]
        vid = cv2.VideoCapture(f"{R}/{folder}/flight.mp4")
        vid.set(cv2.CAP_PROP_POS_FRAMES, r["frame_index"])
        ok, img = vid.read()
        assert ok, r["frame_index"]
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=gsd / C, fy=gsd / C, interpolation=cv2.INTER_AREA)
        best = (-2, None, None)
        for mirror in (False, True):
            for hd in range(0, 360, 2):
                q = north_up(img, hd, mirror)
                q = q[q.shape[0] // 2 - side // 2:q.shape[0] // 2 + side // 2, q.shape[1] // 2 - side // 2:q.shape[1] // 2 + side // 2]
                score = cv2.matchTemplate(patch, q, cv2.TM_CCOEFF_NORMED).max()
                if score > best[0]:
                    best = (score, hd, mirror)
        off = (best[1] - r["nose_deg"] + 540) % 360 - 180
        offs.append((best[0], off, best[2]))
        print(f"frame {r['frame_index']:4d} at ({r['gz'][0]:6.1f}, {r['gz'][1]:6.1f}): nose {r['nose_deg']:6.1f}  "
              f"best top heading {best[1]:3d}{' mirrored' if best[2] else ''}  NCC {best[0]:.2f}  -> offset {off:+.0f} deg")
        if len(offs) == n:
            break
    good = [o for o in offs if o[0] > 0.3]
    assert len(good) >= n // 2, "most frames match nothing under their true position: frames, truth or floor misaligned"
    med = float(np.median([o[1] for o in good]))
    spread = max(abs((o[1] - med + 180) % 360 - 180) for o in good)
    print(f"image top = nose heading {med:+.0f} deg; {len(good)} of {len(offs)} frames matched, worst {spread:.0f} deg "
          f"from that, mirrored {sum(o[2] for o in good)} of {len(good)}")
    assert spread <= 10 and not any(o[2] for o in good), "frames disagree: the offset is not measured"


if __name__ == "__main__":
    main(sys.argv[1])
