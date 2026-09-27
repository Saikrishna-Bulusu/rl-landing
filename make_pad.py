#!/usr/bin/env python3
"""
make_pad.py -- the delivery pad, three markers, generated not hand-drawn.

WHY A THIRD MARKER, AND WHY IT IS 1.80 m AND NOT 2.40 m

A marker can only be decoded from h = size * PX / (2 * MIN_PX * tan(hfov/2)).
At 1.20 m that is 19.0 m, which is BELOW a mature tree. The aircraft is then
forced to hunt for the pad from inside the canopy layer, which is the only
place flying is dangerous. So the ladder gains a rung at the top.

The first version of this rung was 2.40 m, sized off a detection model that
treated MIN_PX = 34 as a cliff. Sampling detection from the measured curve
instead (bigger_marker.py, DETECT="curve") changed the answer: 1.80 m and
2.40 m BOTH deliver 100% once the time budget is realistic, while 1.20 m
plateaus at 69% however long you give it. 2.40 m buys nothing over 1.80 m, and
under a tight 35 s budget it is WORSE -- 86.7% against 98.0% -- because seeing
the pad from 46 m makes the rule commit to a descent while its position
estimate is still about 12 m wrong.

So: 1.80 m. It is also a third less paint and fits a 5.0 m mat instead of 6.6.

Each marker must be large enough to be acquired from the altitude above it and
small enough to still fit in frame below, and the bands have to overlap or
there is a blind gap in the descent:

    1.80 m  id 2   usable 28.6 m -> 0.76 m
    1.20 m  id 0   usable 19.0 m -> 0.51 m
    0.15 m  id 1   usable  2.4 m -> 0.06 m

NESTING IS NOT AN OPTION. Any centre hole destroys the outer marker's data
bits; all 50 ids of DICT_4X4_50 were tested and not one survives a 7% hole.
They sit side by side, and id 1 STAYS AT THE MAT CENTRE because the model
origin is the touchdown point that find_pad.py measures.

Quiet zone is one marker cell (size/6) of white on every side -- less and the
detector loses the black border against the mat.
"""
import argparse
import os

import cv2
import numpy as np

MAT = 5.0              # metres, square: the paint, mostly quiet zone
PPM = 600              # px per metre; every marker is then a multiple of 6 px
DICT = cv2.aruco.DICT_4X4_50

#            id  size_m   centre_x_m   (y is 0: a row, so the mat stays square)
MARKERS = [(2,  1.80,   -1.275),
           (1,  0.15,    0.000),       # MODEL ORIGIN. Do not move.
           (0,  1.20,   +0.875)]

MODEL = ("/home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/models/"
         "arucotag")

SDF = """<?xml version="1.0" encoding="UTF-8"?>
<sdf version='1.9'>
  <model name='{name}'>
    <static>true</static>
    <pose>0 0 0.001 0 0 0</pose>
    <link name='base'>
      <visual name='base_visual'>
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>{mat} {mat}</size>
          </plane>
        </geometry>
        <material>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.4 0.4 0.4 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://{name}/{name}.png</albedo_map>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def build(markers=MARKERS, ppm=PPM):
    n = int(round(MAT * ppm))
    img = np.full((n, n), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICT)
    gen = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
    for mid, size_m, cx_m in markers:
        s = int(round(size_m * ppm))
        assert s % 6 == 0, f"id {mid}: {s} px is not a whole number of cells"
        m = gen(d, mid, s)
        x0 = int(round(n / 2 + cx_m * ppm)) - s // 2
        y0 = n // 2 - s // 2
        img[y0:y0 + s, x0:x0 + s] = m
    return img


def check(img, markers=MARKERS, ppm=PPM):
    """Every claim this pad makes, verified against the pixels."""
    n = img.shape[0]
    d = cv2.aruco.getPredefinedDictionary(DICT)
    try:
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        c, i, _ = det.detectMarkers(img)
    except AttributeError:
        c, i, _ = cv2.aruco.detectMarkers(img, d)
    got = sorted(int(x) for x in i.flatten()) if i is not None else []
    want = sorted(m[0] for m in markers)
    assert got == want, f"decoded {got}, expected {want}"

    found = {int(x): cc.reshape(4, 2) for cc, x in zip(c, i.flatten())}
    for mid, size_m, cx_m in markers:
        p = found[mid]
        side = float(np.linalg.norm(p[0] - p[1])) / ppm
        ctr = p.mean(0)
        dx = float(ctr[0] - n / 2) / ppm
        dy = float(ctr[1] - n / 2) / ppm
        assert abs(side - size_m) < 0.02, f"id {mid} measures {side:.3f} m, want {size_m}"
        assert abs(dx - cx_m) < 0.02, f"id {mid} at x={dx:+.3f}, want {cx_m:+.3f}"
        assert abs(dy) < 0.02, f"id {mid} is off the centre line at y={dy:+.3f}"
        print(f"   ok  id {mid}: {side:.3f} m at ({dx:+.3f}, {dy:+.3f}) m")

    # The fine marker IS the touchdown point: it must sit on the model origin.
    p = found[next(m[0] for m in markers if m[2] == 0.0)].mean(0)
    off = np.hypot(p[0] - n / 2, p[1] - n / 2) / ppm
    assert off < 0.01, f"the fine marker is {off:.3f} m off the model origin"
    print(f"   ok  fine marker is {off*1000:.1f} mm from the model origin")

    # No two markers may touch, and each needs a cell of white around it.
    span = {m[0]: (m[2] - m[1] / 2, m[2] + m[1] / 2, m[1]) for m in markers}
    order = sorted(span.items(), key=lambda kv: kv[1][0])
    for (a, sa), (b, sb) in zip(order, order[1:]):
        gap = sb[0] - sa[1]
        need = max(sa[2], sb[2]) / 6.0
        assert gap >= need - 1e-9, (
            f"ids {a} and {b} are {gap:.3f} m apart, quiet zone needs {need:.3f} m")
        print(f"   ok  ids {a}->{b} gap {gap:.3f} m >= quiet zone {need:.3f} m")
    edge = min(order[0][1][0] + MAT / 2, MAT / 2 - order[-1][1][1])
    need = max(m[1] for m in markers) / 6.0
    assert edge >= need - 1e-9, f"only {edge:.3f} m of mat outside the markers"
    print(f"   ok  mat margin {edge:.3f} m >= {need:.3f} m")


def write_model(model_dir, name, markers=MARKERS, img=None, ppm=PPM):
    """A pad model with these markers (same layout, other ids): a neighbour's pad."""
    img = build(markers, ppm) if img is None else img
    check(img, markers, ppm)
    os.makedirs(model_dir, exist_ok=True)
    cv2.imwrite(os.path.join(model_dir, f"{name}.png"), img)
    with open(os.path.join(model_dir, "model.sdf"), "w") as f:
        f.write(SDF.format(mat=MAT, name=name))
    cfg = os.path.join(model_dir, "model.config")
    if not os.path.exists(cfg):
        with open(cfg, "w") as f:
            f.write(f'<?xml version="1.0"?>\n<model><name>{name}</name><version>1.0</version>'
                    f'<sdf version="1.9">model.sdf</sdf><description>ArUco pad, ids '
                    f'{[m[0] for m in markers]}, make_pad.py</description></model>\n')


def detect_alt(size_m, px=1280, min_px=34.0, fov_deg=99.7):
    return size_m * px / (2.0 * min_px * np.tan(np.radians(fov_deg) / 2.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="install into PX4")
    a = ap.parse_args()

    img = build()
    print(f"1. built {img.shape[0]}x{img.shape[0]} px on a {MAT} m mat "
          f"({PPM} px/m)")
    print("2. verifying against the pixels")
    check(img)

    print("\n3. the descent ladder")
    prev_lo = None
    for mid, size_m, _ in sorted(MARKERS, key=lambda m: -m[1]):
        hi = detect_alt(size_m)
        lo = size_m / (2 * np.tan(np.radians(99.7) / 2))   # fits in frame
        gap = "" if prev_lo is None or hi >= prev_lo else "  <-- BLIND GAP"
        print(f"   id {mid}  {size_m:.2f} m   usable {hi:5.1f} m -> {lo:.2f} m{gap}")
        assert prev_lo is None or hi >= prev_lo, (
            f"nothing is readable between {prev_lo:.1f} m and {hi:.1f} m")
        prev_lo = lo
    print("   ok  the bands overlap: no altitude with nothing to see")

    if a.write:
        write_model(MODEL, "arucotag", MARKERS, img)
        print(f"\n4. installed into {MODEL}")
    else:
        print("\n   (dry run -- pass --write to install)")
