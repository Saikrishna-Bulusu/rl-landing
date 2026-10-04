#!/usr/bin/env python3
"""
Why did the frame check fail? The survey's buildings overlapped the map's
footprints best after shifting the survey 2.5 m east, and even then only IoU 0.495.
  1. Is the offset one constant (the footprint source is offset: imagery) or does
     it vary across the map (the conversion is wrong: scale or rotation)?
     -> best shift per 1 km block.
  2. Is IoU ~0.5 simply what 1 m cells do to 15 x 12 m houses?
     -> overlap allowing one cell of edge error.
Run with /usr/bin/python3.
"""
import numpy as np

import lidar_map as L

hgt, g = L.survey()
E, N = L.map_to_mga()
foot = np.load("logan_map.npz")["H"].astype(np.float32) > 0
B = 1000                                                    # 4 x 4 blocks of 1 km
steps = np.arange(-4, 4.01, 0.5)
inter = np.zeros((len(steps), len(steps), 4, 4)); union = np.zeros_like(inter)
for a, dx in enumerate(steps):
    for b, dy in enumerate(steps):
        s = L.sample(hgt["bldg"], E, N, dx, dy) > 2.0
        inter[a, b] = (s & foot).reshape(4, B, 4, B).sum((1, 3))
        union[a, b] = (s | foot).reshape(4, B, 4, B).sum((1, 3))
iou = inter / np.maximum(union, 1)

print("best survey shift per 1 km block (east, north) m and IoU there; row = south to north")
n_foot = foot.reshape(4, B, 4, B).sum((1, 3))
for r in range(4):
    cells = []
    for c in range(4):
        if n_foot[r, c] < 2000:
            cells.append("        (few buildings)        "); continue
        a, b = np.unravel_index(iou[:, :, r, c].argmax(), iou.shape[:2])
        cells.append(f"({steps[a]:+.1f},{steps[b]:+.1f}) IoU {iou[a, b, r, c]:.2f} vs {iou[8, 8, r, c]:.2f}")
    print("  " + " | ".join(cells))

tot = inter.sum((2, 3)) / np.maximum(union.sum((2, 3)), 1)
a, b = np.unravel_index(tot.argmax(), tot.shape)
dx, dy = steps[a], steps[b]
s = L.sample(hgt["bldg"], E, N, dx, dy) > 2.0


def grow(m):                                               # one cell in every direction
    o = m.copy()
    for ax in (0, 1):
        for d in (1, -1):
            o |= np.roll(m, d, ax)
    return o


print(f"\nwhole map, best shift ({dx:+.1f}, {dy:+.1f}): IoU {tot[a, b]:.3f}")
print(f"  survey building cells within 1 m of a footprint: {(s & grow(foot)).sum() / s.sum():.3f}")
print(f"  footprint cells within 1 m of a survey building: {(foot & grow(s)).sum() / foot.sum():.3f}")
print(f"  footprint cells with NO survey building within 3 m: "
      f"{(foot & ~grow(grow(grow(s)))).sum() / foot.sum():.3f}  (demolished, never built, or not a building)")
