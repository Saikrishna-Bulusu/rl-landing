#!/usr/bin/env python3
"""
What the 2023 survey adds to the Logan map (lidar_logan.npz, from lidar_map.py):
the ground's real shape, every tree, and where the survey and the map disagree
about which buildings exist. Also: how many back-yard pads have a tree over them.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap

L, M = np.load("lidar_logan.npz"), np.load("logan_map.npz")
HALF = 2000
ext = (-HALF, HALF, -HALF, HALF)
g = L["ground"].astype(np.float32); tree = L["tree"].astype(np.float32); bldg = L["bldg"].astype(np.float32)
Hm = M["H"].astype(np.float32)
foot = (Hm > 0) & ~np.isin(Hm, (15.0, 10.0))          # buildings only: 15/10 m are OSM wood / tree-row fills

# ---- pads under trees: tallest tree within 3 m of each back-yard pad ----
yy, xx = np.mgrid[-3:4, -3:4]; disc = (xx ** 2 + yy ** 2) <= 9
tallest = []
for x, y in M["pads"]:
    r, c = int(y) + HALF, int(x) + HALF
    win = tree[r - 3:r + 4, c - 3:c + 4]
    tallest.append(win[disc].max() if win.shape == disc.shape else 0.0)
tallest = np.array(tallest)
for h in (2.0, 5.0, 10.0):
    print(f"pads with a tree taller than {h:>4.0f} m within 3 m: {(tallest > h).mean():6.1%}  ({(tallest > h).sum()} of {len(tallest)})")
print(f"ground: {np.nanmin(g):.1f} to {np.nanmax(g):.1f} m AHD (a {np.nanmax(g) - np.nanmin(g):.0f} m range)")
print(f"trees above 5 m cover {(tree > 5).mean():.1%} of the map; tallest {np.nanmax(tree):.0f} m")

fig, axes = plt.subplots(1, 3, figsize=(21, 7.4), layout="constrained")
ax = axes[0]
shade = LightSource(azdeg=315, altdeg=40).shade(np.nan_to_num(g, nan=np.nanmin(g)), cmap=plt.cm.terrain,
                                               vert_exag=4, blend_mode="soft")
ax.imshow(shade, origin="lower", extent=ext)
ax.set_title(f"ground: {np.nanmin(g):.0f}-{np.nanmax(g):.0f} m above sea level")
ax = axes[1]
ax.imshow(np.where(tree > 2, tree, np.nan), origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=30)
ax.imshow(np.where(bldg > 2, 1.0, np.nan), origin="lower", extent=ext, cmap=ListedColormap(["#9a9a9a"]))
ax.scatter(M["pads"][:, 0], M["pads"][:, 1], s=4, c=np.where(tallest > 5, "#c0392b", "#e08a00"), lw=0)
ax.set_title(f"trees (green, darker = taller), buildings (grey)\npads: red = tree over 5 m within 3 m ({(tallest > 5).mean():.0%})")
ax = axes[2]
agree = foot & ~L["map_only"]
rgb = np.ones(foot.shape + (3,))
rgb[agree] = (0.55, 0.55, 0.55); rgb[L["map_only"]] = (0.85, 0.2, 0.15); rgb[L["survey_only"]] = (0.15, 0.4, 0.85)
ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
ax.set_title("buildings: grey = both agree, red = map only, blue = survey only\n(survey shifted "
             f"{float(L['shift'][0]):+.1f} m E, {float(L['shift'][1]):+.1f} m N)")
for a in axes:
    a.set_xlabel("east (m)"); a.set_aspect("equal")
axes[0].set_ylabel("north (m)")
fig.suptitle("Logan 2023 LiDAR on the map's own grid", fontsize=15)
fig.savefig("lidar_logan.png", dpi=110); print("lidar_logan.png")
