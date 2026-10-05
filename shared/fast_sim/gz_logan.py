#!/usr/bin/env python3
"""
gz_logan.py -- a patch of real Logan for Gazebo, built from the 2023 LiDAR survey.

Writes rl-landing/gz_logan/: worlds/aruco.sdf (the mission's world name, so
px4_mission.py talks to it unchanged), models/logan_{buildings,trees,roads}, and
trees_logan.json for the mission's tree-crash scoring (TREES_JSON env var).

Placement. A real rebuilt pad (logan_map_lidar.npz) goes where the mission already
expects its pad, ENU (120, 0): the aircraft spawns 120 m west of it at the origin.
Gazebo ENU = Logan (east, north) - pad + (120, 0). Ground is flat: the fast env
flies height above ground too, and this crop spans a few metres of relief.

Level of detail 1: every surveyed building is a block with a flat roof at its
median surveyed height; every tree is a trunk plus an ellipsoid crown at a
surveyed treetop (local maximum of the canopy), crown radius from its height
(0.22 h + 0.5 m, 1-5 m, never more than 0.7 of the gap to the next treetop).
Visual only, like the forest trials: trees are scored geometrically by the
mission, and the chosen pad keeps every obstacle within 45 m under 15 m.
"""
import json
import os

import numpy as np
from scipy.ndimage import label, maximum_filter

import logan_map_lidar as ML

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "px4_gazebo"))
import make_pad                                          # noqa: E402  the pad painter, reused for neighbours

PAD = (135.5, 1135.5)               # Logan metres: one of the 12 pads safe for an 18 m search
HALF_CROP, SITE = 240, (120.0, 0.0)
N_NEIGH = 15                        # DICT_4X4_50 holds 16 three-marker pads: ours + 15 neighbours
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "px4_gazebo", "gz_logan")
WORLD_SRC = "/home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/worlds/aruco.sdf"


def crop():
    lay, _, _ = ML.layers()
    road = np.load("logan_map_lidar.npz")["road"]
    r0, c0 = int(np.floor(PAD[1] + 2000)), int(np.floor(PAD[0] + 2000))
    sl = np.s_[r0 - HALF_CROP:r0 + HALF_CROP, c0 - HALF_CROP:c0 + HALF_CROP]
    # cell (i, j) of the crop spans Gazebo x in [j + X0, j + X0 + 1], y in [i + Y0, ...]
    X0 = (c0 - HALF_CROP) - 2000 - PAD[0] + SITE[0]
    Y0 = (r0 - HALF_CROP) - 2000 - PAD[1] + SITE[1]
    return {k: v[sl] for k, v in lay.items()}, road[sl], X0, Y0


class Obj:
    """Minimal OBJ writer: quads, counter-clockwise seen from outside, one flat normal
    per quad. Without normals Gazebo lit every mesh plain white (first render)."""

    def __init__(self):
        self.v, self.f, self.n = [], [], []

    def quads(self, q):                                    # q: (n, 4, 3)
        if len(q) == 0:
            return
        base, nb = sum(len(v) for v in self.v), sum(len(n) for n in self.n)
        self.v.append(q.reshape(-1, 3))
        nrm = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])           # diagonals: safe on degenerate pole quads
        self.n.append(nrm / np.linalg.norm(nrm, axis=1, keepdims=True).clip(min=1e-9))
        k = base + 4 * np.arange(len(q))[:, None] + np.array([0, 1, 2, 3])
        m = (nb + np.arange(len(q)))[:, None].repeat(3, 1)
        self.f.append(np.concatenate([np.stack([k[:, [0, 1, 2]], m], -1), np.stack([k[:, [0, 2, 3]], m], -1)]))

    def save(self, path):
        v, n, f = np.concatenate(self.v), np.concatenate(self.n), np.concatenate(self.f) + 1
        with open(path, "w") as fh:
            fh.write("".join(f"v {a:.2f} {b:.2f} {c:.2f}\n" for a, b, c in v))
            fh.write("".join(f"vn {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in n))
            fh.write("".join(f"f {a}//{na} {b}//{nb} {c}//{nc}\n" for (a, na), (b, nb), (c, nc) in f))
        return len(f)


def blocks(hb, X0, Y0):
    """LoD1 buildings: flat-roofed columns per cell, walls wherever the neighbour is lower."""
    roofs, walls = Obj(), Obj()
    i, j = np.nonzero(hb > 0); h = hb[i, j]
    x0, y0 = j + X0, i + Y0; x1, y1 = x0 + 1, y0 + 1
    roofs.quads(np.stack([np.stack([x0, y0, h], -1), np.stack([x1, y0, h], -1),
                          np.stack([x1, y1, h], -1), np.stack([x0, y1, h], -1)], 1))
    pad = np.pad(hb, 1)
    for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):                  # east, west, north, south
        nb = pad[1 + i + di, 1 + j + dj]; k = nb < h
        lo, hi = nb[k], h[k]; a, b, c, d = x0[k], y0[k], x1[k], y1[k]
        if (di, dj) == (0, 1):
            q = [(c, b, lo), (c, d, lo), (c, d, hi), (c, b, hi)]
        elif (di, dj) == (0, -1):
            q = [(a, d, lo), (a, b, lo), (a, b, hi), (a, d, hi)]
        elif (di, dj) == (1, 0):
            q = [(c, d, lo), (a, d, lo), (a, d, hi), (c, d, hi)]
        else:
            q = [(a, b, lo), (c, b, lo), (c, b, hi), (a, b, hi)]
        walls.quads(np.stack([np.stack(p, -1) for p in q], 1))
    return roofs, walls


def treetops(tree, X0, Y0):
    peak = (tree >= 4.0) & (tree == maximum_filter(tree, size=5))
    i, j = np.nonzero(peak); h = tree[i, j]
    order = np.argsort(-h); keep = []
    for k in order:                                        # tallest first, 3 m apart
        if all((i[k] - i[m]) ** 2 + (j[k] - j[m]) ** 2 >= 9 for m in keep):
            keep.append(k)
    i, j, h = i[keep], j[keep], h[keep]
    x, y = j + X0 + 0.5, i + Y0 + 0.5
    xy = np.stack([x, y], 1)
    gap = np.sort(np.linalg.norm(xy[:, None] - xy[None], axis=-1), axis=1)[:, 1] if len(xy) > 1 else np.full(len(xy), 9.0)
    r = np.minimum(np.clip(0.22 * h + 0.5, 1.0, 5.0), 0.7 * gap).clip(min=0.8)
    return x, y, h, r


def trees_mesh(x, y, h, r):
    crowns, trunks = Obj(), Obj()
    u = np.linspace(0, 2 * np.pi, 9)[:-1]; w = np.linspace(-np.pi / 2, np.pi / 2, 6)
    for xi, yi, hi, ri in zip(x, y, h, r):
        cz, az = 0.7 * hi, 0.3 * hi                        # crown spans 0.4 h .. h
        P = lambda a, b: np.array([xi + ri * np.cos(b) * np.cos(a), yi + ri * np.cos(b) * np.sin(a), cz + az * np.sin(b)])
        q = [[P(u[m], w[n]), P(u[(m + 1) % 8], w[n]), P(u[(m + 1) % 8], w[n + 1]), P(u[m], w[n + 1])]
             for n in range(5) for m in range(8)]
        crowns.quads(np.array(q))
        tr = 0.12 + 0.012 * hi; th = 0.45 * hi
        C = lambda a, z: np.array([xi + tr * np.cos(a), yi + tr * np.sin(a), z])
        uu = np.linspace(0, 2 * np.pi, 7)[:-1]
        trunks.quads(np.array([[C(uu[m], 0), C(uu[(m + 1) % 6], 0), C(uu[(m + 1) % 6], th), C(uu[m], th)] for m in range(6)]))
    return crowns, trunks


def flat(mask, X0, Y0, z):
    o = Obj(); i, j = np.nonzero(mask)
    x0, y0 = j + X0, i + Y0; zz = np.full_like(x0, z, dtype=float)
    o.quads(np.stack([np.stack([x0, y0, zz], -1), np.stack([x0 + 1, y0, zz], -1),
                      np.stack([x0 + 1, y0 + 1, zz], -1), np.stack([x0, y0 + 1, zz], -1)], 1))
    return o


def model(name, visuals):
    d = os.path.join(OUT, "models", name); os.makedirs(os.path.join(d, "meshes"), exist_ok=True)
    open(os.path.join(d, "model.config"), "w").write(
        f'<?xml version="1.0"?>\n<model><name>{name}</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>'
        f'<description>Logan 2023 LiDAR, see shared/fast_sim/gz_logan.py</description></model>\n')
    vis = "".join(
        f'<visual name="{v}"><geometry><mesh><uri>model://{name}/meshes/{v}.obj</uri></mesh></geometry>'
        f'<material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse><double_sided>true</double_sided></material></visual>'
        for v, c in visuals)
    open(os.path.join(d, "model.sdf"), "w").write(
        f'<?xml version="1.0"?>\n<sdf version="1.9"><model name="{name}"><static>true</static>'
        f'<link name="link">{vis}</link></model></sdf>\n')
    return d


def main():
    lay, road, X0, Y0 = crop()
    lab, _ = label(lay["bldg"] > 2.5)
    med = np.zeros(lab.max() + 1)
    for k in range(1, lab.max() + 1):                      # flat roof: the block's median height
        med[k] = np.median(lay["bldg"][lab == k])
    hb = med[lab] * (lab > 0)
    roofs, walls = blocks(hb, X0, Y0)
    d = model("logan_buildings", [("roofs", "0.55 0.33 0.27"), ("walls", "0.85 0.82 0.76")])
    n = roofs.save(f"{d}/meshes/roofs.obj") + walls.save(f"{d}/meshes/walls.obj")
    x, y, h, r = treetops(lay["tree"], X0, Y0)
    crowns, trunks = trees_mesh(x, y, h, r)
    d = model("logan_trees", [("crowns", "0.16 0.36 0.13"), ("trunks", "0.35 0.24 0.13")])
    n += crowns.save(f"{d}/meshes/crowns.obj") + trunks.save(f"{d}/meshes/trunks.obj")
    d = model("logan_roads", [("roads", "0.25 0.25 0.27")])
    n += flat(road & (hb == 0), X0, Y0, 0.02).save(f"{d}/meshes/roads.obj")

    trees = [dict(enu=[float(a), float(b)], ned=[float(b), float(a)], trunk_r=float(0.12 + 0.012 * c),
                  canopy_r=float(e), h=float(c), canopy_frac=0.4) for a, b, c, e in zip(x, y, h, r)]
    json.dump(dict(seed="logan", pad_enu=list(SITE), collide=False, trees=trees),
              open(os.path.join(OUT, "trees_logan.json"), "w"))
    clear = min(np.hypot(t["enu"][0] - SITE[0], t["enu"][1] - SITE[1]) - t["canopy_r"] for t in trees)
    assert clear > 2.0, f"a crown within {clear:.1f} m of the pad"

    src = open(WORLD_SRC).read()
    head = src[:src.index("    <model name='tree_0'>")]                # everything before the forest
    head = head.replace("<size>100 100</size>", "<size>1200 1200</size>") \
               .replace("<ambient>0.8 0.8 0.8 1</ambient>\n            <diffuse>0.8 0.8 0.8 1</diffuse>",
                        "<ambient>0.30 0.42 0.22 1</ambient>\n            <diffuse>0.34 0.48 0.25 1</diffuse>") \
               .replace("<background>0.7 0.7 0.7 1</background>", "<background>0.62 0.76 0.90 1</background>")
    inc = "".join(f"    <include><uri>model://{m}</uri><pose>0 0 0 0 0 0</pose></include>\n"
                  for m in ("logan_roads", "logan_buildings", "logan_trees"))
    # The neighbours' pads: each at the spot the survey chose for THAT house, same layout,
    # other ids -- ours stays (2, 1, 0) at the site. The mission must refuse all of them.
    P = np.load("logan_map_lidar.npz")["pads"] - np.array(PAD) + np.array(SITE)
    P = P[np.hypot(*(P - SITE).T) > 1.0]
    P = P[(np.abs(P - [SITE[0], 0]) < HALF_CROP - 10).all(1)]
    keep = [np.array(SITE)]                              # 5 m mats: at least 7 m between centres
    for q in P[np.argsort(np.hypot(*(P - SITE).T))]:     # (two yards' pads can meet at a fence 4 m apart)
        if len(keep) <= N_NEIGH and all(np.hypot(*(q - k)) >= 7.0 for k in keep):
            keep.append(q)
    P = np.array(keep[1:])
    pads = [dict(name="arucotag", enu=list(SITE), ids=[m[0] for m in make_pad.MARKERS], ours=True)]
    for k, (gx, gy) in enumerate(P, start=1):
        mk = [({2: 3 * k + 2, 1: 3 * k + 1, 0: 3 * k}[mid], size, x) for mid, size, x in make_pad.MARKERS]
        name = f"logan_pad_{k}"
        # 240 px/m (1200 px), not 600: seen only from altitude, and sixteen 9 MP textures plus
        # the RL training on the same GPU dropped Gazebo to 0.15-0.3x real time in places.
        make_pad.write_model(os.path.join(OUT, "models", name), name, mk, ppm=240)
        inc += f"    <include><uri>model://{name}</uri><name>{name}</name><pose>{gx:.2f} {gy:.2f} 0.001 0 0 0</pose></include>\n"
        pads.append(dict(name=name, enu=[float(gx), float(gy)], ids=[m[0] for m in mk], ours=False))
    json.dump(pads, open(os.path.join(OUT, "pads_logan.json"), "w"), indent=1)
    gap = np.hypot(*(P - SITE).T)
    print(f"neighbour pads: {len(P)}, {gap.min():.0f}-{gap.max():.0f} m from ours")
    os.makedirs(os.path.join(OUT, "worlds"), exist_ok=True)
    open(os.path.join(OUT, "worlds", "aruco.sdf"), "w").write(head + inc + "  </world>\n</sdf>\n")
    print(f"Logan patch {2 * HALF_CROP} m square around pad {PAD}: {lab.max()} buildings, {len(trees)} trees, "
          f"{n:,} triangles; nearest crown edge {clear:.1f} m from the pad -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
