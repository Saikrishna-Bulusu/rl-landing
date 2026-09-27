#!/usr/bin/env python3
"""
logan_map.py -- the delivery world, built from real map data for Logan, QLD.

WHAT IS REAL AND WHAT IS ASSUMED

  Real  building footprints    Microsoft Global ML Building Footprints (ODbL),
                               4,582 in the study area. OpenStreetMap has only
                               378 here -- most Logan houses are unmapped -- so
                               OSM supplies roads, land use, trees, the helipad
                               and the hub, and Microsoft supplies the houses.
  Real  building heights       Microsoft's ML estimate (median 3.6 m), or the
                               OSM height / levels tag where one exists.
  Real  helipad                Logan Hospital (OSM aeroway=helipad). The world
                               is centred on it because CASA's 1.4 km helicopter
                               landing site rule is the regulation that the
                               helicopter intruder exercises.
  Real  drone hub              Palms Shopping Centre (OSM landuse=retail).
  Real  roads, residential     OSM. Roads carry the vehicle intruders.
  Real  trees, where mapped    OSM, which maps only 39 tree features here. Real
                               Logan has far more canopy; this under-counts it.
  Assumed  roof allowance      +1.0 m on Microsoft heights, floor 3.0 m: the ML
                               estimate reads like eave height, and underrating
                               an obstacle is the unsafe direction to be wrong.
  Assumed  pad placement       one pad per house, in open ground >= 4 m from any
                               obstacle and >= 12 m from a road (a back yard).

The grid is 4 km x 4 km at 1 m, x east and y north, origin on the helipad.
"""
import json
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HELIPAD_LL = (-27.67039, 153.14105)      # Logan Hospital, OSM aeroway=helipad
ARCHERFIELD_LL = (-27.56945, 153.00598)  # nearest controlled airport (YBAF)
HALF, RES = 2000, 1.0
N = int(2 * HALF / RES)
ROOF_M, FLOOR_M = 1.0, 3.0
PAD_CLEAR_M, PAD_ROAD_M, PAD_HOUSE_M = 4.0, 12.0, 25.0
ROAD_W = {"motorway": 14, "trunk": 12, "primary": 11, "secondary": 10,
          "tertiary": 9, "residential": 7, "unclassified": 7, "living_street": 6}


def to_xy(lat, lon):
    """Local metres, east/north of the helipad. Equirectangular.
    ponytail: 110,574 m/deg is the EQUATOR's; at 27.7 S a degree of latitude is
    ~110,814 m, so north is compressed 0.22% (~4 m at the map edge; east is off
    0.07%). Harmless while everything goes through this function -- lidar_map.py
    resamples the survey through it too. Fix it only together with a full rebuild."""
    lat0, lon0 = HELIPAD_LL
    return ((lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0,
            (lat - lat0) * 110_574.0)


def to_px(x, y):
    return ((x + HALF) / RES, (y + HALF) / RES)          # (col, row)


def load_features():
    osm = json.load(open("logan_osm.json"))["elements"]
    ms = json.load(open("ms_logan.geojson"))["features"]
    blds = []                                   # (polygon xy, height, source)
    for f in ms:
        ring = [to_xy(lat, lon) for lon, lat in f["geometry"]["coordinates"][0]]
        h = f["properties"].get("height") or -1
        blds.append((ring, max(FLOOR_M, (h if h > 0 else FLOOR_M) + ROOF_M), "ms"))
    trees, woods, roads, resid, hub = [], [], [], [], None
    for e in osm:
        t, g = e.get("tags", {}), e.get("geometry")
        if "building" in t and g and ("height" in t or "building:levels" in t):
            try:
                h = float(str(t.get("height", "")).split()[0]) if "height" in t \
                    else 3.0 * float(t["building:levels"]) + 1.0
                blds.append(([to_xy(p["lat"], p["lon"]) for p in g], h, "osm"))
            except ValueError:
                pass
        if t.get("natural") == "tree" and "lat" in e:
            trees.append((to_xy(e["lat"], e["lon"]), float(t.get("height", 10))))
        if t.get("natural") in ("wood", "scrub", "tree_row") and g:
            woods.append(([to_xy(p["lat"], p["lon"]) for p in g], t["natural"]))
        if t.get("highway") in ROAD_W and g:
            roads.append(([to_xy(p["lat"], p["lon"]) for p in g], t["highway"]))
        if t.get("landuse") == "residential" and g:
            resid.append([to_xy(p["lat"], p["lon"]) for p in g])
        if t.get("name") == "Palms Shopping Centre" and g:
            hub = np.mean([to_xy(p["lat"], p["lon"]) for p in g], axis=0)
    return blds, trees, woods, roads, resid, hub


def build():
    blds, trees, woods, roads, resid, hub = load_features()

    # Heights: draw shortest first so overlaps keep the taller structure.
    img = Image.new("F", (N, N), 0.0)
    d = ImageDraw.Draw(img)
    for ring, h, _ in sorted(blds, key=lambda b: b[1]):
        if len(ring) >= 3:
            d.polygon([to_px(*p) for p in ring], fill=float(h))
    for ring, kind in woods:
        pts = [to_px(*p) for p in ring]
        if kind == "tree_row":
            d.line(pts, fill=10.0, width=int(6 / RES))
        elif len(pts) >= 3:
            d.polygon(pts, fill=15.0 if kind == "wood" else 3.0)
    for (x, y), h in trees:
        c, r = to_px(x, y)
        d.ellipse([c - 3, r - 3, c + 3, r + 3], fill=float(h))
    H = np.asarray(img, dtype=np.float32)

    road = Image.new("L", (N, N), 0); dr = ImageDraw.Draw(road)
    for line, kind in roads:
        dr.line([to_px(*p) for p in line], fill=1, width=int(ROAD_W[kind] / RES))
    road = np.asarray(road, bool)
    res = Image.new("L", (N, N), 0); dres = ImageDraw.Draw(res)
    for ring in resid:
        if len(ring) >= 3:
            dres.polygon([to_px(*p) for p in ring], fill=1)
    res = np.asarray(res, bool)

    obst = H > 0.5
    clear = distance_transform_edt(~obst) * RES          # metres to nearest obstacle
    rdist = distance_transform_edt(~road) * RES          # metres to nearest road

    # One pad per house: the most back-yard-like open cell near it.
    pads = []
    ok = res & (clear >= PAD_CLEAR_M) & (rdist >= PAD_ROAD_M) & ~road
    for ring, h, src in blds:
        if src != "ms":
            continue
        xy = np.asarray(ring); area = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
                                               - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
        if not 80.0 <= area <= 500.0:                     # house-sized footprints
            continue
        cx, cy = xy.mean(0)
        c0, r0 = (int(v) for v in to_px(cx, cy))
        w = int(PAD_HOUSE_M / RES)
        if not (w <= c0 < N - w and w <= r0 < N - w):
            continue
        win = ok[r0 - w:r0 + w, c0 - w:c0 + w]
        if not win.any():
            continue
        score = np.where(win, rdist[r0 - w:r0 + w, c0 - w:c0 + w], -1)
        rr, cc = np.unravel_index(np.argmax(score), score.shape)
        pads.append(((c0 - w + cc) * RES - HALF + 0.5, (r0 - w + rr) * RES - HALF + 0.5))
    pads = np.unique(np.round(np.asarray(pads, np.float32), 1), axis=0)

    # Roads as flat polylines for the vehicle intruders.
    seg = [np.asarray(l, np.float32) for l, k in roads if k not in ("motorway", "trunk")]
    road_off = np.cumsum([0] + [len(s) for s in seg])
    return dict(H=H, residential=res, road=road, clear=clear.astype(np.float32),
                pads=pads, hub=np.asarray(hub, np.float32), helipad=np.zeros(2, np.float32),
                road_pts=np.concatenate(seg).astype(np.float32), road_off=road_off,
                n_ms=sum(b[2] == "ms" for b in blds), n_osm_h=sum(b[2] == "osm" for b in blds),
                n_trees=len(trees) + len(woods))


def check(m):
    """Every claim in the docstring, verified against the built world."""
    H, pads = m["H"], m["pads"]
    print(f"1. {m['n_ms']:,} Microsoft footprints + {m['n_osm_h']} OSM buildings with "
          f"height tags + {m['n_trees']} OSM tree features")
    b = H[H > 0.5]
    print(f"   obstacle heights: median {np.median(b):.1f} m, p90 {np.percentile(b, 90):.1f} m, "
          f"max {b.max():.1f} m; {100 * (H > 0.5).mean():.1f}% of the ground covered")
    assert 2000 < m["n_ms"] < 20000, "building count is not a suburb's worth"

    ax, ay = to_xy(*ARCHERFIELD_LL)
    far = math.hypot(ax, ay) - math.hypot(HALF, HALF)
    print(f"2. Archerfield (controlled) is {math.hypot(ax, ay)/1000:.1f} km from the helipad; "
          f"the nearest corner of the world is {far/1000:.1f} km from it")
    assert far > 5500, "CASA's 5.5 km controlled-airport rule would bind inside the world"
    print("   -> CASA's 5.5 km controlled-airport rule cannot bind anywhere here")

    ij = ((pads + HALF) / RES).astype(int)
    assert (m["clear"][ij[:, 1], ij[:, 0]] >= PAD_CLEAR_M - 1e-6).all(), "a pad is too close to an obstacle"
    assert m["residential"][ij[:, 1], ij[:, 0]].all(), "a pad is outside residential land"
    assert not m["road"][ij[:, 1], ij[:, 0]].any(), "a pad is on a road"
    near = (np.hypot(pads[:, 0], pads[:, 1]) <= 1400).mean()
    print(f"3. {len(pads):,} back-yard pads, every one >= {PAD_CLEAR_M:.0f} m clear, residential, "
          f"off-road; {100 * near:.0f}% lie within CASA's 1.4 km helipad radius")
    assert len(pads) > 500
    print(f"4. hub (Palms Shopping Centre) at ({m['hub'][0]:+.0f}, {m['hub'][1]:+.0f}) m; "
          f"{len(m['road_off']) - 1} road polylines for vehicles")


def preview(m, path="logan_map.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ext = (-HALF, HALF, -HALF, HALF)
    ax.imshow(np.where(m["residential"], 0.93, 1.0), cmap="Greys_r", vmin=0, vmax=1, extent=ext, origin="lower")
    ax.imshow(np.ma.masked_less(m["H"], 0.5), cmap="viridis", vmin=0, vmax=12, extent=ext, origin="lower")
    ax.imshow(np.ma.masked_equal(m["road"].astype(float), 0), cmap="Greys", vmin=0, vmax=3,
              extent=ext, origin="lower", alpha=0.6)
    ax.add_patch(plt.Circle((0, 0), 1400, fill=False, ls="--", color="#c0392b", lw=1.5))
    ax.scatter(m["pads"][:, 0], m["pads"][:, 1], s=0.4, c="#e08a00", label="back-yard pads")
    ax.scatter([0], [0], marker="P", s=160, c="#c0392b", label="Logan Hospital helipad")
    ax.scatter(*m["hub"], marker="s", s=90, c="#2b7bba", label="hub: Palms Shopping Centre")
    ax.text(990, 990, "CASA 1.4 km\nhelipad radius", color="#c0392b", fontsize=9)
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.set_title("Logan, QLD: 4,582 real buildings (colour = height, m) around Logan Hospital")
    leg = ax.legend(loc="lower left", fontsize=8)
    for hnd, sz in zip(leg.legend_handles, (18, 60, 40)):
        hnd.set_sizes([sz])
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    m = build()
    check(m)
    np.savez_compressed("logan_map.npz", H=m["H"].astype(np.float16), residential=m["residential"],
                        road=m["road"], clear=m["clear"].astype(np.float16), pads=m["pads"],
                        hub=m["hub"], helipad=m["helipad"], road_pts=m["road_pts"],
                        road_off=m["road_off"], half=HALF, res=RES)
    print(f"5. saved logan_map.npz; preview {preview(m)}")
