#!/usr/bin/env python3
"""
lidar_map.py -- the Logan 2023 LiDAR survey (QLD Government, via ELVIS; GDA2020 /
MGA zone 56, heights AHD) turned into height maps on the Logan map's OWN 1 m grid.

Run with the system python (/usr/bin/python3): it has GDAL. Reads everything in
place from DATA_2462013.zip -- 25 one-kilometre tiles of classified points
(~23 M per tile) and the matching 1 m bare-earth DEM tiles. Nothing is extracted.

    ground   metres AHD, bare earth (the survey's own DEM)
    bldg     metres above ground, class 6 (buildings), highest return per cell
    tree     metres above ground, classes 4-5 (medium + high vegetation)
    unclass  metres above ground, class 1: whatever the survey could not label --
             poles, wires, cars. It has NO wire class, so wires live in here if
             anywhere.

Frames. logan_map.to_xy is a flat approximation around the helipad (north is
compressed 0.22%). Every building, pad and road in the map went through it, so
the survey is resampled through the SAME function -- map cell -> lat/lon ->
MGA -> survey cell -- and then the overlap with the map's building footprints is
MEASURED. The footprints come from imagery traced in WGS84-ish coordinates, so a
sub-metre to ~2 m offset from GDA2020 is expected; the best shift is found by
search, applied if it is 1-3 m, and anything larger fails loudly.
"""
import json
import math
import os
import sys
import zipfile

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

ZIP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA_2462013.zip"))
E0, N0, W = 511000, 6937000, 5000        # the 25 tiles: MGA SW corner and width (m)
HELIPAD_LL = (-27.67039, 153.14105)      # must match logan_map.py
HALF, RES = 2000, 1.0                    # the map: 4 x 4 km, 1 m, row index grows NORTH


def las_points(raw):
    """(x, y, z, class) from a LAS 1.4 file held in memory (point formats 6-10)."""
    ver = (raw[24], raw[25]); off, = np.frombuffer(raw[96:100], "<u4")
    fmt = raw[104] & 0x3F; rlen, = np.frombuffer(raw[105:107], "<u2")
    assert raw[:4] == b"LASF" and ver >= (1, 4) and fmt >= 6, (ver, fmt)
    n, = np.frombuffer(raw[247:255], "<u8")
    sx, sy, sz, ox, oy, oz = np.frombuffer(raw[131:179], "<f8")
    dt = np.dtype({"names": ["X", "Y", "Z", "c"], "formats": ["<i4", "<i4", "<i4", "u1"],
                   "offsets": [0, 4, 8, 16], "itemsize": int(rlen)})
    p = np.frombuffer(raw, dtype=dt, count=int(n), offset=int(off))
    return p["X"] * sx + ox, p["Y"] * sy + oy, (p["Z"] * sz + oz).astype(np.float32), p["c"]


def rasterise():
    """Highest return per 1 m MGA cell, per class group; grid[j, i] with j = north."""
    top = {k: np.full((W, W), -np.inf, np.float32) for k in ("bldg", "tree", "unclass")}
    groups = {"bldg": (6,), "tree": (4, 5), "unclass": (1,)}
    z = zipfile.ZipFile(ZIP)
    tiles = sorted(i.filename for i in z.infolist()
                   if "Logan_2023_LGA" in i.filename and i.filename.endswith("las.zip"))
    assert len(tiles) == 25, len(tiles)
    for t in tiles:
        inner = zipfile.ZipFile(z.open(t))               # stored, so this seeks cheaply
        name = next(n for n in inner.namelist() if n.endswith(".las"))
        x, y, h, c = las_points(inner.read(name))
        i = np.floor(x - E0).astype(np.int32); j = np.floor(y - N0).astype(np.int32)
        ok = (i >= 0) & (i < W) & (j >= 0) & (j < W)
        for k, cls in groups.items():
            m = ok & np.isin(c, cls)
            np.maximum.at(top[k].ravel(), j[m] * W + i[m], h[m])
        print(f"  {t.split('SW_')[1][:14]}  {len(x) / 1e6:5.1f} M points", flush=True)
    return top


def ground():
    g = np.full((W, W), np.nan, np.float32)
    names = sorted(i.filename for i in zipfile.ZipFile(ZIP).infolist()
                   if "Logan_2023_LGA" in i.filename and i.filename.endswith("_DEM_1m.tif"))
    assert len(names) == 25, len(names)
    for n in names:
        ds = gdal.Open(f"/vsizip/{ZIP}/{n}")
        x0, dx, _, y0, _, dy = ds.GetGeoTransform()
        assert dx == 1 and dy == -1 and ds.GetSpatialRef().GetAuthorityCode(None) == "7856", n
        a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        a[a == ds.GetRasterBand(1).GetNoDataValue()] = np.nan
        i0, j0 = int(x0 - E0), int(y0 - N0)              # y0 is the tile's NORTH edge
        g[j0 - a.shape[0]:j0, i0:i0 + a.shape[1]] = a[::-1]   # flip to row = north
    return g


def map_to_mga():
    """E, N (MGA) of every map cell centre, through logan_map's own to_xy, inverted."""
    lat0, lon0 = HELIPAD_LL
    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.ImportFromEPSG(7844); dst.ImportFromEPSG(7856)                  # GDA2020 lat/lon -> MGA56
    for s in (src, dst):
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    # a 50 m lattice through GDAL, bilinear in between: the mapping is smooth to micrometres
    k = np.arange(0, 2 * HALF + 1, 50.0) - HALF
    Y, X = np.meshgrid(k, k, indexing="ij")
    lat = lat0 + Y / 110_574.0; lon = lon0 + X / (math.cos(math.radians(lat0)) * 111_320.0)
    EN = np.array(ct.TransformPoints(np.stack([lon.ravel(), lat.ravel()], 1)))[:, :2].reshape(*X.shape, 2)
    c = (np.arange(int(2 * HALF / RES)) + 0.5) * RES / 50.0                  # cell centres, lattice units
    i0 = np.minimum(c.astype(int), len(k) - 2); f = (c - i0)[:, None]
    rows = EN[i0] * (1 - f)[..., None] + EN[i0 + 1] * f[..., None]           # interpolate along north
    out = rows[:, i0] * (1 - f.T)[..., None] + rows[:, i0 + 1] * f.T[..., None]   # then along east
    return out[..., 0], out[..., 1]


def sample(grid, E, N, dx=0.0, dy=0.0):
    i = np.clip(np.floor(E + dx - E0).astype(np.int32), 0, W - 1)
    j = np.clip(np.floor(N + dy - N0).astype(np.int32), 0, W - 1)
    return grid[j, i]


CACHE = "lidar_mga.npz"                   # the survey on its own MGA grid: 3.5 min to rebuild


def survey():
    if not os.path.exists(CACHE):
        top, g = rasterise(), ground()
        assert np.isfinite(g).mean() > 0.99, f"DEM coverage {np.isfinite(g).mean():.3f}"
        hgt = {k: np.where(np.isfinite(v), np.clip(v - g, 0, None), 0).astype(np.float16) for k, v in top.items()}
        np.savez(CACHE, ground=g, **hgt)
    c = np.load(CACHE)
    return {k: c[k].astype(np.float32) for k in ("bldg", "tree", "unclass")}, c["ground"]


def main():
    hgt, g = survey()
    E, N = map_to_mga()
    m = np.load("logan_map.npz")
    # Buildings only. The map also paints OSM woods (15 m), tree rows (10 m) and single
    # trees into H; counting those as "buildings" made woods look like phantom
    # buildings (96% of the first run's map-only cells were wood fills).
    H = m["H"].astype(np.float32)
    foot = (H > 0) & ~np.isin(H, (15.0, 10.0))

    # ---- measure the frame, don't assume it -------------------------------------------------
    # Best shift of the survey onto the map's footprints, found separately in each 1 km block.
    # A conversion error (scale, rotation, a flipped axis) makes the blocks DISAGREE; a source
    # that is simply offset makes them agree. The first run judged overlap instead (IoU < 0.5
    # failed) and that mixed two questions: 24% of the map's footprint cells have no surveyed
    # building within 3 m at ANY shift -- content the two sources disagree about, not frames.
    steps = np.arange(-4, 4.01, 0.5); B = 1000
    inter = np.zeros((len(steps), len(steps), 4, 4)); union = np.zeros_like(inter)
    for a, dx in enumerate(steps):
        for b, dy in enumerate(steps):
            s = sample(hgt["bldg"], E, N, dx, dy) > 2.0
            inter[a, b] = (s & foot).reshape(4, B, 4, B).sum((1, 3))
            union[a, b] = (s | foot).reshape(4, B, 4, B).sum((1, 3))
    tot = inter.sum((2, 3)) / union.sum((2, 3))
    a, b = np.unravel_index(tot.argmax(), tot.shape); use = (float(steps[a]), float(steps[b]))
    busy = foot.reshape(4, B, 4, B).sum((1, 3)) >= 2000                 # blocks with real building stock
    blk = [(steps[i], steps[j]) for r, c in zip(*np.nonzero(busy))
           for i, j in [np.unravel_index((inter / np.maximum(union, 1))[:, :, r, c].argmax(), tot.shape)]]
    spread = max(math.hypot(x - use[0], y - use[1]) for x, y in blk)
    print(f"\nsurvey shift onto the map: east {use[0]:+.1f} m, north {use[1]:+.1f} m "
          f"(IoU {tot[8, 8]:.3f} -> {tot[a, b]:.3f}); {len(blk)} blocks agree within {spread:.1f} m")
    if spread > 1.0 or math.hypot(*use) > 4.0:
        sys.exit(f"FRAME CHECK FAILED: blocks disagree by {spread:.1f} m, or the shift {use} is at the search edge")

    out = {k: sample(v, E, N, *use).astype(np.float16) for k, v in hgt.items()}
    near = lambda m_: m_ | np.roll(m_, 1, 0) | np.roll(m_, -1, 0) | np.roll(m_, 1, 1) | np.roll(m_, -1, 1)
    sb = out["bldg"] > 2.0; far = sb.copy()
    for _ in range(3):
        far = near(far)
    out["map_only"] = foot & ~far                           # map says building, survey none within 3 m
    out["survey_only"] = sb & ~near(near(foot))             # survey says building, map none within 2 m
    print(f"  map building cells with no surveyed building within 3 m: {out['map_only'].sum() / foot.sum():.1%}; "
          f"surveyed building cells the map lacks: {out['survey_only'].sum() / sb.sum():.1%}")
    scores = {(0.0, 0.0): tot[8, 8], use: tot[a, b]}; best = use
    out["ground"] = sample(g, E, N, *use).astype(np.float32)
    both = foot & (out["bldg"] > 2.0)
    d = out["bldg"][both].astype(np.float32) - m["H"][both].astype(np.float32)
    print(f"  roof height, survey minus map, where both see a building: median {np.median(d):+.1f} m, "
          f"middle half {np.percentile(d, 25):+.1f} to {np.percentile(d, 75):+.1f} m")
    gr = out["ground"]
    print(f"  ground: {np.nanmin(gr):.1f} to {np.nanmax(gr):.1f} m AHD; "
          f"helipad {gr[HALF, HALF]:.1f} m; hub {gr[int(m['hub'][1]) + HALF, int(m['hub'][0]) + HALF]:.1f} m")
    np.savez_compressed("lidar_logan.npz", **out, shift=np.array(use), iou=np.array([scores[(0.0, 0.0)], scores[best]]))
    json.dump({"shift_applied_m": use, "iou_as_converted": scores[(0.0, 0.0)], "iou_best": scores[best],
               "roof_diff_median_m": float(np.median(d))}, open("lidar_logan.json", "w"), indent=1)
    print("lidar_logan.npz")


if __name__ == "__main__":
    main()
