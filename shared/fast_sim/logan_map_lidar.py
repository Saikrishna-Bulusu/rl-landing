#!/usr/bin/env python3
"""
logan_map_lidar.py -- the Logan map rebuilt from the 2023 LiDAR survey
(lidar_logan.npz, from lidar_map.py). Writes logan_map_lidar.npz with the keys
urban_env reads, next to the old map (which stays, so earlier results reproduce).

What changes and why:
  obstacles  every surveyed building, tree and unlabelled tall thing (poles,
             wires: the survey has no wire class) at its measured height. The old
             map's Microsoft footprints stopped at a tile edge near y = -1400 m,
             and OpenStreetMap knew 39 single trees in 16 km^2.
  pads       one per surveyed house (80-500 m^2 of roof, in residential land),
             moved within its own yard to the spot FURTHEST from the road among
             those with no obstacle over 2 m within PAD_CLEAR_M, clear ground
             (nothing over 0.3 m within 1.5 m) under the mat, and 12 m back.
             Houses with no such spot get no deliveries -- as a real service
             would refuse them. Old rule: 4 m clear of OSM obstacles, which
             barely included trees; 58% of those pads had a tree over 5 m within
             3 m of them.
  PAD_CLEAR_M = 2.5 m: in the forest study every crash happened where the clear
             column above the pad was under 4.2 m WIDE (2.1 m radius); 0.4 m more
             covers the marker-fix error on descent. Sensitivity is printed.
  ground     kept for Gazebo; the fast env flies height above ground (terrain
             following), so the 41 m of relief changes nothing there.
Roads, residential land, hub and helipad are the old map's (OpenStreetMap).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label, maximum_filter

HALF, RES = 2000, 1.0
PAD_CLEAR_M, PAD_ROAD_M, PAD_HOUSE_M, OBST_M = 2.5, 12.0, 25.0, 2.0
PAD_GROUND_R, GROUND_M = 1.5, 0.3     # the mat's own patch: nothing over 0.3 m within 1.5 m
HOUSE_M2 = (80.0, 500.0)


RING = np.ones((3, 3), bool); RING[1, 1] = False


def despike(a, tol=5.0):
    """A cell may stand at most tol above the highest of its 8 neighbours. Each cell
    holds the single HIGHEST return, so one stray return the survey did not flag as
    noise becomes a 1 m pillar. This lowers 6,231 unlabelled cells (730 of them over
    20 m, one 77 m above its neighbours), 2,592 tree cells and 53 building cells.
    Crowns, roofs and wires (a line has neighbours along it) keep their height --
    which is why the two crane jibs survive it and need their own handling below.
    (The first diagnosis blamed "746 isolated spikes" for the cruise crashes; that
    count used a broken isolation test and was mostly the crane jibs.)"""
    m8 = maximum_filter(a, footprint=RING, mode="nearest")
    return np.where(a - m8 > tol, m8, a)


def layers():
    """The survey's building / tree / unlabelled heights, cleaned: despiked, cranes out.
    Returns (layers, crane mask, ground). Shared with the Gazebo world builder."""
    L = np.load("lidar_logan.npz")
    lay = {k: despike(np.nan_to_num(L[k].astype(np.float32))) for k in ("bldg", "tree", "unclass")}
    # Two tower cranes: straight 100 m jibs crossing at ~80 m over the hospital extension
    # being built in 2023, 70 m from the helipad. Real in the survey, gone today; left out
    # of the base world (21 of 1,024 rule flights flew into them at cruise) and kept as
    # `crane` for a surprise-obstacle test. Nothing else unlabelled is this big and tall.
    tall, _ = label(lay["unclass"] > 45.0)
    crane = np.isin(tall, np.nonzero(np.bincount(tall.ravel())[1:] > 100)[0] + 1)
    assert 500 < crane.sum() < 1500, crane.sum()
    lay["unclass"][crane] = 0.0
    return lay, crane, L["ground"]


def build():
    old = np.load("logan_map.npz")
    lay, crane, ground = layers()
    unc = np.where(lay["unclass"] > 3.0, lay["unclass"], 0.0)       # above car height: poles, wires
    H = np.maximum.reduce([lay["bldg"], lay["tree"], unc]).astype(np.float32)
    res, road = old["residential"], old["road"]
    clear = distance_transform_edt(H <= OBST_M) * RES                 # metres to the nearest obstacle > 2 m
    # The mat itself lies on lawn or paving: nothing over GROUND_M within PAD_GROUND_R.
    # The first rule allowed up to 2 m ON the pad, so 10% of pads sat on a shrub (the
    # survey's 0.3-2 m vegetation): the camera lost the marker behind it on every descent
    # and those flights bobbed over their pad until the clock ran out -- 100% of the
    # "timeout, other" flights, none of the delivered ones.
    ground_ok = distance_transform_edt(H <= GROUND_M) * RES >= PAD_GROUND_R
    rdist = distance_transform_edt(~road) * RES

    lab, n = label(lay["bldg"] > 2.5)
    # Its own yard: ground closer to this house than to any other building (no cadastre
    # here). Without it the "clearest spot within 25 m" was often a clearing several
    # houses shared: 1,480 houses qualified but only 975 distinct pads were left.
    _, near = distance_transform_edt(lab == 0, return_indices=True)
    owner = lab[near[0], near[1]]; del near
    w = int(PAD_HOUSE_M / RES)
    best, houses = [], 0
    for k, sl in enumerate(find_objects(lab), start=1):
        area = (lab[sl] == k).sum() * RES * RES
        r0, c0 = [int((s.start + s.stop) / 2) for s in sl]
        if not (HOUSE_M2[0] <= area <= HOUSE_M2[1] and res[r0, c0]) or not (w <= r0 < 4000 - w and w <= c0 < 4000 - w):
            continue
        houses += 1
        win = np.s_[r0 - w:r0 + w, c0 - w:c0 + w]
        ok = res[win] & ~road[win] & (rdist[win] >= PAD_ROAD_M) & (owner[win] == k)
        # Clear enough, THEN as far from the road as the yard allows. "Clearest spot"
        # alone chose front lawns (trees and sheds live in back yards): median 31 m
        # from a road against the old pads' 44 m, a quarter within 20 m, and the
        # rule's 30 m-from-vehicles violations went 0.6% -> 3.4%.
        valid = ok & (clear[win] >= PAD_CLEAR_M) & ground_ok[win]
        score = np.where(valid, rdist[win] + 1e-3 * clear[win], -1.0)
        rr, cc = np.unravel_index(score.argmax(), score.shape)
        best.append((clear[win][ok & ground_ok[win]].max() if (ok & ground_ok[win]).any() else 0.0,
                     valid.any(), (c0 - w + cc) * RES - HALF + 0.5, (r0 - w + rr) * RES - HALF + 0.5))
    best = np.array(best, np.float32)
    print(f"surveyed houses (80-500 m^2 of roof, residential land): {houses:,}")
    for r in (2.0, 2.5, 3.0, 4.0, 5.0):
        print(f"  with clear ground and {r:.1f} m clear of anything over {OBST_M:.0f} m: {(best[:, 0] >= r).mean():6.1%}")
    keep = best[best[:, 1] > 0]
    pads = np.unique(np.round(keep[:, 2:], 1), axis=0)
    return dict(H=H, residential=res, road=road, clear=clear.astype(np.float32), pads=pads,
                hub=old["hub"], helipad=old["helipad"], road_pts=old["road_pts"], road_off=old["road_off"],
                half=old["half"], res=old["res"], ground=ground, houses=np.int64(houses), crane=crane)


def check(m):
    """Fails loudly if a pad is not what the rule says it is."""
    px = (m["pads"] + HALF).astype(int)                                 # (col, row)
    assert (m["clear"][px[:, 1], px[:, 0]] >= PAD_CLEAR_M).all(), "a pad without its clear radius"
    assert m["residential"][px[:, 1], px[:, 0]].all() and not m["road"][px[:, 1], px[:, 0]].any()
    assert np.isfinite(m["H"]).all() and (m["H"] >= 0).all()
    ground = distance_transform_edt(m["H"] <= GROUND_M) * RES
    assert (ground[px[:, 1], px[:, 0]] >= PAD_GROUND_R).all(), "a pad on something over 0.3 m"
    old = np.load("logan_map.npz")
    assert np.allclose(m["hub"], old["hub"]) and np.allclose(m["helipad"], old["helipad"])
    n_south = (m["pads"][:, 1] < -1400).sum()
    print(f"pads: {len(m['pads']):,} (old map {len(old['pads']):,}); in the strip the old map lost: {n_south}; "
          f"median clear radius {np.median(m['clear'][px[:, 1], px[:, 0]]):.1f} m")
    return True


def preview(m, path="logan_map_lidar.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
    ext = (-HALF, HALF, -HALF, HALF)
    ax.imshow(np.where(m["road"], 1.0, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=4)
    im = ax.imshow(np.where(m["H"] > OBST_M, m["H"], np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=25)
    ax.scatter(*m["pads"].T, s=5, c="#e08a00", lw=0, label=f"pads ({len(m['pads']):,}): {PAD_CLEAR_M} m clear")
    ax.scatter(*m["hub"], s=90, marker="s", c="#2b7bba", label="hub")
    ax.scatter(*m["helipad"], s=120, marker="P", c="#c0392b", label="Logan Hospital helipad")
    ax.set_title("Logan rebuilt from the 2023 LiDAR survey (colour = obstacle height, m)")
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)"); ax.legend(loc="lower left", markerscale=2)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=130); print(path)


if __name__ == "__main__":
    m = build()
    check(m)
    np.savez_compressed("logan_map_lidar.npz", **m)
    preview(m)
