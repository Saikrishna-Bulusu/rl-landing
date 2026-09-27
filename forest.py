#!/usr/bin/env python3
"""
forest.py -- replant the trees at runtime, without restarting the simulator.

WHY. A forest baked into the world SDF is loaded once, so every mission in a
batch flies the SAME trees. Twenty seeds against one layout measures that
layout precisely; it says nothing about whether the result holds for a
different arrangement of trees. Varying the forest per mission is what buys
generality, and doing it through Gazebo's create/remove services costs about a
second instead of the 45 s a full SITL restart costs.

The geometry comes from make_trees.py, which takes it from the env class, so
all three -- point-mass env, world file, and runtime spawn -- are one source.
"""
import time

import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.scene_pb2 import Scene
from gz.transport13 import Node

import make_trees as MT

WORLD = "aruco"
_node = Node()


def _sdf(i, p, r, h):
    return "<sdf version='1.9'>" + MT.TREE.format(
        i=i, x=MT.PAD_ENU[0] + p[0], y=MT.PAD_ENU[1] + p[1],
        r=r, R=MT.canopy_r(r),
        th=MT.Base.CANOPY_FRAC * h, tz=0.5 * MT.Base.CANOPY_FRAC * h,
        ch=(1 - MT.Base.CANOPY_FRAC) * h,
        cz=MT.Base.CANOPY_FRAC * h + 0.5 * (1 - MT.Base.CANOPY_FRAC) * h,
        tc="", cc="") + "</sdf>"


def standing(tries=4):
    """
    Tree models actually present in the world.

    The remove service answers True whether or not the entity existed, so
    counting successful removals reports 40 removals from a world holding ten
    trees. Ask the scene instead -- otherwise "the forest was replanted" is
    unfalsifiable, which is the same trap as a crash detector that never fires.

    RETRIES, because a single 2 s request is not enough. Gazebo is rendering a
    1280x960 camera at 30 Hz and the caller is subscribed to it, so scene/info
    can miss that window under load -- it did, on the third replant of a batch,
    killing two 20-trial runs at trial 3. Back off and ask again rather than
    treating one slow reply as a broken world.
    """
    last = None
    for k in range(tries):
        ok, scene = _node.request(f"/world/{WORLD}/scene/info", Empty(), Empty,
                                  Scene, 2000 * (k + 1))
        if ok:
            return sorted(m.name for m in scene.model
                          if m.name.startswith("tree_"))
        last = 2000 * (k + 1)
        time.sleep(0.5)
    raise RuntimeError(f"gazebo would not report its scene after {tries} "
                       f"attempts (last waited {last} ms)")


def _settle(want, timeout=10.0):
    """
    Wait until the scene reports `want` trees.

    create/remove return as soon as the request is accepted, not once the world
    has ticked, so reading the scene immediately after reports the state before
    the change. A fixed sleep would work most of the time, which is the worst
    kind of working -- poll the scene and fail loudly if it never agrees.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        n = len(standing())
        if n == want:
            return n
        time.sleep(0.2)
    raise RuntimeError(f"scene settled at {len(standing())} trees, wanted {want}")


def clear(n=40):
    """Remove every tree_* model, then verify against the scene."""
    for i in range(n):
        req = Entity(); req.name = f"tree_{i}"; req.type = Entity.MODEL
        _node.request(f"/world/{WORLD}/remove", req, Entity, Boolean, 300)
    _settle(0)
    return True


def plant(seed):
    """Place a fresh forest and return it in the two frames the code needs."""
    trees = MT.place(seed)
    out = []
    for i, (p, r, h) in enumerate(trees):
        req = EntityFactory(); req.sdf = _sdf(i, p, r, h); req.allow_renaming = False
        ok, rep = _node.request(f"/world/{WORLD}/create", req, EntityFactory,
                                Boolean, 800)
        if not (ok and rep.data):
            raise RuntimeError(f"gazebo refused to spawn tree_{i}")
        out.append(dict(enu=[MT.PAD_ENU[0] + p[0], MT.PAD_ENU[1] + p[1]],
                        ned=[p[1], MT.PAD_ENU[0] + p[0]],
                        trunk_r=float(r), canopy_r=float(MT.canopy_r(r)),
                        h=float(h), canopy_frac=MT.Base.CANOPY_FRAC))
    _settle(len(out))
    return out


def replant(seed):
    """Clear and replant; both steps verified against the scene graph."""
    clear()
    return plant(seed)


if __name__ == "__main__":
    print("1. clearing whatever is there")
    clear(); print("   scene now holds 0 trees")
    for s in (11, 12):
        t = plant(s)
        worst = min(np.linalg.norm(np.array(x["ned"]) - np.array([0.0, 120.0]))
                    - x["canopy_r"] for x in t)
        print(f"2. seed {s}: planted {len(t)} trees, "
              f"heights {min(x['h'] for x in t):.1f}-{max(x['h'] for x in t):.1f} m, "
              f"clear column {worst:.2f} m")
        assert worst > 0, "a canopy covers the pad"
        print(f"   scene confirms {len(standing())} tree models present")
        clear()
        print(f"   scene confirms {len(standing())} after clearing")
    print("3. spawn and removal both verified against the scene graph")
