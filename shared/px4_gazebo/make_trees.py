#!/usr/bin/env python3
"""
make_trees.py -- put the obstacle model into the Gazebo world.

WHY. Every crash number in this project comes from the point-mass env. The
Gazebo world is a flat grey plane, so the one claim that actually motivates a
bigger marker -- that it stops you flying into trees -- has never been tested
on real firmware. This builds the same forest in Gazebo.

SAME CONSTANTS, NOT SIMILAR ONES. The geometry is imported from the env class
rather than retyped, so the two cannot drift apart. A tree is a trunk cylinder
under a wider canopy cylinder occupying the top 45%, which is the shape the
env's _crashed() tests against.

VISUAL ONLY, NO COLLISION. Deliberate, and a real limitation.
  - With collision geometry a brush against a canopy tumbles the airframe, and
    the next mission in the batch starts from wherever the wreck came to rest.
    That breaks the seed pairing that every comparison here depends on.
  - The crash is instead detected geometrically in px4_mission.py using the
    SAME test as the env, which makes the two directly comparable.
  - What Gazebo adds that the point-mass model cannot: the canopies really do
    occlude the camera, so a marker behind a tree is genuinely not decoded.
To fly the physical version instead, set COLLIDE = True below.
"""
import argparse
import json
import os

import numpy as np

import forest_spec as Base

HERE = os.path.dirname(os.path.abspath(__file__))

WORLD = ("/home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/worlds/"
         "aruco.sdf")
PAD_ENU = (120.0, 0.0)          # matches the arucotag include pose
COLLIDE = False

TREE = """    <model name='tree_{i}'>
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <link name='trunk'>
        <pose>0 0 {tz:.3f} 0 0 0</pose>
        <visual name='v'>
          <geometry><cylinder><radius>{r:.3f}</radius><length>{th:.3f}</length></cylinder></geometry>
          <material><ambient>0.25 0.17 0.09 1</ambient><diffuse>0.35 0.24 0.13 1</diffuse></material>
        </visual>{tc}
      </link>
      <link name='canopy'>
        <pose>0 0 {cz:.3f} 0 0 0</pose>
        <visual name='v'>
          <geometry><cylinder><radius>{R:.3f}</radius><length>{ch:.3f}</length></cylinder></geometry>
          <material><ambient>0.09 0.20 0.07 1</ambient><diffuse>0.13 0.31 0.11 1</diffuse></material>
        </visual>{cc}
      </link>
    </model>
"""
COL = ("\n        <collision name='c'><geometry><cylinder>"
       "<radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder>"
       "</geometry></collision>")


def place(seed):
    """The env's own placement rule, so the layouts are the same problem."""
    rng = np.random.default_rng(seed)
    pad = np.zeros(2)
    n = int(rng.integers(Base.N_OBST[0], Base.N_OBST[1] + 1))
    out = []
    for _ in range(n):
        for _try in range(20):
            th = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(Base.PAD_CLEAR, Base.OBST_SPAN)
            p = pad + d * np.array([np.cos(th), np.sin(th)])
            rad = rng.uniform(*Base.OBST_R)
            if np.linalg.norm(p - pad) > Base.PAD_CLEAR + rad:
                out.append((p, float(rad), float(rng.uniform(*Base.OBST_H))))
                break
    return out


canopy_r = Base.canopy_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    trees = place(a.seed)
    print(f"1. {len(trees)} trees, heights "
          f"{min(t[2] for t in trees):.1f}-{max(t[2] for t in trees):.1f} m, "
          f"within {Base.OBST_SPAN:.0f} m of the pad")

    # THE PAD MUST NOT BE UNDER A CANOPY. Sai's requirement, and the placement
    # rule is supposed to guarantee it -- check rather than assume.
    worst = min(np.linalg.norm(p) - canopy_r(r) for p, r, h in trees)
    print(f"2. nearest canopy edge is {worst:.2f} m from the pad centre")
    assert worst > 0, "a canopy covers the pad; nobody puts a landing pad under a tree"

    # And the descent corridor must be narrower than the localization error,
    # or there is no decision to study.
    print(f"   clear column above the pad: {worst:.1f} m radius")

    sdf = "".join(
        TREE.format(
            i=i, x=PAD_ENU[0] + p[0], y=PAD_ENU[1] + p[1],
            r=r, R=canopy_r(r),
            th=Base.CANOPY_FRAC * h, tz=0.5 * Base.CANOPY_FRAC * h,
            ch=(1 - Base.CANOPY_FRAC) * h,
            cz=Base.CANOPY_FRAC * h + 0.5 * (1 - Base.CANOPY_FRAC) * h,
            tc=COL.format(r=r, h=Base.CANOPY_FRAC * h) if COLLIDE else "",
            cc=COL.format(r=canopy_r(r), h=(1 - Base.CANOPY_FRAC) * h) if COLLIDE else "")
        for i, (p, r, h) in enumerate(trees))

    # ENU here, NED in the mission: the SDF pose "120 0 0" is 120 m EAST, which
    # px4_mission.py calls PAD_XY = (0, 120). Store both and let the mission
    # assert the mapping rather than trusting this comment.
    meta = dict(
        seed=a.seed, pad_enu=list(PAD_ENU), collide=COLLIDE,
        trees=[dict(enu=[PAD_ENU[0] + p[0], PAD_ENU[1] + p[1]],
                    ned=[p[1], PAD_ENU[0] + p[0]],
                    trunk_r=r, canopy_r=canopy_r(r), h=h,
                    canopy_frac=Base.CANOPY_FRAC)
               for p, r, h in trees])

    if a.write:
        w = open(WORLD).read()
        w = _strip(w)
        w = w.replace("  </world>", sdf + "  </world>")
        open(WORLD, "w").write(w)
        json.dump(meta, open(f"{HERE}/trees.json", "w"), indent=1)
        print(f"3. wrote {len(trees)} trees into the world and trees.json")
    else:
        json.dump(meta, open(f"{HERE}/trees.json", "w"), indent=1)
        print("3. wrote trees.json   (dry run -- pass --write for the world)")


def _strip(w):
    """Remove any trees already in the world so this is idempotent."""
    while "<model name='tree_" in w:
        i = w.index("<model name='tree_")
        j = w.index("</model>", i) + len("</model>\n")
        w = w[:w.rindex("\n", 0, i) + 1] + w[j:]
    return w


if __name__ == "__main__":
    main()
