#!/usr/bin/env python3
"""
forest_spec.py -- the obstacle geometry, in one place, with no dependencies.

THREE THINGS need these numbers: the point-mass env (sensing_compare.Base), the
world-file generator (make_trees.py), and the runtime spawner (forest.py). The
last two run under the SYSTEM python, which has Gazebo's bindings but no torch,
so they cannot reach the numbers through the env class. Retyping them is how
the simulator and the model quietly stop being the same experiment.

So they live here, imported by all three. numpy is not needed either.
"""

N_OBST = (10, 13)        # trees per garden
OBST_H = (15.0, 25.0)    # height, m -- taller than the marker's detection ceiling
OBST_R = (0.8, 1.8)      # TRUNK radius, m
TRUNK_R = (0.8, 1.8)     # the part that reaches the ground
CANOPY_R = (2.5, 4.5)    # the part that overhangs
CANOPY_FRAC = 0.55       # canopy occupies the top 45%
PAD_CLEAR = 6.0          # no TRUNK closer than this to the pad
OBST_SPAN = 22.0         # scattered within this of the pad


def canopy_r(rad):
    """Canopy radius for a trunk of this radius. Linear between the ranges."""
    lo, hi = CANOPY_R
    t = (rad - TRUNK_R[0]) / max(TRUNK_R[1] - TRUNK_R[0], 1e-6)
    return lo + t * (hi - lo)
