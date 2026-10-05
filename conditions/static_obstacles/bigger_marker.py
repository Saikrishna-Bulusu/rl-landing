#!/usr/bin/env python3
"""
bigger_marker.py -- does the marker-size result survive a measured detector?

THE ORIGINAL CLAIM. On identical obstacle layouts a 1.20 m pad crashed 44.8% of
the time and a 2.40 m one 0.0%, because detection altitude is linear in marker
size and a 1.20 m marker cannot be decoded from above a 25 m tree.

THE REASON TO DOUBT IT. That rested on MIN_PX = 34 as a CLIFF -- invisible
below, perfect above. Flying the pad in PX4/Gazebo showed the 1.20 m marker
decoding at 14 px, and re-flying measure_detect.py from 60 m instead of 26 m
(the old floor, which is why nothing below 25 px had ever been sampled) gave
the real shape: a ramp from 0% at 10-15 px to 100% by 35 px, reaching 50% at
27.2 m for a 1.20 m marker where the cliff said a flat 19.1 m.

So the cliff was PESSIMISTIC about the small marker, and the effect it produced
should shrink. Run both detectors over the same layouts and see by how much.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "archive", "sigma"))   # part 1: rule_agent, sensing_compare
import rule_agent as R
from sensing_compare import Base


def score(pad_size, detect, make_agent, n=400, seed0=7_500_000):
    cls = type("E", (Base,), {"PAD_SIZE": float(pad_size), "DETECT": detect})
    env = cls(pool=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "fast_sim", "sigma_pool.npz"), arm="sigma", seed=0, t_max=35,
              rho=0.707, sigma_scale=3.5, sense_mode="sectors")
    rew, ok, cr = [], [], []
    for i in range(n):
        o, _ = env.reset(seed=seed0 + i)
        ag = make_agent(env); ag.reset(); tot = 0.0
        while True:
            o, r, t, tr, info = env.step(np.clip(ag.act(o), -1, 1)); tot += r
            if t or tr: break
        rew.append(tot); ok.append(info["success"])
        cr.append(info.get("outcome") == "crash")
    return (float(np.mean(rew)), 100*np.mean(ok), 100*np.mean(cr), env.h_detect)


if __name__ == "__main__":
    rule = lambda e: R.SigmaRuleMean(e, 0.5)
    out = {}
    for detect in ("threshold", "curve"):
        lab = ("MIN_PX cliff at 34 px" if detect == "threshold"
               else "measured ramp, detect_curve_60m.json")
        print(f"\n  {detect.upper():<10} ({lab})")
        print(f"    {'pad':>6} {'reliable from':>14} {'reward':>8} {'delivered':>10} {'crash':>7}")
        for ps in (1.20, 1.80, 2.40):
            rw, ok, cr, hd = score(ps, detect, rule)
            out[(detect, ps)] = (rw, ok, cr)
            print(f"    {ps:5.2f} m {hd:12.1f} m {rw:+8.1f} {ok:9.1f}% {cr:6.1f}%")

    print("\n  THE EFFECT OF GOING FROM 1.20 m TO 2.40 m")
    for detect in ("threshold", "curve"):
        a, b = out[(detect, 1.20)], out[(detect, 2.40)]
        print(f"    {detect:<10} reward {b[0]-a[0]:+8.1f}   delivered "
              f"{b[1]-a[1]:+6.1f} pp   crash {b[2]-a[2]:+6.1f} pp")
    ta = out[("threshold", 2.40)][2] - out[("threshold", 1.20)][2]
    ca = out[("curve", 2.40)][2] - out[("curve", 1.20)][2]
    print(f"\n    the cliff credits the bigger marker with {abs(ta):.1f} pp of crash")
    print(f"    reduction; the measured detector credits it with {abs(ca):.1f} pp.")
