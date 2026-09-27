#!/usr/bin/env python3
"""
crash_geometry.py -- WHY the rule hits trees, before deciding what to buy.

Two candidate causes, and they call for opposite purchases:

  A. SENSING. It cannot see the canopy in time. -> a better depth sensor.
  B. LOCALIZATION. It sees fine, but descends over the pad it BELIEVES in,
     and that point is under a canopy even though the real pad is clear.
     -> a better matcher. No depth camera fixes this.

Distinguishable from the layout alone. The pad sits in a clear column whose
radius is set by how far the nearest canopy edge is; if the localization error
exceeds that radius, a perfect-sensing aircraft still descends into a tree.
"""
import numpy as np
import sensing_compare as C
import rule_agent as R


def clear_radius(env):
    """How far from the pad you may descend and still be in open sky."""
    if not env.obst:
        return np.inf
    return min(np.linalg.norm(p - env.pad) - env._canopy_r(rad)
               for p, rad, h in env.obst)


def main():
    env = C.make("sectors")
    col, err, blocked_belief, out = [], [], [], []
    for i in range(400):
        o, _ = env.reset(seed=7_500_000 + i)
        rule = R.SigmaRuleMean(env, 0.5); rule.reset()
        tot = 0.0
        while True:
            o, r, t, tr, info = env.step(np.clip(rule.act(o), -1, 1)); tot += r
            if t or tr: break
        col.append(clear_radius(env))
        # where the rule THOUGHT the pad was, at the end. SigmaRuleMean aims
        # at the average of every fix, which the env keeps as _fix_mean().
        bel = np.asarray(env._fix_mean(), float)[:2]
        err.append(np.linalg.norm(bel - env.pad))
        blocked_belief.append(env.descent_blocked(bel))
        out.append(info.get("outcome"))

    col, err = np.array(col), np.array(err)
    out = np.array(out); bb = np.array(blocked_belief)
    crash = out == "crash"

    print(f"clear column above the pad: radius {np.median(col):.1f} m median, "
          f"{np.percentile(col,10):.1f} m at the 10th pct")
    print(f"localization error at touchdown: {np.median(err):.1f} m median, "
          f"{np.percentile(err,90):.1f} m at the 90th pct")
    print(f"error exceeds the column in {100*np.mean(err > col):.0f}% of episodes\n")

    print(f"the point it descends on is under a canopy: {100*bb.mean():.0f}% of episodes")
    print(f"  of those, {100*crash[bb].mean():.0f}% crash")
    print(f"  of the rest, {100*crash[~bb].mean():.0f}% crash\n")

    print("crash rate split by whether the BELIEF was descendable:")
    print(f"  belief clear   n={np.sum(~bb):3d}   crash {100*crash[~bb].mean():5.1f}%")
    print(f"  belief blocked n={np.sum(bb):3d}   crash {100*crash[bb].mean():5.1f}%")
    lift = 100*crash[bb].mean() - 100*crash[~bb].mean()
    print(f"\n  -> descending on a blocked belief raises the crash rate by {lift:+.0f} pp")
    if lift > 20:
        print("     This is cause B. The aircraft aims at the wrong point;")
        print("     sharper depth sensing does not move the aim point.")
    else:
        print("     Belief quality does not explain the crashes: cause A.")


if __name__ == "__main__":
    main()
