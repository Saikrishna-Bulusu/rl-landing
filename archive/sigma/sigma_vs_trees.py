#!/usr/bin/env python3
"""
sigma_vs_trees.py -- the experiment the clear-field study could not run.

Every earlier measurement scored DELIVERIES on an empty field, and there
sharper localization bought nothing: the aircraft searched a little less and
arrived anyway. So the matcher work looked like it had no payoff.

With trees the miss is not merely slower, it is aimed. The pad sits in a clear
column about 4.7 m across; the aircraft descends over the pad it BELIEVES in,
and when that point is under a canopy it crashes twice as often. Localization
error is therefore priced in airframes, not in seconds -- an outcome no clear-
field experiment could ever have shown.

Sweep the error and watch the crash rate. Same layouts throughout, so nothing
but the localization changes.
"""
import numpy as np
import sensing_compare as C
import sensing_headroom as S
import rule_agent as R


def run(scale, n=400):
    env = C.make("sectors"); env.sigma_scale = scale
    rew, ok, crash, blocked = [], [], [], []
    for i in range(n):
        o, _ = env.reset(seed=7_500_000 + i)
        ag = R.SigmaRuleMean(env, 0.5); ag.reset(); tot = 0.0
        while True:
            o, r, t, tr, info = env.step(np.clip(ag.act(o), -1, 1)); tot += r
            if t or tr: break
        rew.append(tot); ok.append(info["success"])
        crash.append(info.get("outcome") == "crash")
        blocked.append(env.descent_blocked(np.asarray(env._fix_mean(), float)[:2]))
    return (float(np.mean(rew)), 100*np.mean(ok), 100*np.mean(crash),
            100*np.mean(blocked))


def main():
    print("localization error swept against the SAME tree layouts\n")
    print("  sigma   reward   delivered   crash   aims under a canopy")
    base = None
    for s in (3.5, 2.5, 1.75, 1.0, 0.5):
        rew, ok, cr, bl = run(s)
        if base is None: base = (rew, cr)
        print(f"  x{s:<4} {rew:+8.1f}     {ok:5.1f}%   {cr:5.1f}%      {bl:5.1f}%")
    print(f"\n  halving the error from x3.5 to x1.75 is worth the difference")
    print(f"  between rows 1 and 3 above -- compare it against the +6.4 reward")
    print(f"  that stereo-grade obstacle sensing bought.")


if __name__ == "__main__":
    main()
