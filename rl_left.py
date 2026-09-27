#!/usr/bin/env python3
"""
rl_left.py -- is there anything left for a learned policy to decide?

THE CASE FOR RL IN THIS PROJECT RESTED ON OBSTACLES. On a clear field the
winning agent was a one-line rule -- aim at the running mean of the fixes and
descend -- and twelve trained policies lost to it by 3.58 pp. Obstacles were
the exception: a decider told the tree layout beat the best avoidance rule by
+15.1 reward, and that gap was the argument for learning.

Then a larger marker dissolved it. On real firmware, in twenty different
forests, a 1.80 m coarse marker delivered 20/20 with that same one-line rule.

So ask the question properly, in the two regimes that actually differ:

  CAN REPAINT     1.80 m marker. If the headroom is ~0 here, there is nothing
                  for a policy to win and that is a clean negative result.

  CANNOT REPAINT  1.20 m marker -- existing infrastructure, someone else's
                  pad. This is where 15 of 20 firmware missions failed. If
                  headroom survives here, THIS is the decision worth learning,
                  and the honest framing of the RL case changes from "avoid
                  trees" to "cope with a marker you were not allowed to size".

Headroom is measured against an oracle, through the guard that refuses to
return a number when the oracle scores below the rule it is supposed to bound.
"""
import numpy as np

import rule_agent as R
import sensing_headroom as S
from sensing_compare import Base


def env_for(pad_size, detect="curve", **kw):
    cls = type("E", (Base,), {"PAD_SIZE": float(pad_size), "DETECT": detect})
    return cls(pool="sigma_pool.npz", arm="sigma", seed=0, t_max=45,
               rho=0.707, sigma_scale=3.5, sense_mode="sectors", **kw)


def main():
    n = 300
    print(f"  headroom over the one-line rule, {n} matched episodes, "
          f"detection sampled from the measured curve\n")
    print(f"    {'marker':>8} {'':>12} {'reward':>9} {'delivered':>10} {'crash':>7}")
    out = {}
    for ps in (1.80, 1.20):
        env = env_for(ps)
        # strict=False only because a SATURATED regime is exactly what is
        # being tested for: where nothing is left to win, the oracle's extra
        # caution can put it a hair below the rule. A genuinely broken teacher
        # still shows up -- as a large negative headroom, reported not hidden.
        h = S.validated_headroom(
            env, lambda: R.SigmaRuleMean(env, 0.5),
            lambda: S.ObstacleTeacher(env), n, strict=False)
        out[ps] = h
        for lab, k in (("rule", "rule"), ("oracle", "teacher")):
            v = h[k]
            print(f"    {ps:7.2f}m {lab:>12} {v[0]:+9.1f} {v[1]:9.1f}% {v[2]:6.1f}%")
        tag = ("   (a tie: the oracle wins nothing)" if h["tie"] else "")
        if h.get("guard_failed"):
            tag = "   (oracle sits BELOW the rule: its caution costs a step)"
        print(f"    {'':>8} {'HEADROOM':>12} {h['headroom']:+9.1f}"
              f"  +/- {2*h['se']:.1f}{tag}\n")

    a, b = out[1.80]["headroom"], out[1.20]["headroom"]
    print(f"  can repaint    (1.80 m): {a:+.1f} reward of headroom")
    print(f"  cannot repaint (1.20 m): {b:+.1f} reward of headroom")
    print()
    if a < 0.15 * abs(b) or a < 3.0:
        print("  -> With a marker you are allowed to size, an oracle that knows")
        print("     the whole layout barely beats a one-line rule. There is no")
        print("     decision left for a policy to learn, and no amount of")
        print("     training buys what is not there.")
    else:
        print("  -> Headroom survives even at 1.80 m; the marker did not close it.")
    if b > 3.0 and b > 1.5 * a:
        print("  -> The remaining decision lives in the CANNOT-REPAINT case. That")
        print("     is the honest RL problem: not avoiding trees, but coping with")
        print("     a marker too small to see from above them.")


def learnable():
    """
    Headroom is necessary for RL to be worth it. It is not sufficient.

    The earlier imitation diagnostic found the oracle's advantage was
    INFORMATION, not optimisation -- a network could not even FIT the oracle's
    actions, because the same observation maps to different ones. Repeat that
    test in the only regime where headroom survives. Clone the oracle by
    DAgger, so distribution shift cannot be blamed, and clone the rule as a
    control: if the control fails too, the setup is broken and nothing can be
    concluded.
    """
    import sensing_compare as C
    import torch
    torch.manual_seed(0); np.random.seed(0)

    env = env_for(1.20)
    h = S.validated_headroom(env, lambda: R.SigmaRuleMean(env, 0.5),
                             lambda: S.ObstacleTeacher(env), 300, strict=False)
    print(f"\n  CANNOT-REPAINT REGIME: is the {h['headroom']:+.1f} reachable?\n")
    print(f"    {'agent':<34} {'reward':>9} {'delivered':>10} {'fit loss':>9}")
    print(f"    {'rule (the baseline)':<34} {h['rule'][0]:+9.1f} {h['rule'][1]:9.1f}%")
    print(f"    {'oracle (the bound)':<34} {h['teacher'][0]:+9.1f} {h['teacher'][1]:9.1f}%")

    rew, ok, cr, loss = C.dagger(env, S.ObstacleTeacher(env))
    print(f"    {'clone of the oracle':<34} {rew:+9.1f} {ok:9.1f}% {loss:9.4f}")
    e2 = env_for(1.20)
    rew2, ok2, cr2, loss2 = C.dagger(e2, R.SigmaRuleMean(e2, 0.5))
    print(f"    {'CONTROL: clone of the rule':<34} {rew2:+9.1f} {ok2:9.1f}% {loss2:9.4f}")

    got = rew - h["rule"][0]
    print(f"\n    the clone recovers {got:+.1f} of the {h['headroom']:+.1f} available")
    if loss2 < loss / 2:
        print(f"    the rule fits {loss/loss2:.0f}x better than the oracle does:")
        print("    the same observation maps to different oracle actions, so the")
        print("    gap is INFORMATION the drone does not have, not a policy it")
        print("    has failed to find. More training cannot reach it; a sensor")
        print("    or a better position estimate might.")


if __name__ == "__main__":
    learnable()
