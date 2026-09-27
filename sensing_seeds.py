#!/usr/bin/env python3
"""Same comparison, three training seeds per arm, paired eval episodes.

One seed cannot separate a real gap from the luck of an initialisation.
Reward per episode has a large spread here, so the number that matters is
the spread across seeds, not the mean of one run.
"""
import numpy as np, torch
from scipy import stats
import sensing_compare as C
import sensing_headroom as S


def run(mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    e = C.make(mode)
    return C.dagger(e, S.ObstacleTeacher(e), seed0=6_000_000 + seed * 100_003)


def main():
    res = {}
    for mode in ("sectors", "rich"):
        rows = [run(mode, s) for s in (0, 1, 2)]
        res[mode] = rows
        rew = [r[0] for r in rows]
        print(f"{mode:<8} reward {np.mean(rew):+7.1f} +/- {np.std(rew, ddof=1):4.1f}   "
              f"per seed {['%+.1f' % x for x in rew]}")
        print(f"         {np.mean([r[1] for r in rows]):5.1f}% delivered   "
              f"{np.mean([r[2] for r in rows]):5.1f}% crash")
    a = [r[0] for r in res["sectors"]]; b = [r[0] for r in res["rich"]]
    t, p = stats.ttest_ind(b, a, equal_var=False)
    print(f"\nrich - sectors = {np.mean(b)-np.mean(a):+.1f} reward   p = {p:.3f}   (n=3 seeds each)")
    ca = np.mean([r[2] for r in res["sectors"]]); cb = np.mean([r[2] for r in res["rich"]])
    print(f"crash          = {cb-ca:+.1f} pp  ({ca:.1f}% -> {cb:.1f}%)")
    da = np.mean([r[1] for r in res["sectors"]]); db = np.mean([r[1] for r in res["rich"]])
    print(f"delivered      = {db-da:+.1f} pp  ({da:.1f}% -> {db:.1f}%)")


if __name__ == "__main__":
    main()
