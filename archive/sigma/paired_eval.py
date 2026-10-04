#!/usr/bin/env python3
"""
paired_eval.py -- policy against the rule on IDENTICAL episodes.

Unpaired comparisons were misleading here. The rule measured 80.5% on one
200-episode sample and 75.0% on another of 500; that 5.5-point swing is bigger
than any effect we are looking for. Running both agents over the same seeds
removes the episode draw from the comparison entirely, so the difference per
episode can be averaged and given a confidence interval directly.
"""
import glob, os, numpy as np
from approach_env_v5 import ApproachEnvV5
import rule_agent as R


class E(ApproachEnvV5):
    N_OBST = (8, 10); OBST_H = (7.2, 12.0); PAD_CLEAR = 0.5; OBST_R = (2.5, 5.0)


def main():
    e = E(pool="sigma_pool.npz", arm="sigma", seed=0, t_max=35, rho=0.707,
          sigma_scale=3.5)
    N = 400
    seeds = [9_000_000 + i for i in range(N)]

    def vec(mk):
        ag = mk()
        return np.array([R.rollout(ag, e, s)["success"] for s in seeds], float)

    rule = vec(lambda: R.SigmaRuleMean(e, 0.5))
    orac = vec(lambda: R.PadOracle(e))
    print(f"rule (aim at averaged fix) {100*rule.mean():.1f}%   "
          f"oracle {100*orac.mean():.1f}%   on {N} matched episodes\n", flush=True)
    print(f"{'policy':<18} {'deliv':>6} {'vs rule':>9} {'paired 95% CI':>18}", flush=True)
    diffs = []
    for d in sorted(glob.glob("overnight/obst_full_s*") +
                    glob.glob("overnight/obst_ent_s*")):
        z = glob.glob(f"{d}/ppo_*.zip")
        if not z:
            continue
        p = vec(lambda: R.PolicyAgent(z[0]))
        dd = p - rule
        m, se = dd.mean(), dd.std(ddof=1) / np.sqrt(N)
        diffs.append(m)
        print(f"{os.path.basename(d):<18} {100*p.mean():5.1f}% {100*m:+8.2f}pp "
              f" [{100*(m-1.96*se):+.2f},{100*(m+1.96*se):+.2f}]", flush=True)
    d = np.array(diffs)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nacross {len(d)} independently trained policies:")
    print(f"  mean advantage over the rule {100*d.mean():+.2f} pp +/- {100*se:.2f}")
    print(f"  95% CI [{100*(d.mean()-1.96*se):+.2f}, {100*(d.mean()+1.96*se):+.2f}] pp")
    room = orac.mean() - rule.mean()
    print(f"  headroom to the oracle is {100*room:.1f} pp; the policies capture "
          f"{100*d.mean()/room:.0f}% of it")


if __name__ == "__main__":
    main()
