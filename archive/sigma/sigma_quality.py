#!/usr/bin/env python3
"""
sigma_quality.py -- HOW GOOD does sigma have to be before it is worth anything?

Three points are now measured, and together they are strange:

    old sigma  (corr +0.445)   90.27%   -- WORSE than no uncertainty at all
    v2 sigma   (corr +0.611)   91.43%   -- exactly equal to no uncertainty
    perfect    (corr  1.000)   93.53%   -- worth +2.08 points

So improving the predictor moved sigma from actively harmful to merely useless.
There is evidently a threshold of estimator quality below which the channel
carries nothing a policy can act on, and +0.611 is still under it.

Knowing WHERE that threshold sits turns "make sigma better" into a spec:
either the predictor clears the bar or building it is not worth the effort.

METHOD

Build pools whose sigma has a CONTROLLED informativeness, holding everything
else fixed. A Gaussian copula couples sigma's rank to the error's rank at a
chosen strength, then the real sigma distribution's own quantiles are applied,
so the marginal spread of sigma is identical across pools and only how much it
KNOWS changes. Same 440 frames, same errors, same scenes throughout.
"""

import argparse
import math

import numpy as np


def make_pool(err, sigma_ref, rho, rng):
    """Sigma with rank-correlation `rho` to the error, real sigma's marginals."""
    from scipy.stats import norm, rankdata
    n = len(err)
    u_err = (rankdata(err) - 0.5) / n
    z = rho * norm.ppf(u_err) + math.sqrt(max(1 - rho ** 2, 0.0)) * rng.normal(size=n)
    u = norm.cdf(z)
    return np.quantile(sigma_ref, np.clip(u, 1e-6, 1 - 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.60, 0.80, 0.90, 1.00])
    ap.add_argument("--base", default="sigma_pool.npz")
    a = ap.parse_args()

    d = np.load(a.base, allow_pickle=True)
    err, scene, inl = d["err"], d["scene"], d["inliers"]
    sigma_ref = d["sigma"]
    rng = np.random.default_rng(0)

    print(f"{'target rho':>11} {'pearson':>9} {'spearman':>10}  file")
    for rho in a.rhos:
        s = make_pool(err, sigma_ref, rho, rng)
        from scipy.stats import spearmanr
        c = float(np.corrcoef(s, err)[0, 1])
        rc = float(spearmanr(s, err).statistic)
        out = f"sigma_pool_q{int(rho*100)}.npz"
        np.savez(out, sigma=s, err=err, scene=scene, inliers=inl,
                 conformal_k=float(d["conformal_k"]), ence=float(d["ence"]),
                 alpha=0.10, rho=float(d["rho"]), rho_frame=float(d["rho_frame"]))
        print(f"{rho:11.2f} {c:+9.3f} {rc:+10.3f}  {out}")
    print(f"\n  for reference: old sigma corr "
          f"{float(np.corrcoef(sigma_ref, err)[0,1]):+.3f}, v2 corr +0.611")


if __name__ == "__main__":
    main()
