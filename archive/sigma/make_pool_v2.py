#!/usr/bin/env python3
"""
make_pool_v2.py -- a replay pool whose sigma comes from the improved predictor.

Everything else is held identical to sigma_pool.npz: the same 440 frames, the
same errors, the same scenes, the same conformal and rho machinery. ONLY the
sigma column changes. So a policy trained on this pool differs from one trained
on the old pool in exactly one respect, which is the point.

WHY THIS AND NOT ENCE

On the clean subset the new predictor is WORSE by ENCE (0.418 against 0.313)
and BETTER by rank correlation (+0.611 against +0.445). Those disagree, and
ENCE is a proxy: the policy never reads a calibration curve, it reads a number
and decides how far to search. Rank is closer to what it uses, but arguing
about which proxy to prefer is how this project has wasted time before. Train
on it and measure deliveries.
"""

import argparse
import numpy as np

import calibrate as C
import sigma_v2 as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/GeoAnchor/results/s8_train")
    ap.add_argument("--out", default="sigma_pool_v2.npz")
    ap.add_argument("--keep-scene21", action="store_true")
    a = ap.parse_args()

    X, y, scene, names = V.build(a.root, use_alt=True)
    if not a.keep_scene21:
        m = scene != "Scene_21"
        X, y, scene = X[m], y[m], scene[m]

    sigma = V.loso(X, y, scene, V.pinball_fit)
    inl = np.expm1(X[:, names.index("log_inliers")])

    e, _ = C.ence(sigma, y)
    k, cov, nf = C.conformal(sigma, y, scene, 0.10)
    old = np.load("sigma_pool.npz", allow_pickle=True)

    np.savez(a.out, sigma=sigma, err=y, scene=scene, inliers=inl,
             conformal_k=k, ence=e, alpha=0.10,
             rho=float(old["rho"]), rho_frame=float(old["rho_frame"]))
    print(f"  {a.out}: {len(y)} frames, {len(np.unique(scene))} scenes")
    print(f"  ENCE {e:.3f}   conformal k {k:.2f}   coverage {cov:.3f}   "
          f"corr {np.corrcoef(sigma, y)[0,1]:+.3f}")
    print(f"  sigma p50 {np.median(sigma):.1f} m   err p50 {np.median(y):.1f} m")
    print(f"  (old pool: ENCE {float(old['ence']):.3f}, k {float(old['conformal_k']):.2f}, "
          f"corr {np.corrcoef(old['sigma'], old['err'])[0,1]:+.3f})")


if __name__ == "__main__":
    main()
