#!/usr/bin/env python3
"""
calibrate.py -- turn GeoAnchor fixes into (1) an ENCE number, (2) a conformal
coverage guarantee, and (3) the (sigma, error) replay pool the landing env uses.

    python3 calibrate.py ~/GeoAnchor/results/s8_train --exclude-scene Scene_21

Why this exists
---------------
CLAUDE.md reports the estimator as "12.0->12.0, 21->21, 38->45". That is an
eyeballed, unnamed calibration curve. ENCE is the published metric for exactly
that claim (MUSE, arXiv 2605.17421), so reporting it makes the result
comparable to other work instead of a table only this project can read.

Conformal calibration then upgrades the claim from "sigma tracks error" to
"sigma has >= 90% coverage", which is the sentence a downstream policy needs.

The sigma predictor here is deliberately the SIMPLEST thing that works: bin by
inlier count, and predict sigma as a high quantile of |error| within the bin,
fitted leave-one-scene-out. It uses only `best_inliers`, which is in
all_results.csv, so this runs on data already on disk with no harness patch and
no .pkl parsing.

To swap in the real step16 estimator, replace predict_sigma() -- everything
downstream (ENCE, conformal, pool) is agnostic to where sigma came from.

HARD RULES inherited from GeoAnchor/CLAUDE.md, enforced here:
  - pred_error_m is the LABEL. It is never a feature. Same for pdm_at_k,
    recall_at_k, retrieval_gt_rank.
  - Cross-validation is grouped by SCENE, never by row.
  - Never quote the mean. Catastrophes reach 2e93 m.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# Columns that are computed from ground truth. A flying aircraft has none of
# them, so using any as a predictor is leakage. CLAUDE.md calls this
# non-negotiable and it is.
BANNED = {
    "pred_error_m", "pdm_at_k", "recall_at_k", "retrieval_gt_rank",
    "PDE", "R_i", "location_error_list", "truePos",
}

# A PnP solution this far out is a degenerate solve, not a localisation result.
# The harness sets pnp_success=True whenever inliers > 0, so it cannot filter
# these itself.
ABSURD_M = 1000.0

SIGMA_FLOOR, SIGMA_CEIL = 0.5, 100.0  # EKF3 clamps horizontal to [0.01, 100]


def load(roots, ref="satellite", matcher="XFEAT_LG"):
    """Read every all_results.csv under the given result roots."""
    import csv

    rows = []
    for root in roots:
        pattern = os.path.join(os.path.expanduser(root), "**", "all_results.csv")
        for path in glob.glob(pattern, recursive=True):
            # Per-matcher subdirectories duplicate rows into the parent
            # aggregate file. Take only the per-matcher ones or every fix is
            # counted twice.
            d = os.path.basename(os.path.dirname(path))
            # d == ref is the aggregate file, which is the concatenation of the
            # per-matcher ones sitting beside it. Matching on `ref in d` alone
            # accepts both and double-counts every fix.
            if ref not in d or d == ref:
                continue
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if matcher and r.get("matching_method") != matcher:
                        continue
                    rows.append(r)
    return rows


def to_arrays(rows, exclude_scenes=()):
    """Extract the three things we need: inliers, error, scene."""
    inl, err, scene = [], [], []
    dropped_absurd = 0
    for r in rows:
        sc = r.get("scene_name", "")
        if sc in exclude_scenes:
            continue
        if str(r.get("pnp_success", "")).lower() != "true":
            continue
        try:
            e = float(r["pred_error_m"])
            i = float(r["best_inliers"])
        except (KeyError, ValueError, TypeError):
            continue
        if not np.isfinite(e) or not np.isfinite(i):
            continue
        if e > ABSURD_M:
            # Kept out of calibration fitting but counted, because the fraction
            # that land here is itself a reportable number.
            dropped_absurd += 1
            continue
        inl.append(i)
        err.append(e)
        scene.append(sc)
    return (np.asarray(inl, float), np.asarray(err, float),
            np.asarray(scene, object), dropped_absurd)


def fit_sigma_bins(inl, err, n_bins=8, q=0.68):
    """
    Fit the sigma predictor on ONE fold.

    Returns bin edges over log-inliers plus the per-bin sigma. q=0.68 makes
    sigma a one-sigma-equivalent quantile of the absolute error, which is what
    a Gaussian sigma means and what EKF3 expects to receive.
    """
    li = np.log1p(inl)
    # Quantile edges, so every bin holds a comparable number of fixes rather
    # than whatever an equal-width split happens to catch.
    edges = np.quantile(li, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    sig = np.empty(n_bins)
    for b in range(n_bins):
        m = (li >= edges[b]) & (li < edges[b + 1])
        sig[b] = np.quantile(err[m], q) if m.sum() >= 5 else np.quantile(err, q)
    return edges, np.clip(sig, SIGMA_FLOOR, SIGMA_CEIL)


def predict_sigma(inl, edges, sig):
    """Apply a fitted predictor. Swap this function to use step16's estimator."""
    idx = np.clip(np.searchsorted(edges, np.log1p(inl), side="right") - 1,
                  0, len(sig) - 1)
    return sig[idx]


def loso_sigma(inl, err, scene, n_bins=8):
    """
    Leave-one-scene-out predicted sigma for every row.

    Grouped by scene, never by row. With few scenes, rho/log_rho/altitude
    separate the scenes perfectly and a row-wise split leaks scene identity --
    CLAUDE.md flags this and it is why this is not a plain KFold.
    """
    out = np.empty_like(err)
    for held in np.unique(scene):
        tr = scene != held
        te = ~tr
        if tr.sum() < 20:
            out[te] = np.quantile(err, 0.68)
            continue
        edges, sig = fit_sigma_bins(inl[tr], err[tr], n_bins)
        out[te] = predict_sigma(inl[te], edges, sig)
    return out


def ence(sigma, err, n_bins=10):
    """
    Expected Normalized Calibration Error (MUSE, arXiv 2605.17421).

    Bin by predicted uncertainty; in each bin compare root-mean-variance
    against empirical RMSE:

        ENCE = (1/M) * sum_j |RMSE(j) - RMV(j)| / RMV(j)

    Lower is better; 0 is perfect calibration. The RMSE inside a bin is a
    mean-of-squares, which is why absurd fixes are excluded upstream -- one
    2e93 m row would otherwise define the metric.
    """
    order = np.argsort(sigma)
    bins = np.array_split(order, n_bins)
    total, table = 0.0, []
    for b in bins:
        if len(b) == 0:
            continue
        rmv = float(np.sqrt(np.mean(sigma[b] ** 2)))
        rmse = float(np.sqrt(np.mean(err[b] ** 2)))
        total += abs(rmse - rmv) / max(rmv, 1e-9)
        table.append({"n": len(b), "rmv_m": rmv, "rmse_m": rmse,
                      "median_sigma_m": float(np.median(sigma[b])),
                      "median_err_m": float(np.median(err[b]))})
    return total / max(len(table), 1), table


def conformal(sigma, err, scene, alpha=0.10):
    """
    Split-conformal calibration, grouped by scene.

    The nonconformity score is err/sigma. Taking its (1-alpha) quantile on held
    -out scenes gives a multiplier k such that

        P(err <= k * sigma) >= 1 - alpha

    distribution-free -- no assumption that the error is Gaussian, which it
    very much is not. Reported coverage is measured on the held-out scenes, so
    it is an honest estimate rather than the training quantile read back.
    """
    ks, covs = [], []
    for held in np.unique(scene):
        tr, te = scene != held, scene == held
        if tr.sum() < 20 or te.sum() < 5:
            continue
        s = err[tr] / np.maximum(sigma[tr], 1e-9)
        # Finite-sample correction: the conformal quantile uses (n+1) so the
        # guarantee holds for a genuinely unseen point, not just asymptotically.
        n = len(s)
        k = float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
        ks.append(k)
        covs.append(float(np.mean(err[te] <= k * sigma[te])))
    return float(np.median(ks)), float(np.mean(covs)), len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="GeoAnchor result roots")
    ap.add_argument("--ref", default="satellite")
    ap.add_argument("--matcher", default="XFEAT_LG")
    # action="append" appends to the default rather than replacing it, so a
    # literal default would make Scene_21 inescapable and print it twice.
    ap.add_argument("--exclude-scene", action="append", default=None)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default="sigma_pool.npz")
    a = ap.parse_args()
    if a.exclude_scene is None:
        a.exclude_scene = ["Scene_21"]

    rows = load(a.roots, a.ref, a.matcher)
    if not rows:
        sys.exit(f"no rows: check --ref/--matcher and that {a.roots} has "
                 f"all_results.csv under a *{a.ref}* directory")

    inl, err, scene, n_absurd = to_arrays(rows, set(a.exclude_scene))
    if len(err) < 50:
        sys.exit(f"only {len(err)} usable fixes -- too few to calibrate")

    sigma = loso_sigma(inl, err, scene)
    e, table = ence(sigma, err)
    k, cov, n_folds = conformal(sigma, err, scene, a.alpha)

    print(f"\n{a.matcher} / {a.ref} / excluding {a.exclude_scene}")
    print(f"{len(err)} fixes, {len(np.unique(scene))} scenes, "
          f"{n_absurd} absurd (>{ABSURD_M:.0f} m) held out\n")

    print("error percentiles (never the mean):")
    for p in (50, 90, 99):
        print(f"  p{p:<3} {np.percentile(err, p):8.2f} m")
    for t in (5, 10, 20):
        print(f"  within {t:2d} m: {100 * np.mean(err < t):5.1f}%")

    print(f"\nENCE = {e:.3f}   (0 = perfect; leave-one-scene-out)\n")
    print(f"  {'n':>5} {'RMV':>9} {'RMSE':>9}   {'med sigma':>9} {'med err':>9}")
    for t in table:
        print(f"  {t['n']:>5} {t['rmv_m']:>8.2f}m {t['rmse_m']:>8.2f}m   "
              f"{t['median_sigma_m']:>8.2f}m {t['median_err_m']:>8.2f}m")

    print(f"\nconformal @ alpha={a.alpha}: k = {k:.2f}, "
          f"held-out coverage = {100 * cov:.1f}% over {n_folds} scene folds")
    print(f"  -> true position is within {k:.2f} * sigma "
          f"at least {100 * (1 - a.alpha):.0f}% of the time")

    np.savez(a.out, sigma=sigma, err=err, scene=scene.astype(str),
             inliers=inl, conformal_k=k, ence=e, alpha=a.alpha)
    print(f"\nwrote {a.out}  ({len(err)} pairs) -- this is the replay pool\n")

    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as fh:
        json.dump({"ence": e, "conformal_k": k, "coverage": cov,
                   "n": len(err), "n_absurd": n_absurd,
                   "matcher": a.matcher, "excluded": a.exclude_scene,
                   "bins": table}, fh, indent=2)


if __name__ == "__main__":
    main()
