#!/usr/bin/env python3
"""
sigma_v2.py -- a better uncertainty predictor, and an honest test of it.

WHY BOTHER

Feeding the landing policy a PERFECT uncertainty is worth +2.08 points
(95% CI [+1.01, +3.15], p=0.001, replicated on fresh seeds). Feeding it our
current sigma is worth -0.23 points -- nothing. The concept works; the
estimator is what fails. So the highest-value work left is making sigma better,
and this measures how much better it can get.

WHAT THE CURRENT PREDICTOR THROWS AWAY

It bins log(best_inliers) into eight buckets and takes the 68th percentile of
the error in each. One feature, eight numbers. Two things sit unused in the
same CSV:

1. THREE MATCHERS RAN ON EVERY FRAME (XFEAT_LG, XFEAT_MNN, XFEAT_STAR) and
   only XFEAT_LG is kept. Whether independent matchers AGREE on a pose is a
   classic confidence signal and costs nothing but the compute to run them.
2. THE SOLVE PREDICTS AN ALTITUDE. An aircraft knows its own height from the
   barometer even when it has no idea where it is horizontally, so a predicted
   z that disagrees with the barometer is a self-consistency check that needs
   no ground truth at all.

LEAKAGE DISCIPLINE

pred_error_m, pdm_at_k, recall_at_k and retrieval_gt_rank are computed from
ground truth and are never features -- calibrate.BANNED lists them and this
respects it. The altitude check is the one judgement call: it uses gt_z as a
stand-in for what a barometer would report. That is a modelling ASSUMPTION, so
the feature set is evaluated both with and without it and the difference is
reported rather than buried.
"""

import argparse
import collections
import csv
import glob
import os

import numpy as np

import calibrate as C

MATCHERS = ("XFEAT_LG", "XFEAT_MNN", "XFEAT_STAR")
PRIMARY = "XFEAT_LG"          # the fix actually used; the others only vote


def read_rows(root, ref="satellite"):
    rows = []
    for path in glob.glob(os.path.join(os.path.expanduser(root), "**",
                                       "all_results.csv"), recursive=True):
        d = os.path.basename(os.path.dirname(path))
        if ref not in d or d == ref:          # skip the aggregate duplicate
            continue
        with open(path, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def _f(r, k, default=np.nan):
    try:
        v = float(r[k])
        return v if np.isfinite(v) else default
    except (KeyError, ValueError, TypeError):
        return default


def build(root, use_alt=True):
    """One row per frame: features from every matcher, label from the primary."""
    by = collections.defaultdict(dict)
    for r in read_rows(root):
        by[(r.get("scene_name", ""), r.get("sample_id", ""))][
            r.get("matching_method", "")] = r

    X, y, scene, names = [], [], [], None
    for (sc, sid), m in sorted(by.items()):
        p = m.get(PRIMARY)
        if p is None or str(p.get("pnp_success", "")).lower() != "true":
            continue
        err = _f(p, "pred_error_m")
        if not np.isfinite(err) or err > C.ABSURD_M:
            continue                          # same protocol as the baseline

        inl = _f(p, "best_inliers", 0.0)
        cand = _f(p, "num_retrieval_candidates", 0.0)
        pxy = np.array([_f(p, "pred_x", 0.0), _f(p, "pred_y", 0.0)])

        # --- what the other matchers say -------------------------------
        others, o_inl = [], []
        for name in MATCHERS:
            if name == PRIMARY:
                continue
            o = m.get(name)
            if o is None or str(o.get("pnp_success", "")).lower() != "true":
                continue
            oxy = np.array([_f(o, "pred_x", np.nan), _f(o, "pred_y", np.nan)])
            if np.all(np.isfinite(oxy)):
                # clip before log: degenerate solves reach 1e34 and would
                # otherwise dominate every scale in the feature
                others.append(min(float(np.linalg.norm(oxy - pxy)), 1e6))
            o_inl.append(_f(o, "best_inliers", 0.0))

        n_ok = len(others)
        disagree = np.median(others) if others else 1e6   # nobody agreed
        o_inl = o_inl or [0.0]

        f = [np.log1p(max(inl, 0)),
             np.log1p(max(cand, 0)),
             np.log1p(disagree),
             float(n_ok),
             np.log1p(max(max(o_inl), 0)),
             np.log1p(max(min(o_inl), 0)),
             _f(p, "retrieval_time", 0.0) * 100.0,
             np.log1p(max(_f(p, "total_time", 0.0), 0)),
             1.0 if str(p.get("fix_plausible", "")).lower() == "true" else 0.0]
        nm = ["log_inliers", "log_candidates", "log_matcher_disagreement",
              "n_matchers_agreed", "log_max_other_inliers",
              "log_min_other_inliers", "retrieval_time", "log_total_time",
              "fix_plausible"]

        if use_alt:
            # |predicted altitude - what a barometer would say|
            dz = abs(_f(p, "pred_z", 0.0) - _f(p, "gt_z", 0.0))
            f.append(np.log1p(min(dz, 1e6)))
            nm.append("log_altitude_mismatch")

        if not all(np.isfinite(v) for v in f):
            continue
        X.append(f); y.append(err); scene.append(sc); names = nm
    return np.array(X), np.array(y), np.array(scene), names


# --------------------------------------------------------------------------
def pinball_fit(Xtr, ytr, Xte, q=0.68, epochs=1200, seed=0, n_models=5):
    """Median of several fits. A single fit was unstable -- leaving one feature
    out swung ENCE from 1.8 to 206, which is an optimisation artefact, not a
    statement about the feature. Ensembling removes it."""
    P = [_one_fit(Xtr, ytr, Xte, q, epochs, s) for s in range(n_models)]
    return np.median(np.stack(P), axis=0)


def _one_fit(Xtr, ytr, Xte, q=0.68, epochs=1200, seed=0):
    """
    Conditional quantile regression: predict the q-th quantile of |error|
    GIVEN the features, which is exactly what sigma is defined to be.

    The baseline reaches the same definition by splitting one feature into
    eight buckets and taking a quantile inside each. That cannot express "many
    inliers BUT the matchers disagree", which is the case worth catching.
    """
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(np.log1p(ytr), dtype=torch.float32).unsqueeze(1)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        d = yt - net(xt)
        (torch.maximum(q * d, (q - 1) * d)).mean().backward()
        opt.step(); sch.step()
    with torch.no_grad():
        p = net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy().ravel()
    return np.clip(np.expm1(p), C.SIGMA_FLOOR, C.SIGMA_CEIL)


def loso(X, y, scene, fit):
    out = np.zeros(len(y))
    for s in np.unique(scene):
        te = scene == s
        out[te] = fit(X[~te], y[~te], X[te])
    return out


def baseline_loso(X, y, scene, inl_col=0):
    """The current predictor, on the same folds: bins over log-inliers only."""
    out = np.zeros(len(y))
    for s in np.unique(scene):
        te = scene == s
        e, sig = C.fit_sigma_bins(np.expm1(X[~te, inl_col]), y[~te])
        out[te] = C.predict_sigma(np.expm1(X[te, inl_col]), e, sig)
    return out


def report(name, sigma, err, scene):
    e, _ = C.ence(sigma, err)
    k, cov, _ = C.conformal(sigma, err, scene, 0.10)
    r = float(np.corrcoef(sigma, err)[0, 1])
    # what actually matters downstream: does a big sigma mean a big error?
    hi = sigma >= np.median(sigma)
    sep = float(np.median(err[hi]) - np.median(err[~hi]))
    print(f"  {name:<34} ENCE {e:5.3f}   corr {r:+.3f}   k {k:4.2f}  "
          f"cov {cov:5.3f}   median-err split {sep:+5.1f} m")
    return dict(ence=e, corr=r, k=k, cov=cov, sep=sep)


def per_scene(sigma, err, scene):
    print(f"      {'scene':<10} {'n':>5} {'median err':>11} {'median sigma':>13}")
    for sc in np.unique(scene):
        m = scene == sc
        print(f"      {sc:<10} {int(m.sum()):5d} {np.median(err[m]):10.1f} m "
              f"{np.median(sigma[m]):12.1f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/GeoAnchor/results/s8_train")
    ap.add_argument("--ablate", action="store_true")
    a = ap.parse_args()

    X, y, scene, names = build(a.root, use_alt=True)
    print(f"{len(y)} frames, {X.shape[1]} features, scenes "
          f"{ {s: int((scene==s).sum()) for s in np.unique(scene)} }")

    # Scene_21 is a documented MATCHER FAILURE (GeoAnchor/CLAUDE.md: 93% bad,
    # 140 of 141 catastrophes), hand-excluded to make the published numbers.
    # So there are two questions, not one:
    #   A. can we beat the baseline where it already works?
    #   B. can we make the EXCLUSION UNNECESSARY -- can the estimator detect
    #      the bad scene itself, instead of a human removing it by name?
    # B is the one worth having: a drone cannot be told which scenes to skip.
    for tag, keep in (("A. clean subset, Scene_21 hand-excluded (the published setting)",
                       scene != "Scene_21"),
                      ("B. FULL data, Scene_21 included (nothing hand-removed)",
                       np.ones(len(y), bool))):
        Xs, ys, ss = X[keep], y[keep], scene[keep]
        print(f"\n=== {tag} ===  {len(ys)} frames")
        b = report("baseline: inliers, 8 bins", baseline_loso(Xs, ys, ss), ys, ss)
        v = report("v2: features + quantile reg", loso(Xs, ys, ss, pinball_fit), ys, ss)
        print(f"      ENCE {b['ence']:.3f} -> {v['ence']:.3f}"
              f"   ({b['ence']/max(v['ence'],1e-9):.1f}x better)"
              f"      coverage {b['cov']:.3f} -> {v['cov']:.3f} (target 0.900)")
        if keep.all():
            per_scene(loso(Xs, ys, ss, pinball_fit), ys, ss)

    if a.ablate:
        base = report("full feature set", loso(X, y, scene, pinball_fit), y, scene)
        print("\n  leave-one-feature-out on the FULL data (higher ENCE = feature helps)")
        for i, nm in enumerate(names):
            keep = [j for j in range(X.shape[1]) if j != i]
            e, _ = C.ence(loso(X[:, keep], y, scene, pinball_fit), y)
            print(f"    {nm:<28} {e:6.3f}   {e - base['ence']:+.3f}")


if __name__ == "__main__":
    main()
