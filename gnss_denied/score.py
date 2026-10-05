#!/usr/bin/env python3
"""score.py -- GeoAnchor's fixes on one run against the truth: how many frames got a fix, and the errors of the
accepted ones as median / p90 / p99 / share within 5, 10, 20 m (never the mean: one lost fix drags it anywhere).
    python gnss_denied/score.py outputs/gnss_denied/runs/<run> outputs/gazebo_geo/floor/flight_gt.jsonl
"""
import json
import sys

import numpy as np


def score(run, sidecar):
    recs = [json.loads(l) for l in open(f"{run}/records.jsonl")]
    recs = [r for r in recs if "accepted" in r]
    n = sum(1 for _ in open(sidecar))
    e = np.array([r["error_m"] for r in recs if r["accepted"] and r["error_m"] is not None], float)
    out = dict(frames=n, fixes=len(recs), accepted=len(e), accepted_pct=round(100 * len(e) / max(n, 1), 1))
    if len(e):
        out.update(median_m=round(float(np.median(e)), 2), p90_m=round(float(np.percentile(e, 90)), 2),
                   p99_m=round(float(np.percentile(e, 99)), 2),
                   **{f"within_{b}m_pct": round(100 * float((e <= b).mean()), 1) for b in (5, 10, 20)})
    return out


if __name__ == "__main__":
    s = score(sys.argv[1], sys.argv[2])
    json.dump(s, open(f"{sys.argv[1]}/score.json", "w"), indent=1)
    print(json.dumps(s))
