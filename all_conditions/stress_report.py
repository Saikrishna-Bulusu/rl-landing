#!/usr/bin/env python3
"""
stress_report.py -- the full-stack stress test's flights (queue/stress_30sep.sh), one row each plus the totals, into
results/stress_2026-09-30.json. Every value comes from the flight's own files: run.log (the result line and the
GNSS-denied report), <stem>_geo.json (the cruise's rows: phase, simulated and wall-clock time), <stem>.csv (the decider's
loop: marker reads), manifest.json (what flew) and wall.json (wall-clock seconds, queue side).

    python3 geo/stress_report.py                       # every gazebo_geo/stress_e*/
    python3 geo/stress_report.py geo_magpie6 magpie    # one earlier flight: checks the reader, writes nothing
"""
import csv
import glob
import json
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAR_M = 2.0     # urban/local_planner.py C_CLEAR: the planner keeps this far from anything blocked
PAD_M = 1.0       # gz_learner.py: a touchdown farther than this from the pad center is "landed_off_pad"


def flight(run, stem="stress"):
    d = f"{R}/gazebo_geo/{run}"
    log = open(f"{d}/run.log").read().splitlines()
    res = json.loads([l for l in log if l.startswith("{") and '"outcome"' in l][-1])
    geo = json.loads([l for l in log if l.startswith("geo ")][-1][4:])
    rows = json.load(open(f"{d}/{stem}_geo.json"))["rows"]
    land = [r for r in rows if r["phase"].startswith("landing") or r["phase"] == "local 3D planner"]   # the decider's phase
    rate = lambda rs: round((rs[-1]["t"] - rs[0]["t"]) / (rs[-1]["wall"] - rs[0]["wall"]), 3) if len(rs) > 1 else None
    with open(f"{d}/{stem}.csv") as f:
        loop = list(csv.DictReader(f))
    first = next((r for r in loop if float(r["seen"]) > 0.5), None)
    alt = None if first is None else float(first["gz_z"]) if first["gz_z"] != "nan" else float(first["z"])
    man = json.load(open(f"{d}/manifest.json")) if os.path.exists(f"{d}/manifest.json") else {}
    wall = json.load(open(f"{d}/wall.json")) if os.path.exists(f"{d}/wall.json") else {}
    cl = res.get("closest_m", {})
    out = res["outcome"]
    viol = ([out] if out in ("crashed", "struck", "landed_off_pad", "hard") else []) \
        + (["crane within 2 m"] if cl.get("crane", 1e9) < CLEAR_M else []) \
        + (["GNSS restored"] if geo.get("gnss_restored_at") is not None else [])
    return dict(run=run, seed=res["seed"], crane_at=man.get("crane_at"), wind_dir_deg=man.get("wind_dir_deg"),
                outcome=out, miss_m=round(res["miss_m"], 3), miss_est_m=round(res["miss_est_m"], 3), miss_declared_m=res.get("miss_declared_m"),
                crane_clear_m=cl.get("crane"), clear_any_m=min(cl.values()) if cl else None, crashed_into=res.get("crashed_into"),
                swoops=res["swoops"], closest_bird_m=round(res["closest_bird_m"], 2), violations=viol,
                denied_s=geo["denied_s"], gnss_restored_at=geo.get("gnss_restored_at"),
                agp_on_after_denial_pct=geo.get("agp_on_after_denial_pct"), fixes_per_sim_s=geo["fixes_per_sim_s"],
                fix_err_m=geo["fix_err_m"], fix_within_10m_pct=geo["fix_within_10m_pct"], est_err_m=geo["est_err_m"],
                marker_fixes=geo.get("marker_fixes"), marker_first_alt_m=None if alt is None else round(alt, 1),
                decider_s=round(res["t_s"], 1), rtf_whole=geo.get("rtf"), rtf_landing=rate(land),
                wall_s=wall.get("wall_s"), rig=res.get("rig"), commit=man.get("commit"), dirty=man.get("dirty"),
                sha256={k: v for k, v in man.get("sha256", {}).items() if k != "wind.config"})


def main():
    if len(sys.argv) > 2:                       # the reader on an earlier flight
        print(json.dumps(flight(sys.argv[1], sys.argv[2]), indent=1)); return
    runs = sorted(os.path.basename(p) for p in glob.glob(f"{R}/gazebo_geo/stress_e*") if os.path.exists(f"{p}/wall.json"))
    F = [flight(r) for r in runs]
    assert F, "no flights yet"
    # frozen: every flight ran the same commit and the same code, checkpoint, detector and world
    assert len({f["commit"] for f in F}) == 1 and all(f["sha256"] == F[0]["sha256"] for f in F) and not any(f["dirty"] for f in F), \
        "the flights did not all fly the same frozen version"
    n = lambda k: sum(f["outcome"] == k for f in F)
    rng = lambda k: [min(f[k] for f in F if f[k] is not None), max(f[k] for f in F if f[k] is not None)] if any(f[k] is not None for f in F) else None
    tot = dict(flights=len(F), landed=n("landed"), crashed=n("crashed"), struck=n("struck"), timeout=n("timeout"),
               off_pad=n("landed_off_pad"), with_violation=sum(bool(f["violations"]) for f in F),
               violations=sorted({v for f in F for v in f["violations"]}), gnss_restored=sum(f["gnss_restored_at"] is not None for f in F),
               miss_m=rng("miss_m"), crane_clear_m=rng("crane_clear_m"), closest_bird_m=rng("closest_bird_m"),
               swoops=rng("swoops"), marker_first_alt_m=rng("marker_first_alt_m"), denied_s=rng("denied_s"),
               fix_err_median_m=[min(f["fix_err_m"]["median"] for f in F), max(f["fix_err_m"]["median"] for f in F)],
               est_err_max_m=max(f["est_err_m"]["max"] for f in F), agp_on_after_denial_pct=rng("agp_on_after_denial_pct"),
               rtf_whole=rng("rtf_whole"), rtf_landing=rng("rtf_landing"), wall_h=round(sum(f["wall_s"] for f in F) / 3600, 2),
               commit=F[0]["commit"])
    json.dump(dict(flights=F, totals=tot), open(f"{R}/results/stress_2026-09-30.json", "w"), indent=1)
    for f in F:
        print(f"{f['run']:10s} {f['outcome']:15s} miss {f['miss_m']:.2f}  crane {f['crane_clear_m']}  bird {f['closest_bird_m']} "
              f"({f['swoops']} swoops)  marker at {f['marker_first_alt_m']} m  rtf {f['rtf_landing']}  {f['violations']}")
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
