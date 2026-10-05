#!/usr/bin/env python3
"""
overnight_report.py -- what the night's queue (conditions/wind/queue/overnight_27sep.sh) found, as one markdown page:
  1. MJX ladder per recipe: landed / crashed / hard (off-pad or too fast) / gave up at 12-20 m/s, each seed and the
     spread over seeds (1024 paired flights per wind per checkpoint, evaluation seed 10000)
  2. Gazebo ladder: the same 20 yards x 4 winds flown by every decider; paired against the rule, flight by flight
  3. the GeoAnchor flights: outcome, and how well GeoAnchor + the pad marker navigated without GNSS
    python3 conditions/wind/overnight_report.py > conditions/wind/queue/overnight_report.md
"""
import collections
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
WINDS = (12.0, 15.0, 18.0, 20.0)


def mjx():
    out = ["## 1. MJX ladder (1024 paired flights per wind)", "",
           "Each cell: landed / crashed / hard (off the pad or too fast) / gave up, in %.", ""]
    rows = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{HERE}/runs/*_ladder.json")):
        name = os.path.basename(f)[:-len("_ladder.json")]
        d = json.load(open(f))
        d = d if isinstance(d, list) else d.get("rows", [])
        m = re.match(r"(.+)_s(\d+)_best$", name)
        recipe, seed = (m.group(1), int(m.group(2))) if m else (name, None)
        rows[recipe][seed] = {float(r["w10"]): r for r in d}
    head = "| recipe | seed | " + " | ".join(f"{w:g} m/s" for w in WINDS) + " |"
    out += [head, "|" + "---|" * (2 + len(WINDS))]
    cell = lambda r: f"{100 * r['landed']:.0f} / {100 * r['crashed']:.1f} / {100 * r['hard']:.1f} / {100 * r['timeout']:.0f}"
    for recipe in sorted(rows, key=lambda k: (k != "rule", k)):
        seeds = rows[recipe]
        for s in sorted(seeds, key=lambda x: -1 if x is None else x):
            out.append(f"| {recipe} | {'' if s is None else s} | " +
                       " | ".join(cell(seeds[s][w]) if w in seeds[s] else "" for w in WINDS) + " |")
        if len(seeds) > 1:                                     # the spread over seeds
            summ = []
            for w in WINDS:
                land = [100 * seeds[s][w]["landed"] for s in seeds if w in seeds[s]]
                crash = [100 * seeds[s][w]["crashed"] for s in seeds if w in seeds[s]]
                summ.append(f"**{np.median(land):.0f}** ({min(land):.0f}-{max(land):.0f}) / {np.median(crash):.1f} "
                            f"({min(crash):.1f}-{max(crash):.1f})")
            out.append(f"| **{recipe}: median (range) over {len(seeds)} seeds, landed / crashed** | | " + " | ".join(summ) + " |")
    return out


def gazebo():
    path = f"{R}/outputs/gazebo_wind/ladder_rule/flights.jsonl"
    out = ["", "## 2. Gazebo ladder (20 yards x 4 winds, the same gusts for every decider)", ""]
    if not os.path.exists(path):
        return out + ["(no flights)"]
    fl = [json.loads(l) for l in open(path)]
    by = collections.defaultdict(dict)
    for r in fl:
        by[r["decider"]][(r["episode"], r["w10"])] = r
    ws = sorted({r["w10"] for r in fl})
    out += ["Each cell: landed of flown (crashed, hard, gave up).", "",
            "| decider | " + " | ".join(f"{w:g} m/s" for w in ws) + " |", "|" + "---|" * (1 + len(ws))]
    for d in sorted(by, key=lambda k: (k != "rule", k)):
        cells = []
        for w in ws:
            rs = [r for (e, ww), r in by[d].items() if ww == w]
            c = collections.Counter(r["outcome"] for r in rs)
            cells.append(f"{c['landed']}/{len(rs)} ({c['crashed']}, {c['hard']}, {c['timeout']})" if rs else "")
        out.append(f"| {d} | " + " | ".join(cells) + " |")
    out += ["", "Paired against the rule (same yard, same wind, same gusts): learner landed where the rule didn't / "
            "rule landed where the learner didn't.", ""]
    for d in sorted(by):
        if d == "rule":
            continue
        cells = []
        for w in ws:
            keys = [k for k in by[d] if k[1] == w and k in by["rule"]]
            win = sum(by[d][k]["outcome"] == "landed" and by["rule"][k]["outcome"] != "landed" for k in keys)
            lose = sum(by[d][k]["outcome"] != "landed" and by["rule"][k]["outcome"] == "landed" for k in keys)
            cells.append(f"{w:g} m/s: +{win} / -{lose} of {len(keys)}")
        out.append(f"- {d}: " + ", ".join(cells))
    return out


def geo():
    out = ["", "## 3. GeoAnchor flights (GNSS denied over the suburb, the pad marker below 50 m)", ""]
    for name in ("geo_calm", "geo_obst_baro", "geo_calm_rng", "geo_calm2", "geo_wind9", "geo_obst", "geo_magpie"):   # the first two: barometer only (no rangefinder)
        d = f"{R}/outputs/gazebo_geo/{name}"
        fl = f"{d}/flights.jsonl"
        if os.path.exists(fl):                                # gz_windland's record
            r = json.loads(open(fl).read().splitlines()[-1])
        else:                                                 # gz_learner's: its printed record in the run log
            recs = []
            for l in (open(f"{d}/run.log") if os.path.exists(f"{d}/run.log") else []):
                if l.startswith("{"):                        # the log mixes words and records
                    try:
                        x = json.loads(l)
                    except ValueError:
                        continue
                    if isinstance(x, dict) and "outcome" in x:
                        recs.append(x)
            if not recs:
                err = open(f"{d}/run.err").read().strip().splitlines()[-1:] if os.path.exists(f"{d}/run.err") else []
                out.append(f"- {name}: not flown{' -- ' + err[0][:200] if err else ''}")
                continue
            x = recs[-1]
            r = dict(decider=x["decider"] + (f", magpie {x['swoops']} swoops, closest {x['closest_bird_m']:.1f} m"
                                             if x.get("magpie") != "none" else ", unmapped obstacles"),
                     w10=0.0, outcome=x["outcome"], miss=x["miss_m"], t=x["t_s"])
        g = sorted(glob.glob(f"{d}/geo_*.json") + glob.glob(f"{d}/*_geo.json"))
        rep = json.load(open(g[-1]))["report"] if g else {}
        e, fe = rep.get("est_err_m") or {}, rep.get("fix_err_m") or {}
        out.append(f"- **{name}** ({r['decider']}, wind {r['w10']:g} m/s): **{r['outcome']}**, {r['miss']:.2f} m from the "
                   f"pad centre, {r['t']:.0f} s landing; GNSS denied {rep.get('denied_s')} s"
                   f"{'' if rep.get('gnss_restored_at') is None else ' (GNSS came back at t=' + format(rep['gnss_restored_at'], '.0f') + ' s)'}; "
                   f"PX4's position error median {e.get('median')} m, p90 {e.get('p90')} m, worst {e.get('max')} m; "
                   f"{rep.get('sent')} GeoAnchor fixes (median {fe.get('median')} m, p90 {fe.get('p90')} m), "
                   f"{rep.get('marker_fixes')} marker fixes; video {os.path.basename(glob.glob(d + '/*.mp4')[0]) if glob.glob(d + '/*.mp4') else 'none'} (internal: Esri floor)")
    return out


if __name__ == "__main__":
    print("# Overnight 26-27 Sept\n")
    print("\n".join(mjx() + gazebo() + geo()))
