#!/usr/bin/env python3
"""
parse_forest.py -- read the forest sweep out of the flight logs.

The per-trial JSON was not written for most of this run: the dump sat behind
`len(results) > 1` and the sweep flies one mission per process. The logs carry
what the analysis needs -- the forest line, the outcome line, and any CRASH
line -- so parse those rather than throw away 20 flights and refly them.
"""
import re
import sys

import numpy as np

FOREST = re.compile(r"forest seed (\d+): (\d+) trees, clear column ([\d.]+) m")
OUT = re.compile(r"=> (\w+)\s+miss\s+([\d.]+) m\s+alt\s+([\d.]+) m\s+"
                 r"saw_marker=(\w+)\s+first (never|id\d+ at [\d.]+ m)")
CRASH = re.compile(r"CRASH into tree (\d+) at alt ([\d.]+) m")


def parse(path):
    trials, cur = [], None
    for line in open(path):
        m = FOREST.search(line)
        if m:
            if cur is not None:
                trials.append(cur)
            cur = dict(forest=int(m.group(1)), n_trees=int(m.group(2)),
                       clear=float(m.group(3)), crash=False, outcome=None)
            continue
        if cur is None:
            continue
        if CRASH.search(line):
            cur["crash"] = True
        m = OUT.search(line)
        if m:
            cur.update(outcome=m.group(1), miss=float(m.group(2)),
                       saw=m.group(4) == "True", acq=m.group(5))
            trials.append(cur); cur = None
    if cur is not None and cur["outcome"] is None:
        cur["outcome"] = "no_result"      # process died before printing
        trials.append(cur)
    return trials


if __name__ == "__main__":
    for path in sys.argv[1:]:
        t = parse(path)
        done = [x for x in t if x["outcome"] not in (None, "no_result")]
        print(f"\n{path}: {len(t)} trials seen, {len(done)} with a result")
        ld = sum(x["outcome"] == "landed" for x in done)
        ab = sum(x["outcome"] == "abort" for x in done)
        cr = sum(x["crash"] for x in done)
        print(f"  landed {ld}  abort {ab}  crashed {cr}  "
              f"other {len(done)-ld-ab}")
        if done:
            print(f"  clear column {min(x['clear'] for x in done):.2f}"
                  f"-{max(x['clear'] for x in done):.2f} m, "
                  f"trees {min(x['n_trees'] for x in done)}"
                  f"-{max(x['n_trees'] for x in done)}")
