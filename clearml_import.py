"""Put past training logs into ClearML (history projects). Idempotent: a task whose name already exists in its project is skipped.
  python clearml_import.py [--dry-run] [--only wind|fast_sim]
Wind scalars mirror wind/ppo.py's clearml_log (kept in step by hand: importing ppo would pull in jax)."""
import argparse, glob, json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WIND, FAST = "rl-landing/wind/history", "rl-landing/fast_sim/history"
PPO_NAMES = ["total loss", "policy loss", "value loss", "entropy (deciding)", "approx KL", "stopped early (KL)"]   # as ppo.py


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]     # json reads the bare NaN the logs hold


def wind_points(row):
    """(title, series, value) for one round, as ppo.clearml_log (minus speed: old logs lack the round time)."""
    pts = [("outcomes (training flights)", k, row[k]) for k in ("landed", "crashed", "hard", "timeout")]
    pts += [("return", "mean episode return", row["ret"]), ("curriculum", "W (wind m/s)", row["w_cur"]), ("curriculum", "band landed", row["band_landed"])]
    pts += [("PPO", n, v) for n, v in zip(PPO_NAMES, row["loss"])]
    pts += [("policy std", n, v) for n, v in zip(("x", "y"), row["std"])]
    if "val" in row:
        pts += [("validation landed", f"W{w}", v) for w, v in zip((9, 12, 15, 18), row["val"])]
    pts += [("test-yard landed", f"W{e['w10']:.0f}", e["landed"]) for e in row.get("eval", [])]
    return pts


def wind_jobs():
    for p in sorted(glob.glob(f"{ROOT}/wind/runs/*.jsonl")):
        rs = rows(p)
        if not rs or "it" not in rs[0]:
            yield os.path.basename(p), None, "not a training log (no 'it' rows)"
            continue
        name, hp, tags = os.path.basename(p)[:-6], {}, []
        best = p[:-6] + "_best.npz"
        if os.path.exists(best):
            hp = json.loads(str(np.load(best)["meta"]))
            tags = [n for n, on in (("fly", hp.get("fly")), ("decide", hp.get("decide")), ("flyer", hp.get("flyer"))) if on]
        pts = [(it, t, s, v) for r in rs for t, s, v in wind_points(r) for it in [r["it"]]]
        yield os.path.basename(p), dict(project=WIND, name=name, tags=tags + ["history"], hp=hp, pts=pts), None


def fast_jobs():
    for p in sorted(glob.glob(f"{ROOT}/urban/rl_*.json")):
        d = json.load(open(p))
        runs = {k: v for k, v in d.items() if isinstance(v, dict) and "log" in v}
        if not runs:
            yield os.path.basename(p), None, "not a training output (no per-seed 'log')"
            continue
        flags = {k: d[k] for k in ("map", "nav", "birds", "fog", "unmapped", "cameras", "hunter", "rig") if k in d}
        for tag, run in runs.items():
            pts = []
            for r in run["log"]:
                if "gen" not in r:                         # the trailing kept_* summary row
                    continue
                g = r["gen"] + 1                           # gen -1 = the copy before any PPO -> x 0
                pts += [(g, "clean delivery (%)", n, r[k]) for n, k in (("training", "clean"), ("eval greedy", "eval_clean"), ("eval sampled", "eval_sampled"), ("eval mean", "eval_mean")) if k in r]
                pts += [(g, "violations (%)", n, r[k]) for n, k in (("training", "violation"), ("eval greedy", "eval_violation")) if k in r]
                pts += [(g, "policy shares (%)", n, r[k]) for n, k in (("rule", "rule_share"), ("planner", "planner_share")) if k in r]
                pts += [(g, "delivered (%)", "training", r["delivered"])] if "delivered" in r else []
                pts += [(g, "clean delivery (%)", f"baseline {b}", v) for b, v in d["baselines"].items()]
            hp = dict(flags, eyes="_eyes" in tag, **{f"baseline {b}": v for b, v in d["baselines"].items()})
            fin = {f"{m}/{k}": (v[0] if isinstance(v, list) else v) for m, e in run["eval"].items() for k, v in e.items()}   # final ladder eval; vs_* = pp mean
            yield f"{os.path.basename(p)}:{tag}", dict(project=FAST, name=tag, tags=["history", "eyes" if "_eyes" in tag else "no-eyes"], hp=hp, final=fin, pts=pts), None


def existing(project):
    from clearml import Task
    return {t["name"] for t in Task.query_tasks(project_name=project, additional_return_fields=["name"], task_filter={"page_size": 500})}


def push(job):
    from clearml import Task
    t = Task.create(project_name=job["project"], task_name=job["name"], task_type=Task.TaskTypes.training)
    t.add_tags(job["tags"])
    t.connect({k: ("" if v is None else v) for k, v in job["hp"].items()}, name="hyperparameters")
    if job.get("final"):
        t.connect(job["final"], name="final eval")
    lg = t.get_logger()
    for it, title, series, v in job["pts"]:
        if isinstance(v, (int, float)) and not math.isnan(v):     # NaN: nothing finished yet
            lg.report_scalar(title, series, float(v), int(it))
    lg.flush()
    t.mark_started()
    t.mark_completed()
    return t.get_output_log_web_page()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=("wind", "fast_sim"))
    a = ap.parse_args()
    seen = {}
    n = {WIND: 0, FAST: 0}
    for kind, jobs in (("wind", wind_jobs), ("fast_sim", fast_jobs)):
        if a.only and a.only != kind:
            continue
        for src, job, why in jobs():
            if job is None:
                print(f"skip {src}: {why}")
                continue
            if job["project"] not in seen:
                seen[job["project"]] = existing(job["project"])
            if job["name"] in seen.get(job["project"], ()):
                print(f"skip {src}: task '{job['name']}' already in {job['project']}")
                continue
            if a.dry_run:
                print(f"would import {src} -> {job['project']} / {job['name']}  ({len(job['pts'])} points, tags {job['tags']})")
            else:
                print(f"imported {src} -> {job['project']} / {job['name']}  {push(job)}", flush=True)
                seen[job["project"]].add(job["name"])
            n[job["project"]] += 1
    print(f"{'would import' if a.dry_run else 'imported'}: {n[WIND]} to {WIND}, {n[FAST]} to {FAST}")


if __name__ == "__main__":
    main()
