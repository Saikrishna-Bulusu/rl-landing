#!/usr/bin/env python3
"""build_report.py -- fill rig_report.template.html from the results on disk -> rig_report.html."""
import base64, glob, json, os, re
R = "/home/sai/RL_projects/rl-landing"
T = open(f"{R}/papers/report/rig_report.template.html").read()
img = lambda p, mime: f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode()
imgs = {"views": (f"{R}/shared/px4_gazebo/gz_logan/rig/cover2/rig_views.jpg", "image/jpeg"), "pixels": (f"{R}/shared/px4_gazebo/gz_logan/rig/cover2/rig_pixels.png", "image/png"),
        "detect": (f"{R}/shared/px4_gazebo/gz_logan/rig/data2/detect_vs_size.png", "image/png"), "sweep": (f"{R}/outputs/fast_sim/hunter_sweep.png", "image/png"),
        "sensing": (f"{R}/outputs/fast_sim/sensing_corridor.png", "image/png"), "strike": (f"{R}/papers/report/strike.jpg", "image/jpeg"),
        "conditions": (f"{R}/outputs/fast_sim/conditions.png", "image/png")}
for k, (p, m) in imgs.items():
    T = T.replace(f"%%IMG_{k}%%", img(p, m))
labels = [json.loads(l) for l in open(f"{R}/shared/px4_gazebo/gz_logan/rig/data2/labels.jsonl")]
T = T.replace("%%N_PICS%%", f"{len(labels):,}").replace("%%N_NEG%%", f"{sum('_neg_' in r['file'] for r in labels):,}")

# live Gazebo flights: planner with the rig vs handed the truth, same seeds
rows = []
for d, name in (("planner_rig", "rig"), ("planner_omni", "truth")):
    f = f"{R}/outputs/gazebo_rig/{d}/flights.jsonl"
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l); rg = r.get("rig") or {}
            tag = {"landed": "delivered", "landed_off_pad": "landed off pad", "struck": "struck", "timeout": "no delivery (300 s)"}.get(r["outcome"], r["outcome"])
            bad = r["outcome"] != "landed"
            rows.append(f"<tr><td>{name}</td><td class='num'>{r['seed']}</td><td><span class='tag{' bad' if bad else ''}'>{tag}</span></td>"
                        f"<td class='num'>{r['miss_m']:.1f}</td><td class='num'>{r['swoops']}</td><td class='num'>{r['closest_bird_m']:.2f}</td>"
                        f"<td class='num'>{r['t_s']:.0f}</td><td class='num'>{rg.get('found_within_40m', float('nan')) * 100:.0f}%</td>"
                        f"<td class='num'>{rg.get('dir_err_deg', float('nan')):.1f}°</td></tr>" if rg else
                        f"<tr><td>{name}</td><td class='num'>{r['seed']}</td><td><span class='tag{' bad' if bad else ''}'>{tag}</span></td>"
                        f"<td class='num'>{r['miss_m']:.1f}</td><td class='num'>{r['swoops']}</td><td class='num'>{r['closest_bird_m']:.2f}</td>"
                        f"<td class='num'>{r['t_s']:.0f}</td><td class='num'>–</td><td class='num'>–</td></tr>")
table = ("<div class='tablewrap'><table><thead><tr><th>magpie known by</th><th>seed</th><th>outcome</th><th>m off pad</th><th>dives</th>"
         "<th>closest (m)</th><th>time (s)</th><th>found &lt;40 m</th><th>direction err</th></tr></thead><tbody>" + "".join(rows) +
         "</tbody></table></div><p style='font-size:14.5px;color:var(--muted)'>Same world, magpie and seeds for both; "
         "\"found\" = frames with the magpie within 40 m where a detection points within 10° of it. Contact is 0.8 m.</p>") if rows else ""
def tally(d):
    f = f"{R}/outputs/gazebo_rig/{d}/flights.jsonl"
    rs = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    return len(rs), sum(r["outcome"] == "landed" for r in rs), sum(r["outcome"] == "struck" for r in rs)
(nr, dr, sr), (nt, dt, st_) = tally("planner_rig"), tally("planner_omni")
if nr and nt:
    table = (f"<p><b>The same three seeds, live:</b> knowing the magpie through the rig, the planner delivered {dr} of {nr} and was struck on {sr}; "
             f"handed the truth, it delivered {dt} of {nt} and was struck on {st_}. A handful of flights -- a demonstration that the whole chain "
             f"runs, not a statistic; the 4,096-flight ladders above are the measurement.</p>") + table
T = T.replace("%%GZ_TABLE%%", table)

# the rig learner: validation curve so far, and test results if a seed has finished
st = []
for log in sorted(glob.glob(f"{R}/outputs/fast_sim/rl_cond_birds_hunter_rig_s*.log")):
    L = [l for l in open(log) if re.search(r"gen +\d+", l)]
    base = next((l for l in open(log) if "baselines" in l), "")
    if L:
        m = re.search(r"\[(\S+)\] gen +(\d+).*dice +([\d.]+)% avg +([\d.]+)%", L[-1])
        if m:
            st.append(f"<li>{m.group(1).split('_')[-1]}: round {m.group(2)} of 200, validation {m.group(4)}% (average read-out), {m.group(3)}% (dice)</li>")
bl = re.search(r"'rule': ([\d.]+), 'planner': ([\d.]+), 'oracle': ([\d.]+)", base) if base else None
# finished seeds print their test results ("EVAL mean on the ladder's 4096 episodes ...") in the log
res, tests = "", []
for log in sorted(glob.glob(f"{R}/outputs/fast_sim/rl_cond_birds_hunter_rig_s*.log")):
    for l in open(log):
        m = re.search(r"\[(\S+)\] EVAL mean +on the ladder's 4096 episodes: clean ([\d.]+)%  \(vs rule ([+-][\d.]+) \+/-([\d.]+), vs planner ([+-][\d.]+) \+/-([\d.]+) pp\)", l)
        if m:
            tests.append(float(m.group(5)))
            res += (f"<li><b>{m.group(1).split('_')[-1]}</b>, 4,096 untouched test flights: {m.group(2)}% clean "
                    f"(average read-out): {float(m.group(3)):+.1f} ± {m.group(4)} points over the rule, "
                    f"{float(m.group(5)):+.1f} ± {m.group(6)} over the planner it copied</li>")
status = ("<p>The learner copied the planner, then practised against the hunter, knowing the bird only through the fast simulator's rig. "
          f"Validation baselines (2,048 flights): rule {float(bl.group(1)):.1f}%, planner {float(bl.group(2)):.1f}%, "
          f"all-knowing {float(bl.group(3)):.1f}% clean.</p>" if bl else "") + (f"<ul>{res}</ul>" if res else "") + (f"<ul>{''.join(st)}</ul>" if st else "")
T = T.replace("%%TRAIN_STATUS%%", status)
if tests:
    tile = (f"<div><b>{sum(tests) / len(tests):+.1f}</b><span>points: the rig-trained learner over the planner it copied, "
            f"4,096 test flights ({len(tests)} seed{'s' if len(tests) > 1 else ''} so far)</span></div>")
elif st:
    vals = [float(re.search(r"validation ([\d.]+)%", x).group(1)) for x in st]
    tile = (f"<div><b>{max(vals):.1f}%</b><span>validation clean deliveries for the rig-trained learner so far, vs rule "
            f"{float(bl.group(1)):.1f}% and planner {float(bl.group(2)):.1f}% (training)</span></div>") if bl else ""
else:
    tile = ""
T = T.replace("%%LEARN_TILE%%", tile)
assert "%%" not in T, re.findall(r"%%\w+%%", T)
open(f"{R}/outputs/report/rig_report.html", "w").write(T)
print(f"{R}/outputs/report/rig_report.html", f"{len(T) / 1e6:.1f} MB")
