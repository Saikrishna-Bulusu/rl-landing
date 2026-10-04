#!/usr/bin/env python3
"""build_figures.py -- paper/figs/*.png (urban/paper_figs.py) + captions -> paper/figures.html, the review page.
Numbers in the captions are read from the same result files the figures are drawn from."""
import base64
import glob
import json
import os

R = "/home/sai/RL_projects/rl-landing"
F = f"{R}/paper/figs"
css = open(f"{R}/report/rig_report.template.html").read().split("<main>")[0].replace("<title>Magpie Rig Night</title>", "<title>ACRA Figure Draft</title>")
img = lambda n: "data:image/png;base64," + base64.b64encode(open(f"{F}/{n}.png", "rb").read()).decode()
H = lambda s: json.load(open(f"{R}/urban/headroom_lidar_gps{s}.json"))["corridor"]


def learner(suf, read="sampled"):
    runs = {}
    for p in glob.glob(f"{R}/urban/rl_eyes_lidar_gps{suf}_corridor_*.json"):
        runs.update({k: v for k, v in json.load(open(p)).items() if isinstance(v, dict) and "log" in v})
    return [r["eval"][read] for r in runs.values()]


mean = lambda xs: sum(xs) / len(xs)
fog, rig, clear = learner("_fog"), learner("_birds_hunter_rig"), learner("")
hr = H("_birds_hunter_rig"); hf = H("_fog")
best_rig = hr[hr["best_obs"]]["clean"]
w = json.load(open(f"{R}/weather_report.json"))
sc = json.load(open(f"{R}/gz_logan/rig/data2/score.json"))
yolo = json.load(open(f"{R}/gz_logan/rig/data2/score_yolo.json")) if os.path.exists(f"{R}/gz_logan/rig/data2/score_yolo.json") else None
tello = json.load(open(f"{R}/tello/cage_fig.json"))["ladder"]
tl_seeds = json.load(open(f"{R}/tello/cage_fig.json"))["learner_seeds"]
why = json.load(open(f"{R}/urban/why_cond.json")) if os.path.exists(f"{R}/urban/why_cond.json") else None
over = lambda who: who["fail"]["over a person"]
fog_why = (f" In fog, flights that ended with the drone over a person fell from {over(why['fog']['planner']):.1f}% (planner) to "
           f"{mean([over(v) for k, v in why['fog'].items() if k.startswith('learner')]):.1f}% (learner)." if why and "fog" in why else "")
yolo_txt = ""
if yolo:
    op = str(yolo["operating_threshold"]); y = yolo["by_threshold"][op]
    small = lambda d, k: 100 * d["found_by_size_px"][k]["found"]
    yolo_txt = (f" The pretrained YOLO26n, fine-tuned on the same pictures, finds {100 * y['found']:.1f}% with "
                f"{y['false_alarms_per_image']:.4f} false alarms per picture (threshold {op}): "
                f"{sc['by_threshold']['0.5']['false_alarms_per_image'] / max(y['false_alarms_per_image'], 1e-9):.0f} times fewer, "
                f"but it misses more of the smallest magpies (4-7 px: {small(yolo, '4-7'):.0f}% against {small(sc, '4-7'):.0f}%; "
                f"7-10 px: {small(yolo, '7-10'):.0f}% against {small(sc, '7-10'):.0f}%).")

FIGS = [
    ("world", "The world", "Logan, QLD, rebuilt from the 2023 LiDAR survey: obstacle heights, 1,324 candidate pads, the hub "
     "and the hospital helipad, whose air ambulance arrives from the north-west.",
     "Column width. Same data as before, drawn at print resolution (5 m cells, the tallest obstacle kept)."),
    ("system", "One decision layer, three test beds", "", "Not a plot: the prompt for Gemini is in "
     "<code>diagram-prompts/11-system-overview.md</code>. It shows the learner as the one learned box, choosing among moves "
     "the planner has checked, with the hard safety rules outside it and the three test beds below."),
    ("headroom", "Where decisions have room", f"Clean deliveries on 4,096 test flights per condition: the rule, the best planner with the same "
     f"sensors, the learner (mean of three seeds; bar = seed range) and the planner told every intruder's true state. "
     f"In fog the all-knowing planner is {hf[hf['best_full']]['clean']:.1f}% against {hf[hf['best_obs']]['clean']:.1f}%: the gap is information.",
     "The one figure that states the paper: learning pays where information is missing (fog, the magpie through the rig), "
     "and adds 2 points where it isn't. The learner beats the 'all-knowing' planner in three rows: that planner knows the "
     "present, not what happens next, so it is not an upper bound."),
    ("learning", "What practice added", f"(a) Validation curves, three seeds per condition (line = mean, band = range); dashed = the planner "
     f"each learner copied. (b) Learner minus that planner on the 4,096 test flights, dice read-out, paired 95% CI. Fog "
     f"+{mean([x['vs_planner'][0] for x in fog]):.1f}, magpie through the rig +{mean([x['vs_planner'][0] for x in rig]):.1f}, clear "
     f"+{mean([x['vs_planner'][0] for x in clear]):.1f}.",
     f"Honest caveat for the magpie row: the planner it copied (shield, {mean([x['clean'] - x['vs_planner'][0] for x in rig]):.1f}%) is "
     f"not the strongest same-sensor baseline there (safe-plan, {best_rig:.1f}%). Against that one the learner's gain is "
     f"{mean([x['clean'] for x in rig]) - best_rig:+.1f} points (unpaired). Figure 3 shows both."),
    ("why", "What the learner changed", "First failure reason, the copied planner against the learner (mean of three seeds), same test "
     "flights." + fog_why, "Shows HOW the gain happens, condition by condition."),
    ("birds", "Knowing the magpie", "(a) Dives that touch the drone against how late the magpie stops correcting (rule decider, a magpie "
     "on every flight, 1,024 flights a point); the model uses 0.3 s. (b) Magpie flights struck, by how the bird is known and by decider.",
     "(a) says dodging works only against a bird that commits early, and the rig loses most of that advantage. (b) says the "
     "sensor sets the strike rate: 28% handed the truth, 39% through the rig, 45% with the drone's own two cameras; better "
     "deciders move it by a few points at most."),
    ("rig", "The six-camera rig", f"(a) The magpie's size in the rig's pictures by distance. (b) Found in a single frame by size, on drone places "
     f"no detector trained on (pictures worsened: noise, blur, JPEG). The small CNN: {100 * sc['by_threshold']['0.5']['found']:.1f}% at "
     f"{sc['by_threshold']['0.5']['false_alarms_per_image']:.3f} false alarms per picture.{yolo_txt}",
     "Coverage (some camera sees the bird in 100% of directions above the drone, 96-100% below) is now one sentence in the "
     "text rather than a panel with a cut-off axis."),
    ("gazebo", "Live in Gazebo with PX4", "Ten landings per sky (seeds 0-9). Above each bar: median miss from the pad centre. Mist and fog "
     "cost landings, all as safe aborts; wind costs none.",
     f"The pad was first read at about {w['calm']['first_read_alt_m_median']:.0f} m in every sky."),
    ("tello", "The Tello cage", f"(a) One simulated cage flight with a pole the map lacks: the rule flies into it, the planner round it using "
     f"the forward camera. (b) Outcomes on 4,096 simulated cage flights: rule {tello['rule']['delivered']:.1f}%, planner "
     f"{tello['planner']['delivered']:.1f}%, learner {tello['learner']['delivered']:.1f}% (three seeds, "
     f"{min(x['delivered'] for x in tl_seeds):.1f}-{max(x['delivered'] for x in tl_seeds):.1f}%), all-knowing "
     f"{tello['all_knowing']['delivered']:.1f}% delivered.",
     "Simulated only, with every Tello number a guess until measured in the cage. The camera does most of the work; the "
     "learner closes the last 2 points, mostly by hitting the pole less often (crashes "
     f"{tello['planner']['crash']:.1f}% for the planner, {tello['learner']['crash']:.1f}% for the learner) and timing out less."),
]

def live(tag):
    """Live Gazebo magpie flights (planner, hunter, rig in the loop): outcomes per seed."""
    f = f"{R}/gazebo_rig/{tag}/flights.jsonl"
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def live_row(name, rs):
    if not rs:
        return f"<tr><td>{name}</td><td colspan='4'>not flown yet</td></tr>"
    oc = ", ".join(f"s{r['seed']} {r['outcome'].replace('_', ' ')}" for r in rs)
    rig = [r["rig"] for r in rs if r.get("rig")]
    found = f"{100 * mean([x['found_within_40m'] for x in rig]):.0f}%" if rig else "--"
    err = f"{mean([x['dir_err_deg'] for x in rig]):.1f}&deg;" if rig else "--"
    return (f"<tr><td>{name}</td><td>{oc}</td><td class='num'>{sum(r['outcome'] == 'landed' for r in rs)}/{len(rs)}</td>"
            f"<td class='num'>{found}</td><td class='num'>{err}</td></tr>")


cnn, yl, truth = live("planner_rig"), live("planner_rig_yolo"), live("planner_omni")
live_table = ("<div class='tablewrap'><table><thead><tr><th>magpie known by</th><th>outcome per seed</th><th>delivered</th>"
              "<th>found &lt;40 m</th><th>direction err</th></tr></thead><tbody>"
              + live_row("the truth", truth) + live_row("rig, small CNN", cnn) + live_row("rig, YOLO", yl) + "</tbody></table></div>")
decision = f"""<section><h2>Decide: the paper's bird story</h2>
<p>Three ways to tell it. My pick is the first. It says the same thing as fog: learning pays where information is missing.</p>
<ol>
<li><b>Sensing first, then learning (recommended).</b> Handed the magpie's true position, a reflex rule is already as good as any
planner (Figure 3: no room). Known through the six-camera rig, strikes rise from 28% to 39% of magpie flights (Figure 6), and now
the learner finds room: it is struck on {mean([v['fail']['bird strike'] for k, v in why['rig'].items() if k.startswith('learner')]) if why and 'rig' in why else float('nan'):.1f}%
of flights against its planner's {why['rig']['planner']['fail']['bird strike'] if why and 'rig' in why else float('nan'):.1f}% (Figure 5), and delivers
+{mean([x['vs_planner'][0] for x in rig]):.1f} points over that planner, {mean([x['clean'] for x in rig]) - best_rig:+.1f} over the best same-sensor one.
The old straight-diving magpie is dropped.</li>
<li><b>Birds as a sensing study only.</b> Keep the rig, the reaction-time sweep and the sensing ladder; no learning claim for birds.
Safer if the live flights stay poor.</li>
<li><b>Everything against the hunter.</b> Also re-run "birds + fog" and "everything" with the hunter and the rig, about
11 GPU-hours, to keep those two conditions in Figure 3.</li>
</ol>
<p>What should tip it: the live Gazebo flights, the same three seeds, with no true bird position anywhere in the loop:</p>
{live_table}
<p style='font-size:15.5px;color:var(--muted)'>If YOLO goes live: the fast simulator's rig is calibrated on the small CNN, so
swapping detectors means recalibrating it (minutes), re-running the magpie ladder (about 20 min) and retraining the
rig learner (3 seeds, about 7 h, overnight).</p></section>"""


parts = [css, "<main>", "<header style='display:grid;gap:14px'><div class='eyebrow'>rl-landing · ACRA 2026 · 25 Sept · draft 1</div>",
         "<h1>The paper's figures, redrawn</h1>",
         "<p class='lede'>Every figure in its paper form: LaTeX's Computer Modern text, sized to the ACRA column (3.4 in) or page "
         "(7 in), vector PDFs in <code>paper/figs/</code>, drawn from the saved results by <code>urban/paper_figs.py</code>. "
         "The bird conditions that used the old straight-diving magpie handed the truth are left out.</p></header>", decision]
for i, (name, title, cap, note) in enumerate(FIGS, 1):
    if not os.path.exists(f"{F}/{name}.png"):
        parts.append(f"<section><h2>Figure {i} · {title}</h2><div class='note'><p>{note}</p></div></section>")
        continue
    parts.append(f"<section><h2>Figure {i} · {title}</h2><figure><img src='{img(name)}' alt='{title}'><figcaption><b>Proposed caption.</b> {cap}</figcaption>"
                 f"</figure><p style='font-size:15.5px;color:var(--muted)'>{note}</p></section>")
parts.append("</main>")
open(f"{R}/paper/figures.html", "w").write("\n".join(parts))
print(f"{R}/paper/figures.html", f"{sum(map(len, parts)) / 1e6:.1f} MB")
