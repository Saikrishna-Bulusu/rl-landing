"""Every number in acra2026: recompute from the result files or read from the code, then find it in the sentence it belongs to."""
import json, glob, re, numpy as np
R = "/home/sai/RL_projects/rl-landing"; A = f"{R}/papers/acra2026"
TEX = {f.split("/")[-1].split(".")[0]: open(f).read() for f in glob.glob(f"{A}/sections/*.tex") + glob.glob(f"{A}/tables/*.tex") + glob.glob(f"{A}/figs/*.tex")}
ok, bad = 0, []
def chk(label, file, snippet, expected, source):
    """snippet must contain {} where the expected value goes; it must appear in the file."""
    global ok
    s = snippet.format(*expected) if isinstance(expected, tuple) else snippet.format(expected)
    if s in TEX[file]: ok += 1; LEDGER.append((label, file, s, source, "OK"))
    else: bad.append((label, file, s, source)); LEDGER.append((label, file, s, source, "MISMATCH"))
LEDGER = []
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; f0 = lambda x: f"{x:.0f}"
J = lambda p: json.load(open(f"{R}/{p}"))
# ---- fast simulator ladders and learners
C = {}
for suf, name in (("", "clear"), ("_unmapped", "unmapped"), ("_fog", "fog"), ("_birds_hunter", "truth"), ("_birds_hunter_rig", "rig")):
    h = J(f"shared/fast_sim/headroom_lidar_gps{suf}.json")["corridor"]
    runs = {}
    for p in glob.glob(f"{R}/shared/fast_sim/rl_eyes_lidar_gps{suf}_corridor_*.json"):
        d = json.load(open(p)); runs.update({k: v for k, v in d.items() if isinstance(v, dict) and "log" in v})
    C[name] = dict(rule=h["rule"]["clean"], plan=h[h["best_obs"]]["clean"], ak=h[h["best_full"]]["clean"], h=h,
                   L=[r["eval"]["sampled"]["clean"] for r in runs.values()], G=[r["eval"]["sampled"]["vs_planner"] for r in runs.values()])
src = "shared/fast_sim/headroom_lidar_gps*.json (corridor) + shared/fast_sim/rl_eyes_lidar_gps*_corridor_*.json eval.sampled"
# gain over the BEST same-sensor planner: where that is the shield (the copied planner) it is the training run's own
# paired gain; where it is the safe planner, a second paired evaluation of the same learners on the same flights
PM = J("shared/fast_sim/pair_magpie.json"); UM = J("shared/fast_sim/sens/unmapped_base.json")
assert UM["best_planner"] == "safeplan" and abs(UM["shield"] - C["unmapped"]["h"]["shield_obs"]["clean"]) < 0.01 \
    and abs(UM["safeplan"] - C["unmapped"]["h"]["safeplan_obs"]["clean"]) < 0.01, "second evaluation's planners differ from the ladder's"
BEST = {k: C[k]["G"] for k in ("clear", "fog") if C[k]["h"]["best_obs"] == "shield_obs"}
BEST.update(unmapped=[UM[f"s{i}"]["vs_safeplan"] for i in range(3)], rig=[PM[f"s{i}"]["vs_safeplan"] for i in range(3)])
assert C["unmapped"]["h"]["best_obs"] == C["rig"]["h"]["best_obs"] == "safeplan_obs" and len(BEST) == 4
gm = lambda G: np.mean([x[0] for x in G])
# 95% interval over seeds AND flights: the widest per-seed flight interval and a t interval (2 dof) over the seed means, root-sum-square
gw = lambda G: (max(x[1] for x in G) ** 2 + (4.302652729911275 * np.std([x[0] for x in G], ddof=1) / np.sqrt(len(G))) ** 2) ** 0.5
assert all(len(G) == 3 for G in list(BEST.values()) + [C[k]["G"] for k in ("clear", "unmapped", "fog", "rig")])
bold = lambda c, k: f"\\textbf{{{f1(c['h'][k]['clean'])}}}" if c["h"]["best_obs"] == k else f1(c["h"][k]["clean"])
for name, row in (("clear", "Clear"), ("unmapped", "Unmapped structures"), ("fog", "Fog"), ("rig", "Magpie, camera rig")):
    c = C[name]; L = c["L"]
    chk(f"T1 {name}", "t1_ladder", row + " & {} & {} & {} & {} & {} ({}--{}) & $+{} \\pm {}$ & $+{} \\pm {}$ \\\\",
        (f1(c["rule"]), bold(c, "shield_obs"), bold(c, "safeplan_obs"), f1(c["ak"]), f1(np.mean(L)), f1(min(L)), f1(max(L)),
         f1(gm(c["G"])), f1(gw(c["G"])), f1(gm(BEST[name])), f1(gw(BEST[name]))), src + " + shared/fast_sim/pair_magpie.json + shared/fast_sim/sens/unmapped_base.json")
c = C["truth"]; chk("T1 truth", "t1_ladder", "Magpie, known exactly & {} & {} & {} & {} &", (f1(c["rule"]), bold(c, "shield_obs"), bold(c, "safeplan_obs"), f1(c["ak"])), src)
chk("best excl 0", "s5_results", "every seed's interval excludes zero in all four conditions",
    () if all(x[0] - x[1] > 0 for G in BEST.values() for x in G) else ("FALSE",), "gains over the best planner, per seed")
chk("gain clear abstract", "s0_abstract", "the learner beats the best same-sensor planner by {} points of clean deliveries in clear weather", f1(gm(BEST["clear"])), src)
chk("gain rig abstract", "s0_abstract", "It gains {} points against a swooping magpie", f1(gm(BEST["rig"])), "shared/fast_sim/pair_magpie.json")
chk("gain fog abstract", "s0_abstract", "and {} in fog", f1(gm(BEST["fog"])), src)
chk("span clear", "s5_results", "without learning span {} points, from the rule's {}\\% to the privileged planner's {}\\%",
    (f1(C["clear"]["ak"] - C["clear"]["rule"]), f1(C["clear"]["rule"]), f1(C["clear"]["ak"])), src)
chk("unmapped cost", "s5_results", "Unmapped structures cost the rule {} points and leave the privileged planner only {} points",
    (f1(C["clear"]["rule"] - C["unmapped"]["rule"]), f1(C["unmapped"]["ak"] - C["unmapped"]["plan"])), src)
chk("fog gap", "s5_results", "planner reaches {}\\% and the privileged planner {}\\%, a gap of {} points",
    (f1(C["fog"]["plan"]), f1(C["fog"]["ak"]), f1(C["fog"]["ak"] - C["fog"]["plan"])), src)
chk("truth", "s5_results", "({}\\% against {}\\% and {}\\%)", (f1(C["truth"]["rule"]), f1(C["truth"]["plan"]), f1(C["truth"]["ak"])), src)
chk("rig ladder", "s5_results", "the room returns: {}\\% for the rule, {}\\% for the best same-sensor planner and {}\\% for the privileged planner",
    (f1(C["rig"]["rule"]), f1(C["rig"]["plan"]), f1(C["rig"]["ak"])), src)
chk("best is copied", "s5_results", "The copied shield planner is also the best same-sensor planner in clear weather and fog. With unmapped structures and with the magpie seen through the rig, the safe planner does better.",
    () if set(BEST) == {"clear", "fog", "unmapped", "rig"} and all(C[k]["h"]["best_obs"] == "shield_obs" for k in ("clear", "fog")) else ("X",), src)
chk("gains text", "s5_results", "missing: {} points in fog and {} with the magpie seen through the rig. It is {} points in clear weather and {} with unmapped structures",
    tuple(f1(gm(BEST[k])) for k in ("fog", "rig", "clear", "unmapped")), src + " + second evaluations")
chk("rig vs copied", "s5_results", "The gain of {} points over the copied planner is larger", f1(gm(C["rig"]["G"])), src)
chk("beats privileged", "s5_results", "In {} conditions the learner also beats the privileged planner",
    {3: "three"}.get(sum(np.mean(C[k]["L"]) > C[k]["ak"] for k in ("clear", "unmapped", "fog", "rig")), "X"), src)
W = J("shared/fast_sim/why_cond.json"); m = lambda c, k: np.mean([W[c][f"learner_s{i}"]["fail"][k] for i in range(3)])
chk("rig caveat", "s5_results", "reaches only {}\\% there, below the safe planner's {}\\%.", (f1(W["rig"]["planner"]["clean"]), f1(C["rig"]["h"]["safeplan_obs"]["clean"])), "shared/fast_sim/why_cond.json + ladders")
chk("fog over", "s5_results", "fell from {}\\% for the copied planner to {}\\% for the learner", (f1(W["fog"]["planner"]["fail"]["over a person"]), f1(m("fog", "over a person"))), "shared/fast_sim/why_cond.json")
chk("fog 30m", "s5_results", "within 30~m of a person rose from {}\\% to {}\\%", (f1(W["fog"]["planner"]["fail"]["people 30 m"]), f1(m("fog", "people 30 m"))), "shared/fast_sim/why_cond.json")
chk("rig strikes", "s5_results", "strikes fell from {}\\% of all flights to {}\\%", (f1(W["rig"]["planner"]["fail"]["bird strike"]), f1(m("rig", "bird strike"))), "shared/fast_sim/why_cond.json")
chk("clear timeouts", "s5_results", "on {}\\% of flights against the planner's {}\\%", (f1(m("clear", "timeout, people at pad")), f1(W["clear"]["planner"]["fail"]["timeout, people at pad"])), "shared/fast_sim/why_cond.json")
S = {k: J(f"shared/fast_sim/{f}")["corridor"] for k, f in (("truth", "headroom_lidar_gps_birds_hunter.json"), ("rig", "headroom_lidar_gps_birds_hunter_rig.json"),
     ("half", "headroom_lidar_gps_birds_hunter_rig_half.json"), ("cams", "headroom_lidar_gps_birds_cameras_hunter.json"))}
chk("magpie sensing", "s5_results", "{}\\% of magpie flights end in a strike when the drone knows the magpie exactly, and {}\\% through the rig. The rate is {}\\% when the rig sees only half as far, and {}\\%",
    tuple(f1(S[k]["rule"]["struck_magpie"]) for k in ("truth", "rig", "half", "cams")), "shared/fast_sim/headroom_*birds*.json struck_magpie")
chk("magpie planners", "s5_results", "struck on {}\\% of magpie flights and the privileged planner on {}\\%",
    (f1(S["rig"][S["rig"]["best_obs"]]["struck_magpie"]), f1(S["rig"][S["rig"]["best_full"]]["struck_magpie"])), "shared/fast_sim/headroom_lidar_gps_birds_hunter_rig.json")
D = J("gnss_denied/results/geo_flights_2026-09-27.json")["detector_retrain"]
chk("detector2", "s5_results", "Before the photographic renders, it boxed tree crowns and roofs {} times per picture, and after them {} times.",
    (f2(D["photo_ground"]["old"]["false_per_picture"]), f"{D['photo_ground']['new']['false_per_picture']:.3f}"), "gnss_denied/results/geo_flights_2026-09-27.json detector_retrain")
# ---- wind
lad = lambda f: {r["w10"]: r for r in json.load(open(f))}
Rl = lad(f"{R}/conditions/wind/runs/rule_ladder.json"); F = [lad(f) for f in glob.glob(f"{R}/conditions/wind/runs/fly_hp_s*_best_ladder.json")]
S2 = [lad(f) for f in glob.glob(f"{R}/conditions/wind/runs/stage2_c100h100g_s*_best_ladder.json")]; U = lad(f"{R}/conditions/wind/runs/stage2_s0_best_ladder.json")
S2old = [lad(f) for f in glob.glob(f"{R}/conditions/wind/runs/stage2_c100_s*_best_ladder.json")]     # the earlier decider: discount 0.997, hard -30
assert len(S2) == len(S2old) == 3
med = lambda L, w, k: np.median([100 * x[w][k] for x in L]); rng = lambda L, w, k: (min(100 * x[w][k] for x in L), max(100 * x[w][k] for x in L))
cell = lambda l, c: f"{f0(l) if l == 100 else f1(l)} ({f1(c)})"
src = "conditions/wind/runs/*_ladder.json (1024 flights per wind)"
chk("T2 rule", "t2_wind", "Rule & {} & {} & {} \\\\", tuple(cell(100 * Rl[w]["landed"], 100 * Rl[w]["crashed"]) for w in (12., 15., 18.)), src)
chk("T2 fly", "t2_wind", "Flying learner ({}) & {} & {} & {} \\\\", (len(F),) + tuple(cell(med(F, w, "landed"), med(F, w, "crashed")) for w in (12., 15., 18.)), src)
chk("T2 dec", "t2_wind", "Crash-priced decider ({}) & {} & {} & {} \\\\", (len(S2),) + tuple(cell(med(S2, w, "landed"), med(S2, w, "crashed")) for w in (12., 15., 18.)), src)
assert all(Rl[w]["landed"] == 1.0 for w in (0., 3., 6., 9., 12.)); ok += 1
chk("rule text", "s5_results", "up to a W10 of 12~m/s, {}\\% at 15~m/s and {}\\% at 18~m/s", (f1(100 * Rl[15.]["landed"]), f1(100 * Rl[18.]["landed"])), src)
chk("abstract wind", "s0_abstract", "lands {}\\% of flights at 15~m/s against the rule's {}\\%", (f1(med(F, 15., "landed")), f1(100 * Rl[15.]["landed"])), src)
chk("fly text", "s5_results", "It lands {}\\% at 15~m/s ({} to {}\\% over eight seeds) and {}\\% at 18~m/s ({} to {}\\%)",
    (f1(med(F, 15., "landed")),) + tuple(f1(x) for x in rng(F, 15., "landed")) + (f1(med(F, 18., "landed")),) + tuple(f1(x) for x in rng(F, 18., "landed")), src)
chk("fly crashes", "s5_results", "It crashes on at most {}\\% of flights at any speed up to 20~m/s", f1(max(100 * x[w]["crashed"] for x in F for w in x)), src)
chk("dec text", "s5_results", "It lands {}\\% at 18~m/s ({} to {}\\%) but crashes on {}\\% ({} to {}\\%), and at 20~m/s it lands {}\\% and crashes on {}\\%",
    (f1(med(S2, 18., "landed")),) + tuple(f1(x) for x in rng(S2, 18., "landed")) + (f1(med(S2, 18., "crashed")),) + tuple(f1(x) for x in rng(S2, 18., "crashed"))
    + (f1(med(S2, 20., "landed")), f1(med(S2, 20., "crashed"))), src)
chk("unpriced", "s5_results", "One decider priced at $-15$ and $-10$ landed {}\\% and crashed on {}\\%", (f1(100 * U[18.]["landed"]), f1(100 * U[18.]["crashed"])), "conditions/wind/runs/stage2_s0_best_ladder.json")
chk("old priced", "s5_results", "$-100$ for a crash and $-30$ for a hard touchdown gave a median of {}\\% landed at 18~m/s and {}\\% crashed",
    (f1(med(S2old, 18., "landed")), f1(med(S2old, 18., "crashed"))), "conditions/wind/runs/stage2_c100_s*_best_ladder.json")
_old = lad(f"{R}/conditions/wind/runs/stage2_c100h100_s1_best_ladder.json")
chk("pulsing mjx", "s5_results", "trained with the 0.997 discount and $-100$ for a hard touchdown, landed {}\\% at 18~m/s in MJX but alternated between climbing and descending", f1(100 * _old[18.]["landed"]),
    "conditions/wind/runs/stage2_c100h100_s1_best_ladder.json")
# the decider's reward, from its checkpoints and the code (land_env.step, ppo.py GAMMA)
_meta = [json.loads(str(np.load(f)["meta"])) for f in sorted(glob.glob(f"{R}/conditions/wind/runs/stage2_c100h100g_s*_best.npz"))]
assert len(_meta) == 3 and all((m_["crash_r"], m_["hard_r"], m_["shape_g"], m_["flip_r"]) == (-100.0, -100.0, 0.9998, 0.0) for m_ in _meta), "decider prices/discount"
_le, _pp = open(f"{R}/conditions/wind/land_env.py").read(), open(f"{R}/conditions/wind/ppo.py").read()
chk("wind reward", "s4_method", "A crash and a hard or off-pad touchdown each cost ${}$", f0(_meta[0]["crash_r"]), "conditions/wind/runs/stage2_c100h100g_s*_best.npz meta (all three asserted above)")
chk("wind discount", "s4_method", "with the learner's own discount of {}. With {}, which we first used, hovering near the start earned about as much as landing",
    (re.search(r"GAMMA, LAM, CLIP, VF, ENT, LR, MAX_GN = ([\d.]+)", _pp).group(1), re.search(r"SHAPE_G = ([\d.]+)", _le).group(1)), "conditions/wind/ppo.py GAMMA, land_env.py SHAPE_G default")
chk("more than flyer", "s5_results", "At 18 and 20~m/s the crash-priced decider lands more flights than the flying learner and crashes more often",
    () if all(med(S2, w, "landed") > med(F, w, "landed") and med(S2, w, "crashed") > med(F, w, "crashed") for w in (18., 20.)) else ("FALSE",), src)
# ---- wind model checks (the README tables were written from x500_mjx.py --check output)
RM = open(f"{R}/conditions/wind/README.md").read()
for s_ in ("0.29/0.85/1.51 s | 0.28/0.80/1.52 s", "10.36 / 10.43 (9.2 m/s), 22.48 / 22.6 (15.8 m/s)", "2.584 kg"):
    (LEDGER.append(("wind README", "-", s_, "conditions/wind/README.md", "OK")) if s_ in RM else bad.append(("wind README", "-", s_, "")))
chk("step", "s3_stack", "63\\% of the response takes {}~s in Gazebo and {}~s in MJX", ("0.85", "0.80"), "conditions/wind/README.md step table")
chk("hover", "s3_stack", "steady 15.8~m/s wind, the drone tilts {}$^\\circ$ in Gazebo and {}$^\\circ$ in MJX", (f1(22.48), "22.6"), "conditions/wind/README.md hover")
# ---- Gazebo weather
def sky(s):
    rows = [json.load(open(f)) for f in glob.glob(f"{R}/outputs/weather_runs/{s}/m_*.json")]; rows = [r[0] if isinstance(r, list) else r for r in rows]
    land = [r.get("miss_m", r.get("miss")) for r in rows if r["outcome"] == "landed"]; return len(land), len(rows), np.median(land), rows
for s, label in (("calm", "Calm"), ("breeze", "Breeze, 3~m/s"), ("gusty", "Gusty, 7~m/s, gusts near 10~m/s"), ("mist_nocommit", "Mist, before commit-on-lock"),
                 ("mist", "Mist, with commit-on-lock"), ("fog_nocommit", "Fog, before commit-on-lock"), ("fog", "Fog, with commit-on-lock")):
    assert sky(s)[1] == 10, s                                               # ten flights in each set
NW = {0: "no", 1: "one", 3: "three"}; SK = ("calm", "breeze", "gusty", "mist", "fog")    # the five skies as finally flown
chk("weather wind", "s6_gazebo", "Wind of 7~m/s with gusts to about 10~m/s cost {} landing. Mist and thick fog cost {} and {} landings.",
    tuple(NW.get(10 - sky(s)[0], "X") for s in ("gusty", "mist_nocommit", "fog_nocommit")), "outputs/weather_runs/{gusty,mist_nocommit,fog_nocommit}")
chk("weather text", "s6_gazebo", "With commit-on-lock, mist and fog both landed {} of 10, and the median miss was {} to {}~m in every sky.",
    ("10" if sky("mist")[0] == sky("fog")[0] == 10 else "X", f2(min(sky(s)[2] for s in SK)), f2(max(sky(s)[2] for s in SK))), "outputs/weather_runs/<sky>/m_*.json")
assert sorted(J("conditions/weather/weather_report.json")) == sorted(SK)
ab = [r.get("miss_m", r.get("miss")) for r in sky("fog_nocommit")[3] if r["outcome"] == "abort"]
chk("abort", "s6_gazebo", "once only {}~m from its own pad", f2(min(ab)), "outputs/weather_runs/fog_nocommit")
wr = J("conditions/weather/weather_report.json"); fr = [wr[k]["first_read_alt_m_median"] for k in wr]
chk("first read", "s6_gazebo", "first read at about {}~m in every sky", f0(np.mean(fr)) if max(fr) - min(fr) < 1 else "RANGE", "conditions/weather/weather_report.json first_read_alt_m_median")
# ---- crane
rows = [json.loads(l) for l in open(f"{R}/outputs/gazebo_voxel/set10c/flights.jsonl")]
sv = [r for r in rows if r["decider"] == "planner" and not r["voxel"]]; vx = [r for r in rows if r["decider"] == "planner" and r["voxel"]]; l3 = [r for r in rows if r["decider"] == "rule"]
chk("crane survey", "s6_gazebo", "crashed into the crane in {} of {} flights", (sum(r["outcome"] == "crashed" for r in sv), len(sv)), "outputs/gazebo_voxel/set10c/flights.jsonl")
chk("crane voxel", "s6_gazebo", "passed all ten but came within {}~m", f2(min(r["closest_m"]["crane"] for r in vx)) if all(r["outcome"] == "passed" for r in vx) and len(vx) == 10 else "X", "set10c")
chk("crane local3d", "s6_gazebo", "never closer than {}~m (median {}~m)", (f2(min(r["closest_m"]["crane"] for r in l3)), f2(np.median([r["closest_m"]["crane"] for r in l3]))), "set10c")
# ---- live magpie
lm = lambda d: sum(json.loads(l)["outcome"] == "landed" for x in ("", "_gate", "_final") for l in open(f"{R}/outputs/gazebo_rig/{d}_rig_v2l{x}/flights.jsonl"))
chk("live magpie", "s6_gazebo", "The planner and a learner trained with that rig's model each landed {} of {} flights", (lm("planner"), 9) if lm("planner") == lm("learner") else ("DIFF", 0), "outputs/gazebo_rig/*_rig_v2l*/flights.jsonl")
# ---- GNSS-free
G = J("gnss_denied/results/geo_flights_2026-09-27.json")["flights"]
for k, label in (("geo_calm4", "Calm & Rule"), ("geo_wind10", "Wind, W10 9~m/s & Flying learner"), ("geo_obst5", "Crane on the route & Rule, local 3D planner"),
                 ("geo_magpie5", "Magpie & Learner"), ("geo_magpie6", "Magpie, up camera added & Learner"), ("mission_example_pad4", "Mission to another pad & Rule")):
    g = G[k]["geoanchor"]; l = G[k]["landing"]
    chk(f"T4 {k}", "t4_gnssfree", label + " & {} & {} / {} & {} / {} & {} \\\\", (f0(g["denied_s"]), f2(g["fix_err_m"]["median"]), f2(g["fix_err_m"]["p90"]),
        f2(g["est_err_m"]["median"]), f2(g["est_err_m"]["p90"]), f2(l.get("miss", l.get("miss_m")))), f"gnss_denied/results/geo_flights_2026-09-27.json {k}")
ks = ("geo_calm4", "geo_wind10", "geo_obst5", "geo_magpie5", "geo_magpie6", "mission_example_pad4")
v = lambda f: [f(G[k]) for k in ks]
fm = v(lambda x: x["geoanchor"]["fix_err_m"]["median"]); fp = v(lambda x: x["geoanchor"]["fix_err_m"]["p90"]); fw = v(lambda x: x["geoanchor"]["fix_within_10m_pct"])
em = v(lambda x: x["geoanchor"]["est_err_m"]["median"]); ms = v(lambda x: x["landing"].get("miss", x["landing"].get("miss_m"))); dn = v(lambda x: x["geoanchor"]["denied_s"])
chk("gnss ranges", "s6_gazebo", "the median fix error was {} to {}~m, the 90th percentile {} to {}~m, and {} to {}\\% of fixes fell within 10~m. PX4's fused position had a median error of {} to {}~m",
    (f1(min(fm)), f1(max(fm)), f1(min(fp)), f1(max(fp)), f1(min(fw)), f1(max(fw)), f1(min(em)), f1(max(em))), "gnss_denied/results/geo_flights_2026-09-27.json")
chk("gnss miss", "s6_gazebo", "every flight landed within {}~m of the pad center", f2(max(ms)), "gnss_denied/results/geo_flights")
chk("abstract miss", "s0_abstract", "all landed within {}~m of the pad", f2(max(ms)), "gnss_denied/results/geo_flights")
chk("concl miss", "s7_discussion", "the pad's markers guided it to within {}~m of the center", f2(max(ms)), "gnss_denied/results/geo_flights")
def land_rtf(run):
    """Simulated over wall-clock time from the hand-over to the decider on (the phase the magpie attacks in)."""
    rows = [r for r in J(f"outputs/gazebo_geo/{run}/magpie_geo.json")["rows"] if r["phase"].startswith("landing")]
    return (rows[-1]["t"] - rows[0]["t"]) / (rows[-1]["wall"] - rows[0]["wall"])
rt = [land_rtf(k) for k in ("geo_magpie4", "geo_magpie5", "geo_magpie6")]
chk("rtf", "s6_gazebo", "we slowed each magpie landing to {} of real time for the detector", "a tenth" if all(0.095 < x < 0.105 for x in rt) else f"X {rt}",
    "outputs/gazebo_geo/geo_magpie{4,5,6}/magpie_geo.json rows, landing phase")
chk("rtf2", "s6_gazebo", "so its delay cost the drone {} of the simulated time it would in real time", "a tenth" if all(0.095 < x < 0.105 for x in rt) else "X", "same")

gs = BEST["rig"]
chk("paired safeplan", "s5_results", "With the magpie, the seeds range from {} to {} points, each with a 95\\% interval of about $\\pm {}$",
    (f1(min(x[0] for x in gs)), f1(max(x[0] for x in gs)), f1(np.mean([x[1] for x in gs]))), "shared/fast_sim/pair_magpie.json")
cs_extra = [("conditions/wind/land_env.py", r"jnp.where\(h > (8\.0), -2\.5, -0\.8\)", "8.0"), ("shared/px4_gazebo/gz_learner.py", r"MPC_Z_VEL_MAX_DN\", ([\d.]+)", None)]
SD = {f: json.load(open(f)) for f in glob.glob(f"{R}/shared/fast_sim/sens/*.json")}
SD = {f: d for f, d in SD.items() if d["cond"] in ("clear", "fog", "rig")}   # unmapped_base.json is Table 1's second evaluation, not a variation
_b = {c: SD[f"{R}/shared/fast_sim/sens/{c}_base.json"] for c in ("clear", "fog", "rig")}
_mg = lambda d, k="shield": np.mean([d[f"s{i}"][f"vs_{k}"][0] for i in range(3)])
_dr = [abs(_b[d["cond"]][k] - d[k]) for d in SD.values() if d["label"] != "base" for k in ("rule", "shield", "safeplan")]
chk("sens", "s7_discussion", "We changed each assumption alone, {} variations in all", sum(d["label"] != "base" for d in SD.values()), "shared/fast_sim/sens/*.json")
chk("sens2", "s7_discussion", "moved by up to {} points, yet the learner beat the planner it copied in every variation and every seed",
    f0(max(_dr)) if all(d[f"s{i}"]["vs_shield"][0] - d[f"s{i}"]["vs_shield"][1] > 0 for d in SD.values() for i in range(3)) else "X", "shared/fast_sim/sens/*.json")
chk("sens3", "s7_discussion", "The gain stayed at least {} points in fog and with the magpie and at most {} in clear weather",
    (f1(min(_mg(d) for d in SD.values() if d["cond"] != "clear")), f1(max(_mg(d) for d in SD.values() if d["cond"] == "clear"))), "shared/fast_sim/sens/*.json")
chk("sens4", "s7_discussion", "the learner fell {} points below the safe planner", f1(-_mg(SD[f"{R}/shared/fast_sim/sens/clear_tmax200.json"], "safeplan")), "shared/fast_sim/sens/clear_tmax200.json")
_vb = {f: _mg(d, d["best_planner"]) for f, d in SD.items() if d["label"] != "base"}
assert [f.split("/")[-1] for f, v in _vb.items() if v < 0] == ["clear_tmax200.json"], "the 200 s limit must be the only variation below the best planner"
chk("sens5", "s7_discussion", "Against the best same-sensor planner it stayed at least {} points in fog and {} with the magpie. In one clear-weather variation, a 200~s limit",
    tuple(f1(min(v for f, v in _vb.items() if SD[f]["cond"] == c)) for c in ("fog", "rig")), "shared/fast_sim/sens/*.json vs the better of shield and safe planner")
# ---- constants read from code
def const(path, pat):
    m_ = re.search(pat, open(f"{R}/{path}").read()); return m_.group(1) if m_ else None
cs = [("shared/fast_sim/urban_env.py", r"TAU_XY, TAU_Z = ([\d.]+, [\d.]+)", "0.69, 0.23"), ("shared/fast_sim/urban_env.py", r"PAD_M = ([\d.]+)", "1.80"),
      ("shared/fast_sim/urban_env.py", r"DT, T_MAX, FIX_DT = ([\d.]+, [\d.]+)", "0.25, 300.0"), ("shared/fast_sim/urban_env.py", r"CEIL, PEOPLE_R, HLS_R = ([\d., ]+)\n", "120.0, 30.0, 1400.0"),
      ("shared/fast_sim/urban_env.py", r"NMAC_H, NMAC_V, WC_H, WC_V = ([\d.]+, [\d.]+)", "152.4, 30.5"), ("shared/fast_sim/urban_env.py", r"OVER_R, DRONE_SEP, GRACE = [\d.]+, ([\d.]+)", "30.0"),
      ("shared/fast_sim/urban_env.py", r"P_HELI, P_PEOPLE = ([\d.]+)", "0.5"), ("shared/fast_sim/urban_env.py", r"N_DRONE, N_PEOPLE, N_CAR, W = (\d+)", "6"),
      ("shared/fast_sim/urban_env.py", r"FOG_Z, FOG_SEE = \(([\d.]+), [\d., ]+?([\d.]+)\), \(([\d.]+), [\d., ]+?([\d.]+)\)", None),
      ("shared/fast_sim/urban_env.py", r"A_TURN, T_COMMIT, DIVE_R, DIVE_S = ([\d.]+, [\d.]+)", "20.0, 0.3"), ("shared/fast_sim/urban_env.py", r"HIT_BIRD, D_BIRD = [\d.]+, ([\d.]+)", "40.0"),
      ("shared/fast_sim/urban_env.py", r'"SENS_LEAVE", "(\d+,\d+)"', "30,240"), ("shared/fast_sim/urban_env.py", r"P_SWOOP = ([\d.]+)", "0.5"),
      ("shared/fast_sim/urban_rl.py", r"N_ACT, HOLD, GAMMA, LAM = (\d+, \d+)", "14, 2"), ("shared/fast_sim/urban_rl.py", r"VAL_N, VAL_SEED = (\d+)", "2048"),
      ("shared/fast_sim/urban_agents.py", r"H, REPLAN, PROG_S, GAMMA = (\d+, \d+, \d+)", "20, 2, 5"), ("shared/fast_sim/urban_headroom.py", r"return 100 \* d\.mean\(\)\.item\(\), (196) \* se", "196"),
      ("shared/fast_sim/local_planner.py", r"Z_PAD, R_PAD, C_CLEAR, W_CLEAR = [\d.]+, ([\d.]+, [\d.]+)", "2.0, 5.0"), ("shared/fast_sim/local_planner.py", r"(2 s -> 17 ms)", "2 s -> 17 ms"),
      ("shared/px4_gazebo/px4_mission.py", r"LOCK_FRAMES, V_BLIND = (\d+, [\d.]+)", "5, 0.5"), ("gnss_denied/geo_cruise.py", r"ALT, V, DESCEND_AT, H_READ, HOVER_Z = ([\d.]+)", "75.0"),
      ("gnss_denied/geo_cruise.py", r"HEALTHY = (\d+)\s+# accepted fixes in the last (\d+) s", None), ("conditions/wind/land_env.py", r"T_MAX = ([\d.]+)", "90.0"),
      ("conditions/wind/land_env.py", r"PAD_R, R_DRONE, Z_TOUCH = ([\d.]+, [\d.]+)", "0.75, 0.35"), ("conditions/wind/land_env.py", r"V_SINK_OK, V_SLIDE_OK = ([\d.]+, [\d.]+)", "1.0, 0.5"),
      ("conditions/wind/land_env.py", r"CRASH_R, HARD_R = (-[\d.]+)", "-15.0"), ("conditions/wind/ppo.py", r"W_START, W_STEP, W_TOP, W_OK, STUCK = ([\d.]+, [\d.]+)", "4.0, 1.0"),
      ("conditions/wind/queue/overnight_27sep.sh", r"(--w-ok 0\.95 --envs 4096)", "--w-ok 0.95 --envs 4096"), ("conditions/wind/queue/overnight_27sep.sh", r'FH="--no-decide --hp-tau (\d+)', "4"),
      ("conditions/wind/queue/overnight_27sep.sh", r"(--crash-r -100 --hard-r -30)", "--crash-r -100 --hard-r -30"),
      ("conditions/wind/queue/wind_gz2_30sep.sh", r"(--crash-r -100 --hard-r -100 --shape-g 0\.9998)", "--crash-r -100 --hard-r -100 --shape-g 0.9998")]

# ---- Gazebo wind ladder, raw flights (outputs/gazebo_wind/ladder_rule/flights.jsonl)
import collections
GW = [json.loads(l) for l in open(f"{R}/outputs/gazebo_wind/ladder_rule/flights.jsonl")]
gl = collections.defaultdict(collections.Counter)
for r_ in GW: gl[(r_["decider"].split("/")[-1].replace(".npz", ""), r_["w10"])][r_["outcome"]] += 1
gz = lambda d, w: f"{gl[(d, w)]['landed']}/{sum(gl[(d, w)].values())}"
for d, row in (("rule", "Rule"), ("fly_hp_s0_best", "Flying learner, s.~0"), ("fly_hp_s1_best", "Flying learner, s.~1"), ("stage2_c100h100g_s0_best", "Crash-priced decider, s.~0")):
    c18 = gl[(d, 18.0)]["crashed"]
    chk(f"T2 gz {d}", "t2_wind", row + " & {} & {} & {}" + (f" ({c18})" if c18 else "") + " \\\\", (gz(d, 12.0), gz(d, 15.0), gz(d, 18.0)), "outputs/gazebo_wind/ladder_rule/flights.jsonl")
by = {(r_["decider"].split("/")[-1].replace(".npz", ""), r_["w10"], r_["episode"]): r_["outcome"] for r_ in GW}
eps = [e for (d, w, e) in by if d == "rule" and w == 18.0]
plus = sum(by[("stage2_c100h100g_s0_best", 18.0, e)] == "landed" and by[("rule", 18.0, e)] != "landed" for e in eps)
minus = sum(by[("stage2_c100h100g_s0_best", 18.0, e)] != "landed" and by[("rule", 18.0, e)] == "landed" for e in eps)
chk("gz dec", "s5_results", "The crash-priced decider landed {} of {}, with no crash", (gl[("stage2_c100h100g_s0_best", 18.0)]["landed"], sum(gl[("stage2_c100h100g_s0_best", 18.0)].values()))
    if sum(gl[("stage2_c100h100g_s0_best", w)]["crashed"] for w in (9.0, 12.0, 15.0, 18.0)) == 0 else ("X", "X"), "outputs/gazebo_wind/ladder_rule")
# climb<->descend reversals of the sent vertical command (land_env.FLIP_R's test), per second of flight, from each Gazebo flight's CSV
def gz_rev(dec, w):
    out = []
    for f_ in sorted(glob.glob(f"{R}/outputs/gazebo_wind/ladder_rule/e*_w{w:g}_{dec}.csv")):
        d_ = np.genfromtxt(f_, delimiter=",", names=True); s_ = d_["vz_cmd"]
        out.append((((s_[:-1] > 0.5) & (s_[1:] < -0.3)) | ((s_[:-1] < -0.3) & (s_[1:] > 0.5))).sum() / (d_["t"][-1] - d_["t"][0]))
    assert len(out) == 20, (dec, w, len(out)); return out
_hard9 = sorted(r_["sink"] for r_ in GW if r_["decider"].split("/")[-1].replace(".npz", "") == "stage2_c100h100_s1_best" and r_["w10"] == 9.0 and r_["outcome"] == "hard")
chk("pulsing gz", "s5_results", "In Gazebo at 9~m/s it reversed {} times per second and landed {} of 20, and its {} hard landings touched down at {} to {}~m/s",
    (f1(np.median(gz_rev("stage2_c100h100_s1_best", 9.0))), gl[("stage2_c100h100_s1_best", 9.0)]["landed"], {4: "four"}.get(len(_hard9), "X"), f2(_hard9[0]), f2(_hard9[-1])),
    "outputs/gazebo_wind/ladder_rule e*_w9_stage2_c100h100_s1_best.csv (vz_cmd) + flights.jsonl sink")
_rv9 = {s_: json.loads([l for l in open(f"{R}/conditions/wind/runs/stage2_c100h100g_s{s_}_timing.out") if l.startswith("{") and '"w10": 9.0, "hold": 1,' in l][0])["reversals_per_s_median"] for s_ in range(3)}
chk("pulsing seed", "s5_results", "One seed of the crash-priced decider reversed {} times per second in MJX at 9~m/s, so we count reversals before any Gazebo flight",
    f1([v for v in _rv9.values() if v > 0.5][0]) if sum(v > 0.5 for v in _rv9.values()) == 1 and _rv9[0] == 0.0 else "X", "conditions/wind/runs/stage2_c100h100g_s*_timing.out (timing_check.py, hold 1, no delay)")
chk("flown seed rev", "s5_results", "The seed we flew reversed at most {} times per second in Gazebo",
    f2(max(max(gz_rev("stage2_c100h100g_s0_best", w)) for w in (9.0, 12.0, 15.0, 18.0))) if max(range(3), key=lambda i: np.mean(_meta[i]["val"])) == 0 else "X",
    "outputs/gazebo_wind/ladder_rule e*_stage2_c100h100g_s0_best.csv; seed 0 has the best validation mean")
chk("gz paired", "s5_results", "the decider landed {} flights that the rule did not, and the rule landed none", plus if minus == 0 else "X", "outputs/gazebo_wind/ladder_rule")
chk("gz text", "s5_results", "At 15~m/s the rule landed {} of {} flights and the two flying learners {} and {} of 20", (gl[("rule", 15.0)]["landed"], sum(gl[("rule", 15.0)].values()),
    gl[("fly_hp_s0_best", 15.0)]["landed"], gl[("fly_hp_s1_best", 15.0)]["landed"]), "outputs/gazebo_wind/ladder_rule")
assert f1(gm(BEST["clear"])) == f1(gm(BEST["unmapped"])), "the conclusion's 'elsewhere' needs clear and unmapped to agree"
chk("concl", "s7_discussion", "In simulation, against the best same-sensor planner, learning added {} points of clean deliveries in fog, {} against a magpie seen through cameras and {} elsewhere",
    (f1(gm(BEST["fog"])), f1(gm(BEST["rig"])), f1(gm(BEST["clear"]))), src + " + second evaluations")
cs += [("conditions/wind/land_env.py", r"d_wall = jax.random.uniform\(k\[1\], \(\), minval=([\d.]+), maxval=([\d.]+)\)", "2.5, 8.0"),
       ("conditions/wind/land_env.py", r"r0 = jax.random.uniform\(k\[1\], \(\), minval=([\d.]+), maxval=([\d.]+)\)", "15.0, 25.0"),
       ("conditions/wind/land_env.py", r"has_tree = jax.random.uniform\(k\[7\], \(\)\) < ([\d.]+)", "0.6"), ("conditions/wind/land_env.py", r"A_FF = ([\d.]+)", "3.0"),
       ("conditions/wind/land_env.py", r"MODES = jnp.array\(\[([-\d., ]+)\]\)", "0.0, -1.5, -0.6, 0.0, 1.0"), ("conditions/wind/land_env.py", r"(0\.04) s per decision", "0.04"),
       ("shared/px4_gazebo/gz_learner.py", r"if p\[2\] < ([\d.]+) and \(float\(np.linalg.norm\(d_land\)\) > tol", "6.0"), ("conditions/wind/ppo.py", r"W_OK, STUCK = [\d.]+, [\d.]+, [\d.]+, [\d.]+, (\d+)", "150"),
       ("gnss_denied/geo_world.py", r"(2023)-06-29 Esri Wayback capture", "2023"), ("gnss_denied/geo_world.py", r"is the (2021)-07-21 capture", "2021")]

# ---- added 30 Sept for the reviewer points: the copy before PPO, Table 5, Table 2's caption, the fast simulator's start,
# the GNSS-denied development flights, the ambulance direction
same = []
for suf in ("", "_unmapped", "_fog", "_birds_hunter_rig"):
    for p in glob.glob(f"{R}/shared/fast_sim/rl_eyes_lidar_gps{suf}_corridor_*.json"):
        d = json.load(open(p))
        same += [v["log"][0]["gen"] == -1 and v["log"][0]["eval_clean"] == d["baselines"]["planner"] for v in d.values() if isinstance(v, dict) and "log" in v]
chk("copy = planner", "s4_method", "Before PPO, the copy's top choice scored exactly the planner's rate of clean deliveries on the validation flights, in every seed and condition.",
    () if len(same) == 12 and all(same) else ("X",), "shared/fast_sim/rl_eyes_lidar_gps*_corridor_*.json log[0].eval_clean vs baselines.planner")
import subprocess
shp = json.loads(subprocess.run(["/home/sai/miniconda3/envs/mujoco_drones/bin/python", "-c",
    "import torch, json, sys; sd = [torch.load(f, map_location='cpu') for f in sys.argv[1:]]; "
    "print(json.dumps([[sd_['body.0.weight'].shape[1], sum(v.numel() for v in sd_.values())] for sd_ in sd]))",
    f"{R}/shared/fast_sim/rl_corridor_eyes_lidar_gps_s0_sampled.pt", f"{R}/shared/fast_sim/rl_corridor_eyes_lidar_gps_birds_hunter_rig_s0_sampled.pt"],
    capture_output=True, text=True, check=True).stdout)
chk("T5 inputs", "t5_learner", "Inputs & {}, magpie {}", (shp[0][0], shp[1][0]), "checkpoints body.0.weight")
chk("T5 params", "t5_learner", "Parameters & {:,}, magpie {:,}", (shp[0][1], shp[1][1]), "checkpoints, all tensors")
# 221 = own 7 + pad 5 + helipad 5 + 15 intruder slots x 7 + 14 candidates x 3 + blocked 1 + planner's view 14 x 4 (urban_rl.features)
assert 7 + 5 + 5 + 15 * 7 + 14 * 3 + 1 + 14 * 4 == shp[0][0] and shp[1][0] - shp[0][0] == 7
LOGS = ("rl_cond_base.log", "rl_cond_unmapped.log", "rl_cond_fog.log", "rl_cond_fog_s2.log", "rl_cond_birds_hunter_rig_s01.log", "rl_cond_birds_hunter_rig_s2.log")
txt = "".join(open(f"{R}/outputs/fast_sim/{f}").read() for f in LOGS)
rx = r"\[corridor_eyes_lidar_gps(?:_fog|_unmapped|_birds_hunter_rig)?_s\d\] "
dat = {m_.group(0).split("]")[0]: int(m_.group(1).replace(",", "")) for m_ in re.finditer(rx + r"copy round 2 .*?data +([\d,]+)", txt)}
tim = {m_.group(0).split("]")[0]: int(m_.group(1)) for m_ in re.finditer(rx + r"gen 199 .*?\( *(\d+) s\)", txt)}
assert len(dat) == len(tim) == 12, (len(dat), len(tim))
chk("T5 data", "t5_learner", "decisions labeled & {} to {} million", (f1(min(dat.values()) / 1e6), f1(max(dat.values()) / 1e6)), "outputs/fast_sim/rl_cond_*.log, DAgger round 2")
chk("T5 time", "t5_learner", "PPO time per seed & {} to {}~h", (f1(min(tim.values()) / 3600), f1(max(tim.values()) / 3600)), "outputs/fast_sim/rl_cond_*.log, gen 199")
# only the table's four deciders: the wind session keeps appending new deciders to the same flights.jsonl
T2_DEC = ("rule", "fly_hp_s0_best", "fly_hp_s1_best", "stage2_c100h100g_s0_best")
GW_n = collections.Counter((r_["decider"].split("/")[-1].replace(".npz", ""), r_["w10"]) for r_ in GW if r_["decider"].split("/")[-1].replace(".npz", "") in T2_DEC)
chk("T2 caption", "t2_wind", "Two rule flights, at {} and {}~m/s, stopped on a simulator fault and left no record.",
    tuple(f0(w) for (d, w), n in sorted(GW_n.items()) if n != 20 and d == "rule") if sorted(GW_n.values())[:3] == [19, 19, 20] else ("X", "X"),
    "outputs/gazebo_wind/ladder_rule/flights.jsonl (rule e13 at 15 and e12 at 18 m/s have no record)")
UE = open(f"{R}/shared/fast_sim/urban_env.py").read()
chk("start", "s3_stack", "Each flight starts {}~m up, at most {}~km from its pad in the direction of the depot.",
    ("50" if "CRUISE = 50.0" in UE else "X", "1" if "run = dist.clamp(max=1000.0)" in UE and "ang = torch.atan2(to_hub" in UE else "X"), "shared/fast_sim/urban_env.py reset()")
chk("touchdown", "s3_stack", "It models no wind and counts any touchdown within {}~m of the pad center, so it has no landing gate.",
    "1" if "on_pad = self.ever_seen & (off <= 1.0)" in UE and "wind" not in UE.lower() else "X", "shared/fast_sim/urban_env.py step()")
# GNSS-denied development flights, 26-27 Sept: every run folder by scenario, in the order flown (the reported one last)
SCN = {"calm": ["geo_calm", "geo_calm_rng", "geo_calm2", "geo_calm3", "geo_calm4"],
       "wind": ["geo_wind9", "geo_wind9b", "geo_wind9c", "geo_wind9d", "geo_wind9e", "geo_wind10"],
       "crane": ["geo_obst_baro", "geo_obst", "geo_obst2", "geo_obst3", "geo_obst4", "geo_obst5"],
       "magpie": ["geo_magpie", "geo_magpie2", "geo_magpie3", "geo_magpie4", "geo_magpie5"], "magpie_up": ["geo_magpie6"], "mission": ["mission_example_pad4"]}
import os
def outcome(run):
    """(outcome, miss in m) of one GNSS-denied run: flights.jsonl where the run wrote one, else the result line in run.log."""
    log = open(f"{R}/outputs/gazebo_geo/{run}/run.log").read()
    fj = f"{R}/outputs/gazebo_geo/{run}/flights.jsonl"
    if os.path.exists(fj):
        r_ = json.loads(open(fj).read().strip().splitlines()[-1]); return r_["outcome"], r_.get("miss", r_.get("miss_m"))
    if '"outcome"' in log:
        return re.findall(r'"outcome": "([a-z_ ]+)"', log)[-1], float(re.findall(r'"miss_m": ([\d.]+)', log)[-1])
    assert "GeoAnchor lost" in log and "GNSS restored" in log, run    # no record: the fixes were lost and GNSS came back
    return "lost fixes", None
OUT = {r_: outcome(r_) for runs in SCN.values() for r_ in runs}
mt = lambda r_: os.path.getmtime(f"{R}/outputs/gazebo_geo/{r_}/run.log")
assert all(max(runs, key=mt) == runs[-1] for runs in SCN.values()), "a reported flight is not the last of its scenario"
assert [runs[-1] for runs in SCN.values()] == list(ks[:3]) + ["geo_magpie5", "geo_magpie6", "mission_example_pad4"]
earlier = [r_ for runs in SCN.values() for r_ in runs[:-1]]; failed = [r_ for r_ in earlier if OUT[r_][0] != "landed"]
assert all(OUT[runs[-1]][0] == "landed" for runs in SCN.values())
chk("dev flights", "s6_gazebo", "Before them, and after short pipeline tests, we flew {} development flights in the same scenarios. {} failed:",
    (len(earlier), {6: "Six"}.get(len(failed), "X")), "outputs/gazebo_geo/<run>/ run.log and flights.jsonl")
chk("dev 659", "s6_gazebo", "a timeout and a landing {}~m off the pad", f0(OUT["geo_magpie2"][1]), "outputs/gazebo_geo/geo_magpie2/run.log miss_m")
kinds = sorted(OUT[r_][0] for r_ in failed)
assert kinds == sorted(["hard", "crashed", "lost fixes", "crashed", "timeout", "landed_off_pad"]), kinds
# the ambulance from any direction: earlier clear-weather runs with the visual localization error model (nav geoanchor)
RD = {w: [v["eval"]["sampled"]["vs_planner"][0] for v in J(f"shared/fast_sim/rl_eyes_lidar_{w}_0_1_2.json").values() if isinstance(v, dict) and "eval" in v] for w in ("random", "corridor")}
assert all(J(f"shared/fast_sim/rl_eyes_lidar_{w}_0_1_2.json").get("nav") in (None, "geoanchor") for w in RD) and all(len(v) == 3 for v in RD.values())
chk("any direction", "s7_discussion", "The learner's gain over its copied planner was then {} to {} points, against {} to {} from the north-west.",
    (f1(min(RD["random"])), f1(max(RD["random"])), f1(min(RD["corridor"])), f1(max(RD["corridor"]))), "shared/fast_sim/rl_eyes_lidar_{random,corridor}_0_1_2.json eval.sampled.vs_planner")
cs += [("shared/fast_sim/urban_rl.py", r"N_ACT, HOLD, GAMMA, LAM = \d+, \d+, ([\d.]+, [\d.]+)", "0.997, 0.95"),
       ("shared/fast_sim/urban_rl.py", r"def ppo_update\(net, opt, batch, epochs=(\d+), mb=(\d+), clip=([\d.]+), ent=[\d.]+, vcoef=([\d.]+)", "4, 32768, 0.2, 0.5"),
       ("shared/fast_sim/urban_rl.py", r"clip_grad_norm_\(net.parameters\(\), ([\d.]+)\)", "0.5"),
       ("shared/fast_sim/urban_rl.py", r"def fit\(net, X, Y, epochs=(\d+), mb=\d+, smooth=[\d.]+(?:, stats=None)?\):\n    opt = torch.optim.Adam\(net.parameters\(\), lr=([\de.-]+)\)", "8, 1e-3"),
       ("shared/fast_sim/urban_rl.py", r"copy_planner\(world, net, eyes, betas=\(([\d., ]+)\), seed0=20_000 \+ 100 \* seed, smooth=([\d.]+)", "1.0, 0.5, 0.0, 0.05"),
       ("shared/fast_sim/urban_rl.py", r"warm, lr, \(ent0, ent1\) = (\d+), ([\de.-]+), \(\(0.0, 0.0\) if kl else \(([\d.]+), ([\d.]+)\)\)", "5, 1e-4, 0.002, 0.0002"),
       ("shared/fast_sim/urban_rl.py", r"def __init__\(self, f, h=(\d+)\)", "256"), ("shared/fast_sim/urban_rl.py", r"eyes=False, eval_every=(\d+)", "10"),
       ("shared/fast_sim/urban_rl.py", r'ap.add_argument\("--kl", type=float, default=([\d.]+)', "0.0"),
       ("shared/fast_sim/urban_rl.py", r'ap.add_argument\("--n-env", type=int, default=(\d+)\)', "2048")]
assert all("_kl" not in k for suf in ("", "_unmapped", "_fog", "_birds_hunter_rig") for p in glob.glob(f"{R}/shared/fast_sim/rl_eyes_lidar_gps{suf}_corridor_*.json") for k in json.load(open(p)))

# ---- added 30 Sept (mock review 2)
chk("concl 7.7", "s7_discussion", "Against the planner it started from, the magpie gain was {} points.", f1(gm(C["rig"]["G"])), src)
chk("combined ci", "s4_method", "For the mean over seeds we combine the widest per-seed interval with a Student's $t$ interval over the seed means",
    () if all(gm(G) - gw(G) > 0 for G in list(BEST.values()) + [C[k]["G"] for k in ("clear", "unmapped", "fog", "rig")]) else ("SOME CROSS 0",), "Table 2 intervals")
DS = {k: {s: len(os.listdir(f"{R}/shared/px4_gazebo/gz_logan/rig_v2/yolo/{k}/images/{s}")) for s in ("train", "test")} for k in ("ds", "ds_geo")}
SC = {k: J(f"shared/px4_gazebo/gz_logan/{f}") for k, f in (("plain", "rig_v2/data/score_yolo_new.json"), ("photo", "rig_v2_geo/data/score_yolo_new.json"))}
assert SC["plain"]["images"] == DS["ds"]["test"] and SC["photo"]["images"] == DS["ds_geo"]["test"]
assert "--train --ds ds,ds_geo" in open(f"{R}/conditions/magpie/queue/yolo_photo.sh").read()          # the weights in use saw both sets
chk("det sizes", "s5_results", "fine-tuned on {:,} labeled renders over plain ground and {:,} over photographic ground.", (DS["ds"]["train"], DS["ds_geo"]["train"]), "shared/px4_gazebo/gz_logan/rig_v2/yolo/{ds,ds_geo}/images/train")
chk("detector", "s5_results", "it found {}\\% of magpies over plain ground ({} pictures) and {}\\% over photographic ground ({}), at a 0.5 threshold.",
    (f0(100 * D["plain_ground"]["new"]["found"]), DS["ds"]["test"], f0(100 * D["photo_ground"]["new"]["found"]), DS["ds_geo"]["test"]), "detector_retrain + yolo/{ds,ds_geo}/images/test")
fd = SC["photo"]["found_by_distance_m"]; near = [100 * fd[b_]["found"] for b_ in ("0-5", "5-10", "10-20")]
chk("det range", "s5_results", "It found {} to {}\\% of magpies within 20~m and {}\\% at 45 to 70~m.", (f0(min(near)), f0(max(near)), f0(100 * fd["45-70"]["found"])), "score_yolo_new.json found_by_distance_m (photo ground)")
assert 'split = lambda r: "test" if place(r) % 10 == 0' in open(f"{R}/conditions/magpie/rig_detect.py").read()      # held out by drone place
chk("sign test", "s5_results", "and the rule landed none that the decider missed. A two-sided sign test gives $p < {}$.",
    "0.001" if minus == 0 and 2 * 0.5 ** plus < 0.001 else "X", "outputs/gazebo_wind/ladder_rule, 18 m/s pairs")
chk("mjx ci", "t2_wind", "A rate from 1,024 flights has a 95\\% interval of at most $\\pm {}$ points.", f1(100 * 1.96 * 0.5 / 1024 ** 0.5), "binomial, p = 0.5")
chk("weather cp", "s6_gazebo", "10 of 10 shows only that the landing rate exceeds {}\\% at 95\\% confidence", f0(100 * 0.025 ** 0.1), "Clopper-Pearson lower bound, 10 of 10")
lat = [G[k]["geoanchor"]["latency_ms_wall_median"] for k in ks]
chk("fix latency", "s6_gazebo", "a visual fix was ready a median of {} to {}~ms after its image was taken.", (f0(min(lat)), f0(max(lat))), "gnss_denied/results/geo_flights latency_ms_wall_median")
UA = open(f"{R}/shared/fast_sim/urban_agents.py").read()
chk("candidates", "s3_stack", "The other four are landing anyway, hovering, climbing straight up and flying away from the helipad. Every candidate is capped at {}~m/s across, {}~m/s up and {}~m/s down.",
    ("8", "3", "2.5") if "C = torch.stack([v_rule, self.v_any, v_nom, rot(v_nom, 45), rot(v_nom, -45), rot(v_nom, 90),\n                         rot(v_nom, -90), rot(v_nom, 180), half, z, up, flee, hold, climb], 1)" in UA
    and "V_XY, VZ_UP, VZ_DN = 8.0, 3.0, 2.5" in UE else ("X", "X", "X"), "shared/fast_sim/urban_agents.py MPC.candidates")
chk("planner", "s4_method", "It extends each track in a straight line at its sensed velocity and replans every {}~s. Its cost adds progress over {}~s, predicted violations over {}~s, and a deadline term.",
    ("0.5", "5", "20") if "H, REPLAN, PROG_S, GAMMA = 20, 2, 5, 0.95" in UA and "# straight-line extrapolation" in UA and "DT, T_MAX, FIX_DT = 0.25, 300.0" in UE
    and 'over_t = (o["t"] + self.PROG_S + left - U.T_MAX).clamp(min=0.0)' in UA else ("X", "X", "X"), "shared/fast_sim/urban_agents.py MPC")
chk("test-picked", "s4_method", "We report the better of the two on the test flights as the best same-sensor planner.",
    () if 'best_obs = max(("shield_obs", "safeplan_obs"), key=lambda k: f(S[k]["clean"]))' in open(f"{R}/shared/fast_sim/urban_headroom.py").read() else ("X",), "shared/fast_sim/urban_headroom.py")
chk("T5 scaling", "t5_learner", "\\quad scaling & fixed constants \\\\", () if "torch.stack([o[\"pos_z\"] / 50" in open(f"{R}/shared/fast_sim/urban_rl.py").read() else ("X",), "shared/fast_sim/urban_rl.features")
_row = {"base": "As in Table~\\ref{{tab:ladder}}", "leave10_120": "Leave 10--120~s", "leave60_480": "Leave 60--480~s", "tmax200": "Limit 200~s", "tmax400": "Limit 400~s",
        "sep50": "Drones 50~m", "commit0.1": "Commit 0.1~s", "commit0.5": "Commit 0.5~s", "halfrange": "Rig range half"}
_num = lambda x: f"${x:.1f}$" if x < 0 else f"{x:.1f}"
for l_, lab in _row.items():
    cells = []
    for c_ in ("clear", "fog", "rig"):
        f_ = f"{R}/shared/fast_sim/sens/{c_}_{l_}.json"
        cells += [_num(_mg(SD[f_])), _num(_mg(SD[f_], SD[f_]["best_planner"]))] if f_ in SD else ["--", "--"]
    chk(f"T6 {l_}", "t6_sens", lab + " & " + " & ".join(c__.replace("{", "{{").replace("}", "}}") for c__ in cells) + " \\\\", (), f"shared/fast_sim/sens/*_{l_}.json")
assert sum(1 for f_ in SD if not f_.endswith("_base.json")) == 18
chk("T6 first row", "t6_sens", "The first row is a second evaluation, within {} points of Table~\\ref{{tab:ladder}}.",
    f1(max(abs(_mg(_b[c_]) - gm(C[c_]["G"])) for c_ in ("clear", "fog", "rig"))), "sens base vs training-run evaluation")

# ---- the ten repeated GNSS-denied calm flights (gnss_denied/queue/geo_calm10_30sep.sh, 30 Sept)
RP = []
for e_ in range(10):
    rr = json.load(open(f"{R}/outputs/gazebo_geo/geo_calm10_e{e_}/geo_e{e_}_w0_rule.json"))["report"]
    ff = json.loads(open(f"{R}/outputs/gazebo_geo/geo_calm10_e{e_}/flights.jsonl").read().strip().splitlines()[-1])
    assert ff["decider"] == "rule" and ff["w10"] == 0.0 and rr["gnss_restored_at"] is None and rr["sent"] > 400
    RP.append((ff["outcome"], ff["miss"], rr["denied_s"], rr["est_err_m"]["median"]))
chk("repeats", "s6_gazebo", "We then repeated the calm flight {} times on one frozen code version, changing only the landing heading. All {} landed, {} to {}~m from the pad center, with GNSS denied for {} to {}~s and PX4's median error {} to {}~m.",
    ({10: "ten"}[len(RP)], {10: "ten"}[sum(o == "landed" for o, *_ in RP)], f2(min(x[1] for x in RP)), f2(max(x[1] for x in RP)),
     f0(min(x[2] for x in RP)), f0(max(x[2] for x in RP)), f1(min(x[3] for x in RP)), f1(max(x[3] for x in RP))), "outputs/gazebo_geo/geo_calm10_e*/")
chk("repeats abstract", "s0_abstract", "and {} repeated calm flights, all landed within {}~m of the pad", ({10: "ten"}[len(RP)], f2(max(max(ms), max(x[1] for x in RP)))), "outputs/gazebo_geo/geo_calm10_e* + gnss_denied/results/geo_flights")
chk("repeats limits", "s7_discussion", "{} GNSS-denied flights in calm air but one in each harder scenario", {10: "ten"}[len(RP)], "outputs/gazebo_geo/geo_calm10_e*/")

# ---- added 30 Sept, round 4: detector timing, the pad and its figure
DL = J("conditions/magpie/results/detector_latency_2026-09-30.json")["results"]
chk("det latency", "s6_gazebo", "the detector took a median of {} to {}~ms per set of four or five pictures. The slowest set took {}~ms, inside the {}~s reflex period.",
    (f0(min(DL[k]["set_ms_median"] for k in DL)), f0(max(DL[k]["set_ms_median"] for k in DL)), f0(max(DL[k]["set_ms_max"] for k in DL)), "0.25")
    if sorted(DL) == ["4", "5"] and "DT, T_MAX, FIX_DT = 0.25" in UE else ("X",) * 4, "conditions/magpie/results/detector_latency_2026-09-30.json")
PD = open(f"{R}/shared/px4_gazebo/make_pad.py").read()
MAT = float(re.search(r"\nMAT = ([\d.]+)", PD).group(1))
MK = [tuple(float(x) for x in t) for t in re.findall(r"\((\d+),\s+([\d.]+),\s+([-+]?[\d.]+)\)", PD.split("MARKERS = [")[1].split("]")[0])]
assert len(MK) == 3 and min(MK, key=lambda m: m[1])[2] == 0.0                   # the smallest sits at the touchdown point
by_size = sorted(MK, key=lambda m: -m[1])
chk("pad", "s3_stack", "The delivery pad is a {}~m square mat carrying three ArUco markers \\cite{{garrido2014}} in a row, {}, {} and {}~m across",
    (f0(MAT),) + tuple(f2(m[1]) for m in by_size), "make_pad.py MAT, MARKERS")
for i_, sz, x in MK:     # the figure draws every marker from make_pad.py's numbers: 1 m = 0.5 cm, centered on the mat
    chk(f"pad fig {i_:.0f}", "pad", "\\fill ({:g},{:g}) rectangle ({:g},{:g});", (round(x - sz / 2, 3), -sz / 2, round(x + sz / 2, 3), sz / 2), "make_pad.py MARKERS")
chk("pad fig mat", "pad", "node[below] {{{} m mat}}", f1(MAT), "make_pad.py MAT")
chk("pad fig radii", "pad", "\\draw[orange] (0,0) circle ({});\n  \\draw[orange, dashed] (0,0) circle ({});",
    ("0.75", "1.0") if "PAD_R, R_DRONE, V_SINK_OK, V_SLIDE_OK = 0.75" in open(f"{R}/conditions/wind/gz_windland.py").read()
    and "miss < 1.0 else \"landed_off_pad\"" in open(f"{R}/shared/px4_gazebo/gz_learner.py").read() else ("X", "X"), "gz_windland.py PAD_R, gz_learner.py landed_off_pad")
chk("pad fig legend", "pad", "node[right, black] {{{} m, wind}};\n  \\draw[orange, dashed] (2.75,0.35) -- (3.35,0.35) node[right, black] {{{} m, others}}",
    ("0.75", "1"), "the same radii, named in the legend")
chk("pad fig A", "pad", "A: global approach on satellite-image fixes, median fix error {} to {} m", (f1(min(fm)), f1(max(fm))), "gnss_denied/results/geo_flights fix_err_m.median")
chk("pad fig B", "pad", "B: terminal landing on the pad's markers, touchdown within {} m", f2(max(ms)), "gnss_denied/results/geo_flights miss")
chk("pad fig gate", "pad", "GNSS off once {} fixes arrive within {} s", (const("gnss_denied/geo_cruise.py", r"HEALTHY = (\d+)"), "15")
    if "now - f[\"t_wall\"] < 15.0" in open(f"{R}/gnss_denied/geo_cruise.py").read() else ("X", "X"), "gnss_denied/geo_cruise.py HEALTHY")
chk("pad fig lock", "pad", "pad identity read {} times, then commit", {"5": "five"}.get(const("shared/px4_gazebo/px4_mission.py", r"LOCK_FRAMES, V_BLIND = (\d+)"), "X"), "shared/px4_gazebo/px4_mission.py LOCK_FRAMES")

for path, pat, exp in cs:
    m_ = re.search(pat, open(f"{R}/{path}").read())
    got = ", ".join(m_.groups()) if m_ else None
    if exp is None: LEDGER.append(("code", path, pat, got, "READ")); print("   read", path, "->", got); continue
    (LEDGER.append(("code", path, exp, path, "OK")) if got == exp else bad.append(("code", path, f"{pat} expected {exp} got {got}", path)))
print(f"OK {ok + sum(1 for x in LEDGER if x[0] in ('code','wind README') and x[-1]=='OK')}   MISMATCH {len(bad)}")
for b in bad: print("  BAD:", b)
json.dump(LEDGER, open(f"{A}/number_audit.json", "w"), indent=1)
