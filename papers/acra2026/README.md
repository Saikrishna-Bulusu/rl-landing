# acra2026/: ACRA 2026 paper draft

"Where Learning Helps a Delivery Drone Land: A Hierarchical Stack Evaluated in Fast Simulation and GNSS-Denied PX4/Gazebo Flights"
(mock review 2's title with GNSS-Denied, Sai 30 Sept).
Authors as in the journal manuscript. First draft 29 Sept 2026, for the meeting with Nabin. Revised 30 Sept for a mock
review: title in the reviewer's wording, gains over the best same-sensor planner beside gains over the copied one,
"privileged planner" for the planner told every intruder's future, the learner's settings table, and the GNSS-denied
development flights disclosed. Revised again on 30 Sept (round 4): the pad figure and a terminal pad localization
paragraph, the two localization stages kept apart (satellite-image fixes a few meters off, markers for the last meters),
a "where our results sit" paragraph in Related Work (`LITERATURE_COMPARISON.md`), the detector's timing, and three
floats dropped to make room (the weather table, the magpie sensing figure, the GNSS-denied error plot).

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

This produces 10 pages, including references (page 10's right column about 6 lines short). ACRA allows 6 to 10, so
anything added now needs a cut elsewhere. `python3 check_numbers.py` re-derives every number, including those drawn in
`figs/pad.tex`, and must print `MISMATCH 0` (172 checks).

## Template

`acra.sty`, `named.bst` and `template/` (acra.tex, acra.rtf, acra.odt) come from the ARAA template folder linked on
araa.asn.au/conference/acra-2026, downloaded 29 Sept 2026. The page is US letter with two columns, and citations are
author-year. The abstract is at most 200 words. This one is 199.

## Layout

| File | Contents |
|---|---|
| `sections/s0_abstract.tex` … `s7_discussion.tex` | pulled in by `\input` in `main.tex`, in order (s7 also holds the conclusion) |
| `tables/t1_ladder.tex`, `t2_wind.tex`, `t4_gnssfree.tex` | pulled in by s5 and s6 (`tables/unused/t3_weather.tex` is out of the paper since round 4; its numbers are in the s6 text) |
| `tables/t5_learner.tex` | the learner's inputs and training settings, pulled in by s4 (it prints as Table 1) |
| `tables/t6_sens.tex` | the 18 sensitivity variations, gains over the copied and the best planner, pulled in by s7; written from `shared/fast_sim/sens/*.json` (the cells are checked) |
| `refs.bib` | every entry checked against Crossref, arXiv, PMLR or the publisher page on 29 Sept |

Figures in the paper, in print order. Every panel is its own image or picture with its label below it, and every
caption is one sentence on one line (Sai, 30 Sept, the journal's rule); anything more goes into the figure or the text.

| Label | Figure | Made by |
|---|---|---|
| stack | the four layers and three test beds | `figs/stack.tex` (TikZ) |
| pad | (a) the pad from above, (b) the two localization stages | `figs/pad.tex` (TikZ; sizes and positions from `make_pad.py` MARKERS and MAT, radii from `gz_windland.py` PAD_R and `gz_learner.py`, all checked) |
| world | Logan from the 2023 LiDAR | `make_figs.py` -> `shared/fast_sim/paper_figs.py world` |
| learning | (a) validation curves, (b) test gain over the copied planner with Table 2's combined 95% interval | `make_figs.py` -> `shared/fast_sim/paper_figs.py learning` -> `learning_a.pdf`, `learning_b.pdf` |
| gazebo | four Gazebo frames | `make_figs.py gazebo_frames()` -> `gazebo_{a,b,c,d}.jpg` from `figs/src/*_full.png` (plain Gazebo worlds, no photo floor) |

`make_figs.py` still draws `magpie` and `gnssfree`, which left the paper in round 4.

## Where every number comes from

| Numbers | Source |
|---|---|
| Table 2 rule, planners and privileged planner | `shared/fast_sim/headroom_lidar_gps{,_unmapped,_fog,_birds_hunter,_birds_hunter_rig}.json`, corridor, `clean` of `rule`, `best_obs` and `best_full` |
| Table 2 learner and gain | `shared/fast_sim/rl_eyes_lidar_gps*_corridor_*.json`, `eval.sampled.clean` and `vs_planner` [gain, 95% half-width] |
| Table 2 gain over the best same-sensor planner | clear and fog: the same as over the copied shield (it is the best). Unmapped: `shared/fast_sim/sens/unmapped_base.json` (`python sens.py unmapped base`, 30 Sept). Magpie rig: `shared/fast_sim/pair_magpie.json`. Both `vs_safeplan` |
| Table 2 intervals | per seed: `vs_planner` / `vs_safeplan` [gain, half-width]; combined = sqrt(widest per-seed half-width^2 + (t_0.975,2 * sd of the 3 seed gains / sqrt 3)^2) |
| detector sizes and recall by range | `shared/px4_gazebo/gz_logan/rig_v2/yolo/{ds,ds_geo}/images/{train,test}` (split by drone place, `rig_detect.load`), `shared/px4_gazebo/gz_logan/rig_v2{,_geo}/data/score_yolo_new.json` |
| visual fix latency 11-41 ms | `gnss_denied/results/geo_flights_2026-09-27.json`, `latency_ms_wall_median` (capture to fix, wall clock) |
| copy before PPO equals the planner | `log[0].eval_clean` against `baselines.planner` in the same files, all 12 seeds |
| learner settings table | `shared/fast_sim/urban_rl.py` (constants, `ppo_update`, `fit`, `copy_planner`, `train`), checkpoint shapes, `outputs/fast_sim/rl_cond_*.log` (DAgger data, PPO seconds) |
| ambulance from any direction | `shared/fast_sim/rl_eyes_lidar_{random,corridor}_0_1_2.json`, `eval.sampled.vs_planner` (visual-localization error model, 23 Sept) |
| failure reasons (over a person, within 30 m, strikes, timeouts) | `shared/fast_sim/why_cond.json`, `fail`, % of all flights |
| magpie strike rates by sensing | `shared/fast_sim/headroom_lidar_gps_birds{_hunter,_hunter_rig,_hunter_rig_half,_cameras_hunter}.json`, `struck_magpie` |
| detector found / false boxes | `gnss_denied/results/geo_flights_2026-09-27.json`, `detector_retrain` (plain ground: the new weights; photographic ground: old for the false boxes before, new after) |
| detector timing 54-65 ms, at most 157 ms | `conditions/magpie/results/detector_latency_2026-09-30.json` (RTX 5070 laptop, imgsz 1280, fp16, 100 sets of 4 and of 5 pictures, taken while another Gazebo ran, so slow if anything) |
| pad 5 m mat, markers 1.80 / 1.20 / 0.15 m | `make_pad.py` MAT, MARKERS; the figure's rectangles are checked against them |
| Table 3 MJX | `conditions/wind/runs/{rule,fly_hp_s*_best,stage2_c100h100g_s*_best}_ladder.json` (medians and ranges recomputed); the text's earlier deciders: `stage2_c100_s*_best` (discount 0.997, hard -30) and `stage2_s0_best` (-15/-10) |
| Table 3 Gazebo | `outputs/gazebo_wind/ladder_rule/flights.jsonl` (rule e13 at 15 and e12 at 18 m/s left no record); the decider is `stage2_c100h100g_s0_best` (Sai chose it 30 Sept over `stage2_c100_s0`) |
| decider reward and discount | `conditions/wind/land_env.py` step() (landing 10 + 5(1 - miss/0.75), timeout -5, 0.002 per decision, shaping `SHAPE_G * phi' - phi`, phi = -(d + 0.3 h)/10), `conditions/wind/ppo.py` GAMMA 0.9998, checkpoint meta (crash_r, hard_r, shape_g, flip_r); trained by `conditions/wind/queue/wind_flip2_30sep.sh` (s0) and `conditions/wind/queue/wind_gz2_30sep.sh` (s1, s2) |
| "hovering high near the start earned about as much as landing" | arithmetic: at 0.997 a still drone earns 0.003 (d + 0.3 h)/10 - 0.002 per step; at the start (d 15-25 m, h 30 m) about +0.007, times 2,250 steps (90 s) = +15, minus the timeout's 5 = about +10, against 10 to 15 for a landing. First flip-priced runs (`conditions/wind/runs/stage2_c100h100f02_s{0,1}.out`) hovered: timeouts 25-61%, training return up to 61.7 |
| decider that did not transfer (92.1%, 1.7 reversals/s, 16 of 20, 1.06-1.15 m/s) | `conditions/wind/runs/stage2_c100h100_s1_best_ladder.json`; reversals from `outputs/gazebo_wind/ladder_rule/e*_w9_stage2_c100h100_s1_best.csv` vz_cmd; sinks from flights.jsonl. Not in the paper: its PX4 log (collective setpoint at 1.0 in 59% of the descent) was wiped by `launch_px4.sh`'s `rm -rf px4run` on the next launch; `conditions/wind/cmd_replay.py` reproduces the open-loop part |
| recipe seed that reversed 2.2/s; flown seed at most 0.34/s | `conditions/wind/runs/stage2_c100h100g_s{0,1,2}_timing.out` (`conditions/wind/timing_check.py`, hold 1, no delay); Gazebo CSVs of `stage2_c100h100g_s0_best` |
| MJX vs Gazebo step and hover | `conditions/wind/README.md`, section 1 and 2 tables |
| weather text (losses, 10 of 10, medians 0.33-0.37 m over the five skies) | `outputs/weather_runs/{calm,breeze,gusty,mist,mist_nocommit,fog,fog_nocommit}/m_*.json` (medians recomputed); first read, `conditions/weather/weather_report.json` |
| crane, ten positions | `outputs/gazebo_voxel/set10c/flights.jsonl` |
| live magpie 5 of 9 | `outputs/gazebo_rig/{planner,learner}_rig_v2l{,_gate,_final}/flights.jsonl` |
| Table 4 and GNSS-denied text | `gnss_denied/results/geo_flights_2026-09-27.json` (geo_calm4, geo_wind10, geo_obst5, geo_magpie5, geo_magpie6, mission_example_pad4) |
| ten repeated calm GNSS-denied flights (10 of 10 landed) | `outputs/gazebo_geo/geo_calm10_e{0..9}/` (`gnss_denied/queue/geo_calm10_30sep.sh`, 30 Sept, rule, episodes 0-9 = landing heading only) |
| sign test, MJX interval, 69% bound | computed in `check_numbers.py` from `outputs/gazebo_wind/ladder_rule` (the table's four deciders only: the wind session appends new ones), 1,024 flights, 10 of 10 |
| 18 development flights, 6 failed | `outputs/gazebo_geo/<run>/` run.log and flights.jsonl; scenario lists in `check_numbers.py` (`SCN`) |
| A* 2 s to 17 ms | `shared/fast_sim/local_planner.py`, comment on `W_H` |
| fast-simulator constants | `shared/fast_sim/urban_env.py` header; learner recipe, `shared/fast_sim/urban_rl.py` docstring |
