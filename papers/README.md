# papers/: what we write and show

Nothing here flies a drone. Every number in these papers is read from result files kept in the other folders.
Run every command below from the repo root, `/home/sai/RL_projects/rl-landing`.

| Folder | What it is |
|---|---|
| `acra2026/` | The ACRA 2026 paper draft. Frozen: 10 pages, **not submitted** (Sai, 4 Oct). Do not edit its text. |
| `research/` | Four fact-checked reports from 4 Oct. **Read these next.** |
| `slides/`, `deck/`, `docs/` | The slide deck (source, an old snapshot, and its charts). |
| `paper/`, `report/` | Two small review pages built from saved results. |
| `notebooklm/`, `diagram-prompts/` | Plain-words notes for NotebookLM, and one prompt per Gemini diagram. |
| `wind/` (not made yet) | The wind-only paper. It will use `research/` and `conditions/wind/`. |

---

## 1. acra2026: the frozen draft

The paper is "Where Learning Helps a Delivery Drone Land". Its text, bibliography and figures stay as they are.
`acra2026/README.md` has the file layout and where every number comes from.

### 1a. Check that every number still matches its source (2 seconds)

1. Run the checker.

```bash
python3 papers/acra2026/check_numbers.py
```

2. Read the last two lines. Today they say `OK 180   MISMATCH 1`.
3. The one mismatch is known and harmless. The check looks for the line `def fit(net, X, Y, epochs=8, mb=32768, smooth=0.0):` in `shared/fast_sim/urban_rl.py`, and that function has since gained a `stats=None` argument. The values it guards (8 epochs, learning rate 1e-3) did not change. It was already so before the reorganisation. Fixing it means editing the check, not the paper.

It also rewrites `papers/acra2026/number_audit.json`. If `git diff` shows a changed value there (not just a source path), a source file moved a number.

### 1b. Rebuild the PDF (about 1 second)

1. Build in a copy, so the frozen `main.pdf` stays untouched.

```bash
cp -r papers/acra2026 /tmp/acra_build
```

```bash
cd /tmp/acra_build && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

2. Expect `Output written on main.pdf (10 pages, ...)`. ACRA allows 6 to 10 pages, so anything added needs a cut elsewhere.

### 1c. Redraw the figures (only if a result file changed)

1. Redraw the paper's data figures. This **overwrites** `papers/acra2026/figs/`, so run it only on purpose.

```bash
python3 papers/acra2026/make_figs.py
```

2. The Logan world and the learning curves come from `shared/fast_sim/paper_figs.py`. Its output goes to `outputs/paper/figs/`, not into the paper (about 3 seconds per figure).

```bash
cd shared/fast_sim && /home/sai/miniconda3/envs/mujoco_drones/bin/python paper_figs.py world
```

The other names are `headroom learning birds rig gazebo`. With no name it draws all of them.

### 1d. Notes kept beside the draft

| File | What it holds |
|---|---|
| `acra2026/LITERATURE_COMPARISON.md` | How our results compare with published work (30 Sept) |
| `acra2026/REFERENCES_CHECK.md` | Every reference checked against Crossref, arXiv or the publisher (29 Sept) |
| `acra2026/PLAN_next.md` | What would make the paper stronger, with costs |
| `acra2026/review/` | The mock review's reworded copy of the paper (`make_review_pdf.py`) |

---

## 2. research/: the four reports

Each answers one question in plain words. `research/README.md` has the one-line summary of each, and
`research/scripts/` holds the code and small result files they rest on (they import `conditions/wind/`).

| Report | Question it answers |
|---|---|
| `research/hierarchy_literature.md` | Has anyone already built our stack (planner, local navigator, safety filter, autopilot)? |
| `research/aerial_gym_comparison.md` | Our fast simulator, the MJX twin, Aerial Gym or PX4 SITL: which fits us best? |
| `research/x500_wind_limit.md` | How much wind can the x500 take? About 26 m/s steady at the drone, in our model with an assumed body drag; PX4's 45 degree tilt limit binds first. No published rating exists. |
| `research/wind_algorithm_options.md` | Which learning algorithm for the wind learner? A 15-line disturbance-observer feed-forward lands 80.6% at W10 18, against 44-55% for the learned flyer. |

The last two are research, not in the ACRA paper.

---

## 3. The wind paper (to come)

Not started. It will live in `papers/wind/` and draw on:

1. the wind study in `conditions/wind/` (its README has the training and Gazebo steps),
2. the two wind reports above,
3. the ClearML projects `rl-landing/wind` and `rl-landing/fast_sim`.

---

## 4. The slide deck

The deck covers all five parts of the study. It is a private Claude artifact: <https://claude.ai/artifact/QaYeAMh7cZThMmffKxjjNA>.

| Folder | What it holds |
|---|---|
| `slides/` | The deck's source: one HTML file per slide in `slides/slides/` (52 files), the order in `slides/deck.json` |
| `deck/` | An older snapshot (`deck/project/`) and two helpers |
| `docs/figures/` | Seven charts used by the deck |

The films are not in git (Esri imagery rule, see the root README, rule 1).

### 4a. Redraw the deck's wind and GeoAnchor graphs (2 seconds)

1. Run the script.

```bash
/home/sai/miniconda3/envs/mujoco_drones/bin/python papers/deck/deck_figs.py
```

2. Expect lines such as `geo_trace ok` and `pad_markers ok`. The PNGs land in `outputs/gazebo_geo/deck/`. They show numbers only, no Esri pixels.

### 4b. Turn a flight film into a deck clip

1. Convert it. The output is 1280 x 720 VP8, under the deck's 20 MB limit, plus a still.

```bash
python3 papers/deck/to_webm.py film.mp4 out.webm --still-at 0.9 --kbps 750
```

2. Expect `out.webm` and a still beside it. It needs Playwright's ffmpeg at `~/.cache/ms-playwright/ffmpeg-1011/`.
3. Films show the Esri floor. Keep them out of git.

---

## 5. Two review pages

Both write one HTML file into `outputs/` (not in git) and take well under a second.

1. The ACRA figure draft, with captions read from the same result files as the figures. It needs `outputs/paper/figs/` from step 1c.

```bash
python3 papers/paper/build_figures.py
```

Expect `outputs/paper/figures.html` (about 1.5 MB).

2. The magpie-rig report, filled from the results on disk.

```bash
python3 papers/report/build_report.py
```

Expect `outputs/report/rig_report.html` (about 2.4 MB). It stops with an error if a `%%` placeholder is left unfilled.

The template `report/rig_report.template.html` is tracked. `report/strike.jpg` is not in git (its provenance is unknown, so it may hold Esri pixels).

---

## 6. For NotebookLM and Gemini

1. `notebooklm/`: nine plain-words chapters about the sigma study, readable without the code. Start with `00-README-for-notebooklm.md`. They cover the early parts and are older than the wind and magpie work.
2. `diagram-prompts/`: one `.md` per diagram, eleven so far, in the order of the story (`01-what-the-problem-is.md` to `11-system-overview.md`). Paste one into Gemini.
