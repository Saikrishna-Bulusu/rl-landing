# Tasks (Sai, 4 Oct 2026)

Order matters: the repo must stop moving before anything else edits code. `[ ]` open, `[~]` running, `[x]` done.

## Phase 0: safety net
- [ ] 0.1 Commit today's uncommitted wind work (`--flip-r`, `--shape-g`, `--log-test`; the paper's decider needs them) and tag `pre-reorg`

## Phase 1: tidy the code (items 1-3)
- [ ] 1.1 Ponytail audit of the whole codebase: dead code, duplicates, `.bak` copies, what to archive (report only)
- [ ] 1.2 New folder layout, local and GitHub: magpie, wind, dynamic obstacles, static obstacles (crane, A*), all conditions together, mix-and-match, dashboard (+ shared code, GNSS-denied, papers, archive)
- [ ] 1.3 Fix every import and path the move breaks; prove it with the self-checks
- [ ] 1.4 One README per condition (how to run it alone) + one for running all conditions + one for mix-and-match
- [ ] 1.5 Push to GitHub (only after Sai says yes)

## Phase 2: see the training (items 4, 15)
- [ ] 2.1 ClearML in every trainer (wind `ppo.py`, suburb `urban_rl.py`): curves, settings, checkpoints
- [ ] 2.2 Past wind runs imported into ClearML from their logs (no retraining needed to see the curves)
- [ ] 2.3 Wind environment visualized: a video of drones landing in gusts, also logged to ClearML
- [ ] 2.4 Algorithm choice for wind, with reasons (Sai decides after seeing 2.1-2.3)
- [ ] 2.5 Parallel training explained and measured (the wind trainer already flies 4,096 drones at once)

## Phase 3: answers (items 5, 6, 8, 9, 10, 12, 13, 14)
- [ ] 3.1 Who has built a hierarchical stack like ours? Nearest architecture (literature search)
- [ ] 3.2 What the fast simulator is (Gymnasium? PettingZoo?), why 1,324 pads
- [ ] 3.3 Aerial Gym vs our fast simulator vs MJX, for dynamic obstacles and wind (comparative study)
- [ ] 3.4 Explain: PX4's first-order lags (0.69 s / 0.23 s)
- [ ] 3.5 Explain: GNSS off after 4 fixes in 15 s, EKF2, the matcher, offboard vs on-board
- [ ] 3.6 Explain: why three ArUco markers, the 14 candidate commands, + diagram (policy network, observation, action, reward) per condition
- [ ] 3.7 Explain: the wind agent, MIL-F-8785C, 1-cosine gusts, W10, the x500's wind limit
- [ ] 3.8 Should wind split into route-flying + landing policies? (recommendation, decided after measuring)
- [ ] 3.9 Explain Table 1 line by line: shield planner, DAgger rounds, Adam 1e-3/1e-4, why "12 epochs" is a misreading

## Phase 4: the paper (items 7, 11)
- [ ] 4.1 Sharper Figure 3 (the Logan map)
- [ ] 4.2 Wind-only paper (other conditions become their own papers later)
