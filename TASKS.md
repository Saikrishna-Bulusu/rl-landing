# Tasks (Sai, 4 Oct 2026)

Order matters: the repo must stop moving before anything else edits code. `[ ]` open, `[~]` running, `[x]` done.

## Phase 0: safety net
- [x] 0.1 Commit today's uncommitted wind work (`--flip-r`, `--shape-g`, `--log-test`; the paper's decider needs them) and tag `pre-reorg`

## Phase 1: tidy the code (items 1-3)
- [x] 1.1 Ponytail audit of the whole codebase: dead code, duplicates, `.bak` copies, what to archive (report only)
- [x] 1.2 New folder layout, local and GitHub: magpie, wind, dynamic obstacles, static obstacles (crane, A*), all conditions together, mix-and-match, dashboard (+ shared code, GNSS-denied, papers, archive)
- [x] 1.3 Fix every import and path the move breaks; prove it with the self-checks
- [x] 1.4 One README per condition (how to run it alone) + one for running all conditions + one for mix-and-match
- [x] 1.5 Push to GitHub (pushed 6 Oct, 813f472)

## Phase 2: see the training (items 4, 15)
- [x] 2.1 ClearML in every trainer (wind `ppo.py`, suburb `urban_rl.py`): curves, settings, checkpoints
- [x] 2.2 Past wind runs imported into ClearML from their logs (no retraining needed to see the curves)
- [x] 2.3 Wind environment visualized: a video of drones landing in gusts, also logged to ClearML
- [ ] 2.4 Algorithm choice for wind, with reasons (Sai reads papers/research/wind_algorithm_options.md first; recommendation: PPO decider on a disturbance-observer flyer)
- [x] 2.5 Parallel training explained and measured (the wind trainer already flies 4,096 drones at once)

## Phase 3: answers (items 5, 6, 8, 9, 10, 12, 13, 14)
- [x] 3.1 Who has built a hierarchical stack like ours? Nearest architecture (literature search)
- [x] 3.2 What the fast simulator is (Gymnasium? PettingZoo?), why 1,324 pads
- [x] 3.3 Aerial Gym vs our fast simulator vs MJX, for dynamic obstacles and wind (comparative study)
- [x] 3.4 Explain: PX4's first-order lags (0.69 s / 0.23 s)
- [x] 3.5 Explain: GNSS off after 4 fixes in 15 s, EKF2, the matcher, offboard vs on-board
- [x] 3.6 Explain: why three ArUco markers, the 14 candidate commands, + diagram (policy network, observation, action, reward) per condition
- [x] 3.7 Explain: the wind agent, MIL-F-8785C, 1-cosine gusts, W10 (done); the x500's wind limit: ~26 m/s steady at the drone in our model, W10 ~15 practical at 30 m (papers/research/x500_wind_limit.md)
- [x] 3.8 Should wind split into route-flying + landing policies? (recommendation, decided after measuring)
- [x] 3.9 Explain Table 1 line by line: shield planner, DAgger rounds, Adam 1e-3/1e-4, why "12 epochs" is a misreading

## Phase 4: the paper (items 7, 11)
- [~] 4.1 Sharper Figure 3 (the Logan map): drawn (scratchpad world_v2.py), to fold into shared/fast_sim/paper_figs.py
- [ ] 4.2 Wind-only paper (other conditions become their own papers later)

## Phase 5: wind, the agreed plan (6 Oct): see conditions/wind/PLAN.md
- [x] 5.0 Baselines (Phase 0): tuned rule 75.7%, DOB 80.6%, paper decider 93.9-96.2%, DOB + decider 92.5-95.9% at W10 18
- [~] 5.0b Gazebo smoke for Phase 0 (rule, paper decider s0, DOB decider s0; yards 0-1, W10 18)
- [ ] 5.1 Phase 1: clean the interface (rule asks 1.5, pick table, greedy vs sampled, new metrics, W10 22)
- [ ] 5.2 Phase 2: realism in MJX (sensors, drone randomization, gusts) + robustness table
- [ ] 5.3 Phase 3: privileged critic, saturation/slew penalties, DOB input, aligned frame (one at a time, 3 seeds)
- [ ] 5.4 Phase 4: architecture ablation (width, depth, history, tanh/ReLU, GRU)
- [ ] 5.5 Phase 5: safety filter + hard cases
- [ ] 5.6 Phase 6: full Gazebo ladder; Phase 7: the wind paper
- [x] Longer training check (1,500 rounds, seed 0): flat after ~450 at 15-18 m/s, +4.4 at 20 m/s (one seed)
