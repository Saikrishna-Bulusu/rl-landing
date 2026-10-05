# Research notes (4 Oct 2026)

Plain-words reports, one question each. The scripts and small results they rest on are in `scripts/` (run them from there with `/home/sai/miniconda3/envs/mujoco_drones/bin/python`; they import `conditions/wind/`). Nothing here changed the repo's code or the frozen ACRA draft.

- `hierarchy_literature.md`: has someone already built our hierarchical architecture (planner, local navigator, safety filter, autopilot)? Literature check.
- `aerial_gym_comparison.md`: fast simulator, MJX twin, Aerial Gym or PX4 SITL: which is best for us?
- `x500_wind_limit.md`: how much wind can the x500 take? (steady wind and gusts, with the assumed body drag and the checks)
- `wind_algorithm_options.md`: which learning algorithm for the wind learner, and why (includes the disturbance-observer feed-forward baseline).
