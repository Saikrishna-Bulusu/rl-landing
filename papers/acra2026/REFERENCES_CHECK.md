# Reference check, 29 Sept 2026

Every claim the draft makes about a cited source was checked against that source's full text, saved in
`~/RL_projects/papers/acra2026_refs/` (plain-text copies in `txt/`). Paywalled PDFs came through the UTS library.

| Key | What the draft says | Checked against | Verdict |
|---|---|---|---|
| lin2024vosafe | PPO policy picks waypoints for a low-level controller; semantic masks, not raw images; penalty when VO fails; learns to avoid feature-poor areas such as water and trees; no moving obstacles | IEEE PDF (UTS): PPO-clip, 8 discrete 1 m moves, cascaded PID; R_VO = -10 if VO failed; lake/trees in sim, white paper in real test; no obstacles mentioned | OK. Was "texture-poor ground", now the paper's "feature-poor areas" |
| benali2026 | urban last-mile RL; helicopter on one fixed straight path; fixed relative start positions and headings | Vehicles PDF, Sec. 3 ("predetermined, straight-line flight path"; "initial relative positions and headings ... were fixed") | OK |
| polvara2017, polvara2020 | marker landing with a sequence of DQNs, one per stage, trained in simulation with domain randomization | arXiv 1709.03339 ("hierarchy of DQNs"), Robotics 9(1):8 (sequential sub-tasks, one DQN each; domain randomization) | OK |
| rodriguez2018 | lands on a moving platform with deep RL | Springer PDF (UTS): DDPG, Gazebo, moving platform | OK. Year corrected 2018 -> 2019 (printed JINT 93:351-366, 2019) |
| peter2024 | lands on moving platforms under wind disturbance | arXiv 2406.16164 | FIXED. Was "a platform that moves in six degrees of freedom": "6D" appears only in the title |
| li2026wavelander | policy outputs only a vertical velocity reference; conventional controller keeps attitude and lateral position | arXiv 2607.01281, abstract and Sec. III | OK |
| amendola2024 | reviews drone landing with RL | IEEE OJITS PDF | OK |
| xu2025navrl | PPO, static and moving obstacles, safety shield from velocity obstacles | IEEE RA-L PDF (UTS) | OK |
| alshiekh2018 | a shield blocks a learned policy's unsafe actions with a separate check | AAAI PDF ("monitors the actions ... corrects them if and only if the chosen action is unsafe") | OK |
| falanga2020 | dodges fast-moving objects with event cameras, overall latency 3.5 ms, no learning | published version, ZORA open access (no UTS access to Science Robotics) | FIXED. Was "3.5 ms of perception latency": the paper says "overall latency". The word "learning" never appears |
| hayashi2026 | (dropped 29 Sept at Sai's request: full text not obtainable) | abstract only | REMOVED from the draft and refs.bib |
| lyons2018 | magpies chased the drone twice in the breeding season; operator's evasion avoided collision | published AFO version (UNSWorks open access) | STRENGTHENED. Was only the title; now the reported magpie incidents, also cited in the introduction |
| song2023 | learns perception-aware agile flight in clutter | arXiv 2210.01841 (accepted ICRA 2023 version) | OK |
| zhang2018 | perception-aware planning that reduces state-estimation uncertainty | IEEE PDF (UTS) | FIXED. Was grouped as "keep visual features in view"; the paper minimizes estimation uncertainty |
| falanga2018 | keeps a point of interest in view | arXiv 1804.04811 ("maximizing the visibility of a point of interest") | OK, now cited on its own |
| kahn2017 | collision-avoidance learner proceeds cautiously where uncertain | arXiv 1702.01182 | OK |
| puthumanaillam2024 | conditions navigation policies on the localization precision each region needs | arXiv 2410.15178 (Task-Specific Uncertainty Maps) | OK |
| couturier2024 | reviews absolute visual localization | Drones 8(11):622 PDF | OK |
| habel2026gustpilot | learned velocity-reference policy plus INDI controller that rejects gusts | arXiv 2603.19966 | OK |
| johannink2019, silver2018 | residual RL adds a learned correction to a conventional controller | arXiv 1812.03201, 1812.06298 | OK |
| schulman2017, ross2011 | PPO; DAgger = dataset aggregation | arXiv 1707.06347; PMLR 15 PDF | OK |
| meier2015, koenig2004, todorov2012 | PX4; Gazebo; MuJoCo | IEEE PDFs (UTS for the last two) | OK |
| sapkota2025 | YOLO26 (our detector is YOLO26n) | arXiv 2509.25164 | OK. It analyses Ultralytics YOLO26; it is not by the model's authors |
| milf8785c | low-altitude turbulence form | everyspec PDF, 3.7.1-3.7.3 (sigma = 0.1 u20, u20 = wind at 20 ft; discrete gust "1 - cosine") | FIXED. Now says we scale turbulence by W10 where the spec uses the 20 ft wind |
| casa2026 | 120 m; 30 m from people, never over a person; within 1.4 km of a helicopter landing site move away from a helicopter nearby, taking off or landing | casa.gov.au drone rules page, read 29 Sept | FIXED. These are CASA's rules for RECREATIONAL flying; the draft now says we apply three of them |
| astm2023 | NMAC 152 m horizontal / 30 m vertical; covers crewed aircraft | the standard's PDF, downloaded by Sai through UTS (copy in acra2026_refs/astm2023.pdf): 3.3.19 defines an NMAC as two aircraft within 100 ft vertically and 500 ft horizontally; 1.5 limits it to avoiding crewed aircraft, not UA-to-UA or birds | OK. 500 ft = 152.4 m, 100 ft = 30.5 m. The simulator scores NMAC only against the air ambulance (urban_env.py:700). Its well-clear volume, 2000 ft by 250 ft (3.3.34), matches the code's 610 m / 76 m but is not used in the paper |
| garrido2014 | ArUco markers | Pattern Recognition 47:2280-2292 PDF, downloaded by Sai through UTS (acra2026_refs/garrido2014.pdf): the marker system is released as the ArUco library | OK |
