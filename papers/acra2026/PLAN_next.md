# What would make the paper stronger (29 Sept, for the talk with Nabin)

Useful for ACRA or a later journal version. Times come from rates measured this week. GPU work and
Gazebo never run together (training slows the simulator). One PX4 simulator at a time for GNSS-free flights.

| # | Work | Why it matters | Cost |
|---|---|---|---|
| 1 | Gazebo wind ladder for the new decider (running, 80 flights) | Table 2's Gazebo row then matches the MJX row | tonight, ~4 h |
| 2 | GNSS-free flights, 5 seeds each: calm, wind 9 m/s, crane | "one flight per scenario" becomes a landing rate and a miss spread | ~1.5 h Gazebo |
| 3 | GNSS-free magpie flights, 5 seeds | the magpie case at 0.25x real time | ~1.5-2 h Gazebo |
| 4 | Live magpie, planner vs learner, 10 seeds each, fixed landing code | today's 5 of 9 each spans three code versions and is noise | ~8 h Gazebo (one long night) |
| 5 | Sensitivity of the fast-simulator ladder: people leave 10-120 s and 60-480 s (now 30-240); flight limit 200 and 400 s (now 300); magpie commit 0.1 and 0.5 s (now 0.3); magpie seen within 20 and 60 m (now 40); drone separation 50 m (now 30) | shows whether "learning pays where information is missing" survives the assumptions | ~2-3 h GPU (about 20 min per ladder, learners evaluated without retraining) |
| 6 | Flying learner in wind: longer runs or a larger network | 18 m/s is ~51% for the flyer alone; the decider on top already reaches ~93% | ~1 h GPU per try; uncertain gain |

Not in reach before 4 Oct: hardware flights (Tello cage), and the professor's localization-aware hierarchical RL with
other drones and unknown moving objects (about 2-3 weeks, see memory hierarchical-rl-vo-plan).

Suggested order if the paper goes to ACRA: 1 (tonight), 5 (day), 2 + 3 (next night), then 4 only if time remains.
If it goes to a journal instead: all of the above, then the hierarchical RL work and hardware.
