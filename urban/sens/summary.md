# Sensitivity of the fast-simulator results (29 Sept)

Each assumption changed alone; learners NOT retrained; 4,096 test flights (corridor, seed 2026) per run; learner = mean of 3 seeds, dice read-out.
Gain = learner minus the planner it copied (shield), paired; 'all seeds > 0' = every seed's paired 95% interval above zero.


## Clear

| Variation | Rule | Shield | Safe planner | Learner | Gain over copied planner | All seeds > 0 | Gain over best planner |
|---|---|---|---|---|---|---|---|
| as in the paper | 93.7 | 94.5 | 94.3 | 96.3 | +1.8 | yes | +1.8 |
| people leave the pad after 10-120 s (paper: 30-240) | 94.8 | 95.5 | 95.6 | 97.5 | +2.0 | yes | +1.9 |
| people leave after 60-480 s | 65.2 | 65.3 | 65.6 | 66.8 | +1.5 | yes | +1.2 |
| flight limit 200 s (paper: 300) | 61.5 | 61.1 | 63.6 | 63.2 | +2.2 | yes | -0.4 |
| flight limit 400 s | 96.0 | 97.5 | 96.7 | 98.3 | +0.8 | yes | +0.8 |
| drone separation 50 m (paper: 30) | 90.5 | 94.0 | 93.6 | 95.4 | +1.4 | yes | +1.4 |

## Fog

| Variation | Rule | Shield | Safe planner | Learner | Gain over copied planner | All seeds > 0 | Gain over best planner |
|---|---|---|---|---|---|---|---|
| as in the paper | 68.8 | 71.9 | 69.3 | 86.2 | +14.4 | yes | +14.4 |
| people leave the pad after 10-120 s (paper: 30-240) | 75.8 | 78.7 | 77.9 | 91.1 | +12.4 | yes | +12.4 |
| people leave after 60-480 s | 55.4 | 56.9 | 55.8 | 63.3 | +6.5 | yes | +6.5 |
| flight limit 200 s (paper: 300) | 55.5 | 55.9 | 58.2 | 65.1 | +9.2 | yes | +6.9 |
| flight limit 400 s | 69.9 | 73.6 | 70.2 | 88.3 | +14.7 | yes | +14.7 |
| drone separation 50 m (paper: 30) | 66.8 | 71.4 | 69.0 | 85.5 | +14.1 | yes | +14.1 |

## Magpie through the camera rig

| Variation | Rule | Shield | Safe planner | Learner | Gain over copied planner | All seeds > 0 | Gain over best planner |
|---|---|---|---|---|---|---|---|
| as in the paper | 72.2 | 69.8 | 73.4 | 77.6 | +7.8 | yes | +4.1 |
| people leave the pad after 10-120 s (paper: 30-240) | 75.9 | 72.6 | 76.5 | 80.8 | +8.2 | yes | +4.2 |
| people leave after 60-480 s | 52.4 | 49.4 | 52.4 | 55.6 | +6.2 | yes | +3.2 |
| flight limit 200 s (paper: 300) | 48.5 | 43.3 | 50.3 | 54.7 | +11.4 | yes | +4.4 |
| flight limit 400 s | 74.5 | 73.6 | 75.8 | 79.5 | +5.8 | yes | +3.7 |
| drone separation 50 m (paper: 30) | 69.7 | 69.7 | 73.0 | 76.9 | +7.3 | yes | +3.9 |
| magpie commits 0.1 s before contact (paper: 0.3) | 68.1 | 67.5 | 71.4 | 75.3 | +7.9 | yes | +3.9 |
| magpie commits 0.5 s before contact | 75.1 | 73.7 | 77.3 | 80.5 | +6.7 | yes | +3.2 |
| rig sees half as far | 71.6 | 70.3 | 73.5 | 77.8 | +7.6 | yes | +4.3 |
