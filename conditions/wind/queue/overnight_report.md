# Overnight 26-27 Sept

## 1. MJX ladder (1024 paired flights per wind)

Each cell: landed / crashed / hard (off the pad or too fast) / gave up, in %.

| recipe | seed | 12 m/s | 15 m/s | 18 m/s | 20 m/s |
|---|---|---|---|---|---|
| rule |  | 100 / 0.0 / 0.0 / 0 | 78 / 0.0 / 0.1 / 22 | 6 / 0.1 / 0.0 / 94 | 0 / 1.0 / 0.0 / 99 |
| both | 0 | 98 / 0.0 / 2.3 / 0 | 88 / 2.6 / 9.0 / 0 | 40 / 11.4 / 35.1 / 13 | 8 / 8.2 / 40.9 / 43 |
| both_sh2 | 0 | 100 / 0.1 / 0.1 / 0 | 95 / 4.2 / 0.8 / 0 | 48 / 14.7 / 1.3 / 36 | 8 / 10.3 / 0.3 / 81 |
| fly | 0 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.0 / 1 | 46 / 0.2 / 0.2 / 54 | 4 / 0.3 / 0.0 / 95 |
| fly_hp | 0 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.0 / 1 | 55 / 0.1 / 0.1 / 45 | 7 / 0.2 / 0.0 / 92 |
| fly_hp | 1 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.0 / 1 | 44 / 0.0 / 0.1 / 56 | 5 / 0.5 / 0.0 / 95 |
| fly_hp | 2 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.1 / 1 | 49 / 0.1 / 0.0 / 51 | 6 / 0.3 / 0.1 / 93 |
| fly_hp | 3 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.2 / 1 | 50 / 0.2 / 0.0 / 50 | 6 / 0.3 / 0.0 / 93 |
| fly_hp | 4 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.0 / 1 | 50 / 0.1 / 0.2 / 50 | 7 / 0.4 / 0.0 / 93 |
| fly_hp | 5 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.0 / 1 | 53 / 0.1 / 0.0 / 47 | 7 / 0.4 / 0.1 / 92 |
| fly_hp | 6 | 100 / 0.0 / 0.1 / 0 | 99 / 0.0 / 0.0 / 1 | 51 / 0.1 / 0.1 / 48 | 7 / 0.5 / 0.0 / 93 |
| fly_hp | 7 | 100 / 0.0 / 0.0 / 0 | 99 / 0.0 / 0.0 / 1 | 54 / 0.2 / 0.5 / 45 | 8 / 0.2 / 0.0 / 92 |
| **fly_hp: median (range) over 8 seeds, landed / crashed** | | **100** (100-100) / 0.0 (0.0-0.0) | **99** (99-99) / 0.0 (0.0-0.0) | **51** (44-55) / 0.1 (0.0-0.2) | **7** (5-8) / 0.3 (0.2-0.5) |
| stage2 | 0 | 99 / 0.0 / 1.1 / 0 | 95 / 0.6 / 4.1 / 0 | 69 / 11.7 / 18.4 / 1 | 33 / 14.5 / 50.4 / 3 |
| stage2_c100 | 0 | 99 / 0.0 / 1.2 / 0 | 94 / 0.2 / 5.7 / 0 | 87 / 2.1 / 10.1 / 0 | 76 / 7.6 / 10.8 / 5 |
| stage2_c100 | 1 | 100 / 0.0 / 0.0 / 0 | 98 / 0.9 / 1.0 / 0 | 81 / 9.3 / 10.2 / 0 | 45 / 17.0 / 37.9 / 0 |
| stage2_c100 | 2 | 99 / 0.0 / 0.6 / 0 | 98 / 0.4 / 1.5 / 0 | 91 / 3.3 / 5.6 / 0 | 64 / 12.5 / 19.5 / 4 |
| **stage2_c100: median (range) over 3 seeds, landed / crashed** | | **99** (99-100) / 0.0 (0.0-0.0) | **98** (94-98) / 0.4 (0.2-0.9) | **87** (81-91) / 3.3 (2.1-9.3) | **64** (45-76) / 12.5 (7.6-17.0) |
| stage2_c100h100 | 0 | 100 / 0.0 / 0.3 / 0 | 98 / 0.2 / 1.5 / 0 | 94 / 2.4 / 3.9 / 0 | 76 / 8.2 / 12.6 / 3 |
| stage2_c300 | 0 | 100 / 0.0 / 0.1 / 0 | 99 / 0.3 / 1.1 / 0 | 94 / 3.0 / 3.4 / 0 | 76 / 11.4 / 12.9 / 0 |
| stage2hp_c100 | 0 | 54 / 0.0 / 46.4 / 0 | 46 / 2.2 / 51.9 / 0 | 42 / 11.0 / 42.5 / 4 | 13 / 10.7 / 56.2 / 20 |

## 2. Gazebo ladder (20 yards x 4 winds, the same gusts for every decider)

Each cell: landed of flown (crashed, hard, gave up).

| decider | 9 m/s | 12 m/s | 15 m/s | 18 m/s |
|---|---|---|---|---|
| rule | 20/20 (0, 0, 0) | 19/20 (0, 0, 1) | 12/19 (0, 0, 7) | 0/19 (0, 0, 19) |
| fly_hp_s0_best | 20/20 (0, 0, 0) | 20/20 (0, 0, 0) | 18/20 (0, 0, 2) | 6/20 (0, 0, 14) |
| fly_hp_s1_best | 20/20 (0, 0, 0) | 20/20 (0, 0, 0) | 19/20 (0, 0, 1) | 5/20 (0, 0, 15) |
| stage2_c100_s0_best | 20/20 (0, 0, 0) | 19/20 (0, 1, 0) | 19/20 (0, 1, 0) | 15/20 (3, 1, 1) |

Paired against the rule (same yard, same wind, same gusts): learner landed where the rule didn't / rule landed where the learner didn't.

- fly_hp_s0_best: 9 m/s: +0 / -0 of 20, 12 m/s: +1 / -0 of 20, 15 m/s: +7 / -2 of 19, 18 m/s: +6 / -0 of 19
- fly_hp_s1_best: 9 m/s: +0 / -0 of 20, 12 m/s: +1 / -0 of 20, 15 m/s: +6 / -0 of 19, 18 m/s: +5 / -0 of 19
- stage2_c100_s0_best: 9 m/s: +0 / -0 of 20, 12 m/s: +1 / -1 of 20, 15 m/s: +6 / -0 of 19, 18 m/s: +14 / -0 of 19

## 3. GeoAnchor flights (GNSS denied over the suburb, the pad marker below 50 m)

- **geo_calm** (rule, wind 0 m/s): **hard**, 2.32 m from the pad centre, 31 s landing; GNSS denied 119.0 s (GNSS came back at t=191 s); PX4's position error median 3.2 m, p90 5.01 m, worst 7.13 m; 541 GeoAnchor fixes (median 2.14 m, p90 4.75 m), 444 marker fixes; video calm.mp4 (internal: Esri floor)
- **geo_obst_baro** (rule, unmapped obstacles, wind 0 m/s): **timeout**, 0.08 m from the pad centre, 300 s landing; GNSS denied 386.9 s; PX4's position error median 0.45 m, p90 3.57 m, worst 13.33 m; 645 GeoAnchor fixes (median 2.13 m, p90 5.24 m), 8744 marker fixes; video obstacles.mp4 (internal: Esri floor)
- **geo_calm_rng** (rule, wind 0 m/s): **landed**, 0.06 m from the pad centre, 25 s landing; GNSS denied 116.8 s; PX4's position error median 2.44 m, p90 4.97 m, worst 8.07 m; 796 GeoAnchor fixes (median 2.23 m, p90 5.56 m), 388 marker fixes; video calm.mp4 (internal: Esri floor)
- **geo_calm2** (rule, wind 0 m/s): **landed**, 0.06 m from the pad centre, 25 s landing; GNSS denied 117.8 s; PX4's position error median 2.36 m, p90 5.44 m, worst 6.93 m; 542 GeoAnchor fixes (median 2.24 m, p90 5.91 m), 380 marker fixes; video calm.mp4 (internal: Esri floor)
- **geo_wind9** (fly_hp_s0_best, wind 9 m/s): **crashed**, 3.88 m from the pad centre, 30 s landing; GNSS denied 122.9 s (GNSS came back at t=195 s); PX4's position error median 4.46 m, p90 11.37 m, worst 26.55 m; 542 GeoAnchor fixes (median 2.6 m, p90 5.62 m), 379 marker fixes; video wind9.mp4 (internal: Esri floor)
- **geo_obst** (rule, unmapped obstacles, wind 0 m/s): **landed**, 0.21 m from the pad centre, 28 s landing; GNSS denied 111.0 s; PX4's position error median 2.54 m, p90 4.26 m, worst 6.15 m; 601 GeoAnchor fixes (median 2.27 m, p90 5.57 m), 681 marker fixes; video obstacles.mp4 (internal: Esri floor)
- **geo_magpie** (learner, magpie 3 swoops, closest 2.0 m, wind 0 m/s): **landed**, 0.14 m from the pad centre, 28 s landing; GNSS denied 111.1 s (GNSS came back at t=184 s); PX4's position error median 2.31 m, p90 4.02 m, worst 5.2 m; 849 GeoAnchor fixes (median 2.21 m, p90 5.31 m), 694 marker fixes; video magpie.mp4 (internal: Esri floor)
