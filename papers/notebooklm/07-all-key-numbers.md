# Every Key Number in One Place

A reference sheet. All figures from real data.

## Calibration (independent of the landing experiments)

| Quantity | Value |
|---|---|
| Usable position fixes | 440, across 4 scenes |
| ENCE (calibration quality, 0 = perfect) | 0.313 |
| Conformal multiplier k | 1.97 |
| Held-out coverage at k | 90.0% across 4 scene folds |
| Median position error | 11.62 m |
| 90th percentile error | 34.34 m |
| 99th percentile error | 127.52 m |
| Fixes within 5 m | 17.3% |
| Fixes within 10 m | 42.7% |
| Fixes within 20 m | 78.6% |
| Absurd outliers (>1000 m) held out | 1 |
| Fix dropout rate in real data | 0.2% |

## Error time-correlation

| Scene | Rank autocorrelation |
|---|---|
| Scene 09 | +0.257 |
| Scene 10 | +0.358 |
| Scene 16 | +0.468 |
| Scene 22 | +0.550 |
| Pooled (dataset frame spacing) | +0.363 |
| Rescaled to 0.5 s fix interval | **+0.707** |

Frame spacing 7.3 m; at 5 m/s that is 1.5 s between frames.

## Per-scene structure

| Scene | n | σ p10 | σ median | σ p90 | Fixes under 10 m σ |
|---|---|---|---|---|---|
| Scene 09 | 149 | 11.2 | 11.9 | 20.1 | 6.7% |
| Scene 10 | 202 | 11.4 | 15.2 | 29.1 | 0.0% |
| Scene 16 | 52 | 9.1 | 11.4 | 13.0 | 36.5% |
| Scene 22 | 37 | 14.7 | 27.9 | 52.2 | 0.0% |

Scenes 10 and 22 contain no easy fixes at all — that is what makes them hard.

## Physical constants of the task

| Quantity | Value |
|---|---|
| Camera field of view | 60 degrees |
| Camera resolution | 640 pixels |
| Marker size | 1.0 m |
| Minimum readable marker width | 20 pixels |
| Detection ceiling | 30.6 m |
| Camera footprint radius at ceiling | 17.7 m |
| Success radius at touchdown | 0.75 m |
| Spawn altitude | 45–65 m |
| Maximum lateral speed | 5 m/s |
| Fix interval | 0.5 s (2 Hz) |
| Battery reserve (episode limit) | 45 s |

## Point-mass results (20 seeds per version)

Success on the hardest third of episodes:

| Correlation | none | staleness | sigma | hand rule |
|---|---|---|---|---|
| 0.36 | 69.3 | 69.7 | 70.4 | 74.8 |
| 0.71 | 58.5 | 60.1 | 58.2 | 81.9 |
| 0.80 | 52.1 | 55.2 | 57.0 | 83.5 |

Significance (sigma versus alternatives):

| Correlation | vs none | vs stopwatch |
|---|---|---|
| 0.36 | +1.1, p=0.76 | +0.7, p=0.84 |
| 0.71 | −0.4, p=0.89 | −2.0, p=0.52 |
| 0.80 | +4.9, p=0.32 | +1.8, p=0.52 |

## Cage results (16 seeds per version)

| Version | Success | Channel contains |
|---|---|---|
| none | 7.8% | nothing |
| clock (control) | 10.1% | bare elapsed time |
| random (control) | 11.5% | a meaningless number |
| sigma | 13.2% | the real error bar |
| staleness | 14.9% | a stopwatch, zero information |

| Comparison | Difference | p |
|---|---|---|
| sigma vs none | +5.4 | 0.011 |
| **sigma vs random** | **+1.7** | **0.397** |
| random vs none | +3.7 | 0.070 |
| staleness vs clock | +4.8 | 0.059 |

## The channel-content measurements

| Channel | Correlation with actual error |
|---|---|
| sigma (real error bar) | +0.549 |
| random (control) | +0.016 |
| staleness (stopwatch) | +0.003 |

The stopwatch is byte-identical across every episode; maximum difference between
any two episodes is exactly zero.

## Policy sensitivity diagnostic

How far the commanded action moves when a channel swings across its range:

| Point-mass (10 policies) | Value |
|---|---|
| sigma channel | 0.2841 |
| altitude (must be used) | 0.3159 |
| distance to pad (must be used) | 0.3669 |

| Cage (16 policies) | Value |
|---|---|
| sigma channel | 0.1507 |
| random channel | 0.1733 |
| difference | −0.0227, p=0.124 |

## The explanation

| Quantity | Value |
|---|---|
| Camera footprint at ceiling | 17.7 m |
| Real fixes with error below that | **75.9%** |
| Dive-policy success at real accuracy | 96.4% |

Degradation needed before the task becomes a decision problem:

| Error scale | Median error | Dive | Hold-and-wait |
|---|---|---|---|
| 1.0× | 11.6 m | 96.4% | 100.0% |
| 1.5× | 17.4 m | 62.8% | 95.2% |
| 2.0× | 23.2 m | 29.6% | 60.0% |
| 3.0× | 34.8 m | 3.6% | 12.0% |

## Scale of the work

| Quantity | Value |
|---|---|
| Point-mass training runs | ~300 |
| Cage training runs | 84 (first attempt, wasted) + 84 + 56 control |
| Confident results later withdrawn | 7 |
| Wrong results caused by physics or maths bugs | 0 |
