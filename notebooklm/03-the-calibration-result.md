# The Calibration Result

This is the part of the work that stands entirely on its own, independent of
whether the landing experiments succeed or fail. It uses data already on disk and
produces numbers that are directly comparable to other published work.

## What calibration means

GeoAnchor reports both a position and an error bar. Calibration asks a simple
question: **is the error bar honest?**

If the system says "give or take twenty metres," is the real error actually
around twenty metres? Or does it claim twenty when the truth is five, or claim
five when the truth is forty?

This matters more than it might appear. A *dishonest* error bar is worse than no
error bar at all. Consider a system that always reports "give or take ten metres"
regardless of the situation:

- When the fix is genuinely good, accurate to three metres, the drone searches
  when it should have committed, and wastes battery.
- When the fix is genuinely terrible, off by forty metres, the drone confidently
  dives — and lands in the wrong garden.

The second case is the dangerous one. An overconfident error bar makes a system
act decisively at exactly the moment it should hesitate.

## Result one: Expected Normalised Calibration Error

**ENCE = 0.313**, where zero is perfect.

This metric bins the fixes by their claimed uncertainty and, within each bin,
compares the claimed uncertainty against the error actually observed. It is an
established published measure, which means the number can be compared against
other systems rather than only interpreted internally.

The project's own notes had previously recorded the estimator's quality as an
informal table — roughly, claims of 12 metres produced 12 metres of error, claims
of 21 produced 21, claims of 38 produced 45. That is an eyeballed, unnamed
version of exactly this calculation. Reporting ENCE makes it standard.

Crucially, the figure is computed *leave-one-scene-out*: the calibration is fitted
on some scenes and evaluated on a scene it has never seen. Splitting by individual
measurement rather than by scene would leak scene identity and flatter the result.

## Result two: a distribution-free coverage guarantee

**The true position lies within 1.97 times the reported error bar at least 90
percent of the time**, with measured held-out coverage of exactly 90.0 percent
across four scene folds.

This is a stronger and more useful statement than ENCE. It upgrades the claim
from "the error bar broadly tracks the error" to a guarantee a downstream system
can actually act on.

It is produced by conformal calibration, which makes **no assumption that the
errors follow a bell curve**. This matters here because they emphatically do not
— the error distribution has a heavy tail, with rare catastrophic failures. A
method that assumed normality would produce a confident and wrong guarantee.

## Result three: the error percentiles

Reported as percentiles, never as an average — a hard rule in this project,
because a single catastrophic fix can reach absurd magnitudes and would dominate
any mean.

- Median error: 11.62 metres
- 90th percentile: 34.34 metres
- 99th percentile: 127.52 metres
- Within 5 metres: 17.3 percent of fixes
- Within 10 metres: 42.7 percent
- Within 20 metres: 78.6 percent

Based on 440 usable fixes across four scenes, with one absurd outlier above 1000
metres held out and counted separately.

## Result four: how errors behave over time

A new measurement, and the one that exposed a structural flaw in the simulation.

Because the dataset's images are sequential frames, it is possible to ask whether
a bad fix tends to be followed by another bad fix. It does. The rank correlation
between consecutive errors is:

- Scene 09: +0.257
- Scene 10: +0.358
- Scene 16: +0.468
- Scene 22: +0.550
- Pooled: +0.363

Rank correlation rather than the ordinary kind, because the heavy tail means a
single enormous error would otherwise determine the statistic.

Rescaled from the dataset's frame spacing to the half-second interval at which a
flying drone receives fixes, this becomes approximately **0.71**.

## Why these four results matter regardless of the rest

Every one of these numbers comes from data that already existed. None depends on
the reinforcement learning experiments working. They strengthen the underlying
localisation paper by themselves: a named, comparable calibration metric; a
distribution-free coverage guarantee with a proof; honest percentile reporting of
a heavy-tailed error; and a temporal characterisation nobody had measured.
