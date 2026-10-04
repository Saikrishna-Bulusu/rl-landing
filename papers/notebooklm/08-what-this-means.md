# What This Work Actually Established

## The defensible claims

**One. GeoAnchor's error bars are honest, and the honesty is quantified.**
ENCE of 0.313, and a distribution-free guarantee that the true position lies
within 1.97 times the reported error bar at least 90 percent of the time, with
measured held-out coverage of exactly 90.0 percent. This uses only data that
already existed and does not depend on any of the landing work.

**Two. A calibrated error bar gives a learned landing policy no measurable
advantage — at GeoAnchor's current accuracy.** Demonstrated across two
independent simulation environments, three error-correlation settings, twenty
repeats per version, with confidence intervals. Any advantage larger than roughly
five to fifteen percentage points is excluded at 95 percent confidence.

**Three. The reason is that the localisation is already good enough.** Three
quarters of real fixes place the drone inside the camera's field of view from the
detection ceiling. A policy that simply dives succeeds 96 percent of the time.
There is no decision left to make carefully.

**Four. The threshold is quantified.** Localisation would need to degrade by
roughly 1.5 to 2 times before the landing task becomes a genuine
search-versus-commit decision, and therefore before a calibrated error bar could
begin to pay for itself.

**Five. A hand-written rule beats every learned version, under equal tuning.**
Both sides received identical twelve-candidate tuning budgets. This echoes a
finding in the underlying localisation work, where a simple inlier threshold beat
everything learned.

**Six. A hand-written rule cannot exploit the error bar at all.** At every
correlation setting tested, a rule permitted to scale its search by the error bar
tuned that term to zero, collapsing into the version that ignores it. Four
settings, four times.

**Seven. The policy reads the error bar but gains nothing from it.** It responds
to the error bar almost as strongly as to altitude, and exactly as strongly as to
a number known to be meaningless. So the failure is not that the network ignored
the input — it used it, and using it changed behaviour without improving outcomes.

## What is explicitly NOT established

**Not that uncertainty-awareness is useless in general.** Everything here concerns
one specific task at one specific accuracy level. The degradation sweep suggests
the opposite conclusion would hold in a harder regime.

**Not that PerchRL's approach is refuted.** Their stopwatch proxy does not help
here, but in the simplest version of this environment the error bar is fixed for
a whole landing, which makes a stopwatch mathematically incapable of carrying
information about the error. That is a property of the test design, not evidence
against their method.

**Not that the effect is absent — only that it is not demonstrated.** The
confidence intervals admit an advantage of up to about six points. Ruling out a
two-point effect would require roughly 170 repeats per version.

**Not established for real hardware.** Everything is simulated. The natural next
step tests a genuinely different use of the error bar: handing it to the drone's
onboard state estimator as a measure of how much to trust each fix. That is what
error bars are conventionally for, and it is a different question from feeding one
to a decision-making policy.

## Why a negative result here is worth having

The question "does a calibrated uncertainty help a landing policy?" had an
assumed answer. It seemed obviously yes.

The work shows the answer is no *in the regime that actually matters*, and —
more usefully — explains why, and says precisely how much would have to change
for the answer to flip. That is a far stronger contribution than a marginal
positive result would have been.

It also produced a methodological finding with broader reach. Seven confident
results were generated and withdrawn during this work. Every one came from an
unequal comparison or an unchecked assumption. None came from a bug in the
physics or the mathematics. The defences that emerged — equal tuning budgets,
twenty repeats minimum, proving the agent can affect the outcome before spending
compute, checking results mid-run rather than only at the end, and running a
control with the information stripped out whenever an input appears to help —
are general, and would have caught every one of the seven.
