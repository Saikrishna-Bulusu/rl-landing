# How the Simulation Works, and Why It Was Built That Way

## Two environments, deliberately different

**The point-mass environment.** A drone modelled as a simple mass moving under a
velocity controller, descending from about fifty metres toward a pad. No rotor
physics, no attitude. This sounds like a shortcut but it follows published
practice: a comparable system (WaveLander) trains its landing policy on a single
sliding joint constrained to vertical motion, with no quadrotor dynamics at all.
At fifty metres altitude under a velocity controller, aircraft attitude
contributes nothing to a decision about whether to search or commit. The payoff
is that it trains in minutes instead of hours.

**The cage environment.** A full physics simulation in MuJoCo of a small drone
inside a three-by-two-by-three metre cage, landing on a moving pad, controlled by
commanding individual motor speeds. This is much harder: the software must learn
to fly *and* to land.

## The central design principle: replay, do not simulate

The single most important decision is that the position errors are **replayed
from real measurements**, never invented.

The tempting shortcut is to model the error as a bell curve with some standard
deviation. That would break the thing being measured. Calibration does not
transfer for free: the whole question is whether the reported error bar matches
the real error, and inventing both sides of that relationship guarantees a
flattering answer.

Instead, the simulation holds a pool of 440 real (error bar, actual error) pairs
measured from GeoAnchor running on real imagery. Every position fix the simulated
drone receives is a real measurement pulled from that pool. The relationship
between the claimed uncertainty and the actual error is therefore exactly what
was measured, by construction.

## Errors are correlated in time, and that matters enormously

Early versions drew a fresh, independent error every half second. This turned out
to be catastrophic, and finding out why was one of the more instructive moments
of the project.

If each reading is an independent roll of the dice, the winning strategy is to
hover in place and wait for a lucky roll. Sooner or later a fix arrives that is
nearly perfect, and the drone pounces. That is a lottery, not a navigation
problem — and no amount of knowing your error bar helps you win a lottery.

This was proved directly. Freezing the error so it stopped changing made hovering
and diving score identically — 71.8 percent both. Every bit of the advantage that
hovering appeared to confer came from waiting for a lucky draw.

Real errors are not independent. Consecutive frames in the dataset were used to
measure this: the rank correlation between one error and the next is about 0.36.
The simulation was rebuilt to reproduce that.

## A subtlety about time scales that changed the answer

The 0.36 figure carried a trap. It was measured between consecutive *photographs
in the dataset*, which are several metres apart on the ground. A flying drone
receives position fixes every half second, and readings closer together in time
are more alike than readings further apart.

Measuring the spacing between dataset frames — about 7.3 metres — and dividing by
a realistic approach speed of five metres per second gives roughly 1.5 seconds
between frames. Rescaling the correlation to the half-second interval the drone
actually experiences gives approximately 0.71, not 0.36.

The raw number was a substantial underestimate. Using it corresponded to flying
at about seventeen metres per second, which is faster than the aircraft in the
simulation can physically travel.

## Preserving the joint distribution exactly

The technique used to introduce time correlation without distorting the measured
error distribution is a Gaussian copula. In plain terms: a hidden, smoothly
drifting number is used to *choose which real measurement to replay*, rather than
to generate an error directly. Because the hidden number is uniformly distributed
after transformation, the replayed errors have exactly the distribution that was
measured — while gaining realistic time structure.

This is asserted in the code as a runnable check: the replayed error percentiles
must match the measured ones, and the achieved correlation must match the target.

## An episode is a scene

The final version of the point-mass environment makes each landing attempt sample
a real *scene* from the dataset, and draws that landing's position fixes from that
scene's own measurements.

This followed from a measurement. The four usable scenes differ dramatically:

- One scene has a median error bar of 11.4 metres with very little spread, and
  over a third of its fixes are high quality.
- Another has a median of 27.9 metres, spread very wide, and **zero** high-quality
  fixes.

That last property is what makes a landing genuinely hard. In a difficult scene
there is no lucky fix to wait for. An earlier design that let every landing draw
from the whole pool gave every episode access to easy fixes, and the task
saturated — all versions succeeded 97 to 100 percent of the time, including a
version fed a deliberately meaningless number. A saturated task measures nothing.

## The honest difficulty levers

When the task proves too easy, only two things may be adjusted, and both are
physically meaningful: the battery reserve (how long the drone may loiter) and
the airframe's maximum lateral speed.

The camera constants — field of view, resolution, minimum readable marker size,
marker dimensions — are explicitly off limits. They are the physics of seeing a
one-metre marker, and they set the entire structure of the problem. Adjusting
them to make training easier would be adjusting reality to suit the result.
