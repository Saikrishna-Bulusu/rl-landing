# Can a Drone Land Better If It Knows How Lost It Is?

## The one-sentence question

When a delivery drone has no GPS and must work out its own position by matching
its camera view against a satellite map, does telling the landing software *how
wrong that position estimate probably is* make it land more reliably?

## Why this matters

A delivery drone arriving over a suburb needs to put a package on a small pad in
someone's back garden. If GPS is unavailable — jammed, blocked by buildings, or
simply switched off in a denied environment — the drone must localise itself
visually. The system doing that here is called GeoAnchor: it matches what the
drone's camera sees against satellite imagery and returns an estimated position.

That estimate is wrong by some amount. Sometimes five metres, sometimes forty.

Critically, GeoAnchor can also report *how* wrong it probably is — an error bar,
measured in metres, called sigma. A fix might come back as "you are here, give or
take twenty-one metres."

The research question is whether that error bar is worth anything to the landing
policy. Intuitively it should be. If the drone knows its position could be off by
forty metres, it should hesitate before committing to a descent. If it knows the
estimate is good to three metres, it should commit immediately and save battery.

## The three versions being compared

The experiment trains three versions of the same landing software. They are
identical in every respect — same physics, same camera, same reward, same
training budget — except for one extra number fed into the decision-making:

- **"none"** — the policy is told only the estimated position.
- **"staleness"** — the policy is additionally told how many seconds have passed
  since the last position update. This is a stopwatch. It is the approach used by
  a competing published system (PerchRL), which reported that such a reliability
  signal improved its performance.
- **"sigma"** — the policy is additionally told the calibrated error bar in
  metres. This is GeoAnchor's contribution.

The claim under test is specific: a *calibrated, metre-valued* uncertainty should
beat a *stopwatch*. A stopwatch tells you the estimate is old. An error bar tells
you it is wrong by twenty-one metres. Those are different kinds of information.

## What makes the landing hard

Three physical facts create the difficulty, and none of them were tuned for
convenience — they follow from the camera and the marker.

**The pad is invisible from high up.** The landing pad carries a one-metre visual
marker. At 640 pixels of camera resolution across a 60-degree field of view, that
marker spans fewer than twenty pixels above roughly thirty metres of altitude,
and cannot be decoded. So there is a hard ceiling — call it the detection
ceiling — above which the drone is flying blind.

**The view shrinks as you descend.** The camera sees a circle on the ground whose
radius is proportional to altitude. At thirty metres the drone can see a circle
of about seventeen metres radius. At ten metres, under six. So diving the moment
the marker appears can drive it straight back out of frame.

**The error is in the drone's own position, not the pad's.** The pad is a
delivery coordinate, known exactly. What is uncertain is where the *drone* is. If
the drone wrongly believes it sits twenty metres north of its true position, it
will compute the pad as being twenty metres further south than it really is. This
sign is easy to get backwards and doing so destroys the experiment.

## The gap this creates

Put those together and there is a genuine decision to make. The drone knows
roughly where it is, to within ten to forty metres. The pad is invisible until it
descends below thirty metres. Descending costs battery and shrinks the view. So:

- Descend now, or hold position and wait for a better position fix?
- Fly straight to the believed location, or search around it?
- And if the ground arrives and the pad was never found, that is a package
  delivered into somebody's swimming pool.

A policy that is told only a position cannot distinguish a three-metre error bar
from a forty-metre one. It must behave identically in both. That is the
hypothesis this whole project exists to test.

## The headline answer

Across three independent experiments — two simulation environments, hundreds of
training runs, and a deliberately designed control experiment — a calibrated
error bar produced no measurable improvement over either alternative.

But the most important result is not the negative itself. It is the explanation
that emerged for *why*, which is covered in the companion document on findings:
at GeoAnchor's real accuracy, roughly three quarters of position fixes already
place the drone inside the camera's field of view. A policy that simply dives
succeeds about ninety-six percent of the time. There is nothing for
uncertainty-awareness to improve, because the localisation is already good
enough that the decision does not need to be made carefully.
