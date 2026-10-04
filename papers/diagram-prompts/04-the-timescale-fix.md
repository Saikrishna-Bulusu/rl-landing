# Diagram 4 — Why the error correlation number was wrong (0.363 vs 0.707)

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Purpose
Explain why a number measured from a photo dataset had to be converted before it
could be used in a simulation. This is a units/time-scale correction.

## Draw
Two horizontal timelines, one above the other, sharing a time axis in seconds
from 0 to 3 seconds.

### Top timeline - "The DATASET"
- Camera icons placed at 0 s and 1.5 s only (just two, far apart).
- Between them, a double-headed arrow labelled "7.3 metres apart on the ground".
- Second label on the same arrow: "= 1.5 seconds apart, at 5 m/s flying speed".
- Under it: "measured similarity between neighbouring photos = 0.363"

### Bottom timeline - "The REAL DRONE in flight"
- Camera icons at 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s (seven, closely spaced).
- Label: "a position fix every 0.5 seconds"
- Under it: "similarity between neighbouring fixes = 0.707"

## The idea to make obvious
Draw a curve to the right showing similarity decaying with time gap: starts high
near time 0, falls as the gap grows. Mark two points on it:
- at 0.5 s -> 0.707
- at 1.5 s -> 0.363
Label the curve "similarity fades as the time gap grows".

## Caption to include
"Photos taken 1.5 seconds apart are less alike than fixes taken 0.5 seconds
apart. Using the photo number (0.363) in a simulation that updates every 0.5
seconds made the errors look far more random than they really are. Corrected
value: 0.707. This one fix changed the experiment's result from 'no effect' to
'significant effect'."

## Style
Clean timeline figure. Keep the two timelines visually aligned on the same
time axis so the difference in spacing is obvious at a glance.
