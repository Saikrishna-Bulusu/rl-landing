# Diagram 9 — Why one marker cannot cover the whole descent

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Purpose
Show a real geometric bind we measured: a marker big enough to SPOT from high
up is too big to SEE when you are close, because it stops fitting in the
camera's view. The fix is two markers of different sizes on the same pad.

## Draw
A vertical altitude axis on the left, running from 20 metres at the top down to
0.2 metres at the bottom, with tick marks at 20, 18, 12, 6, 3, 1, 0.5, 0.25 m.

To the right of the axis, two vertical "usable range" bars:

- **Bar A, labelled "big marker, 1.2 m wide"** — coloured solid from 18 m down
  to 0.8 m. Above 18 m, fade it out and annotate "too few pixels to decode".
  Below 0.8 m, fade it out and annotate "larger than the camera's whole view".
- **Bar B, labelled "small marker, 0.15 m wide"** — coloured solid from 3 m
  down to 0.25 m. Above 3 m, fade out and annotate "too few pixels to decode".

Draw a shaded horizontal band across the region from 3 m down to 0.8 m where
BOTH bars are solid, and label it "handover — both visible at once".

Beside the axis at three altitudes, draw a small square representing what the
camera sees, to make the point visually:
- at 18 m: a small dark square occupying a tiny fraction of a frame outline
- at 3 m: a comfortable square filling maybe a third of the frame
- at 0.5 m: a square BURSTING OUT past the edges of the frame outline, with its
  corners cut off by the frame — annotate "corners outside the frame, cannot be
  decoded"

## The one sentence to print underneath
"Big enough to find is too big to land on. The pad carries both, and the drone
switches markers on the way down."

## Style
Technical illustration, clean and precise, like a figure in an engineering
report. Off-white background. Two distinct colours for the two bars, grey for
the faded-out regions, a warm accent for the handover band. Sans-serif labels.
No photorealism, no drone renderings. Portrait or square orientation.
