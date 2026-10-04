# Diagram 1 — What the landing problem actually is

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Draw
A side-on view of a drone descending toward a landing pad, plus a top-down inset.

## Side view (main panel)
- A quadcopter drone at 50 metres altitude, top left.
- The ground at the bottom, drawn as a suburban backyard.
- A small square landing pad on the ground, 1 metre across. Label it
  "ArUco marker pad, 1 m".
- A cone spreading downward from the drone's belly camera, 60 degrees wide.
  Label it "camera field of view, 60 degrees".
- A horizontal dashed line across the whole image at 30.6 metres altitude.
  Label it "DETECTION CEILING - 30.6 m".
- Above that line, annotate: "marker is smaller than 20 pixels - INVISIBLE".
- Below that line, annotate: "marker is readable".
- Where the camera cone meets the ground at 30 m altitude, mark the circle's
  radius as 17.3 m. Label: "camera footprint radius at 30 m = 17.3 m".

## Key idea to make obvious
As the drone descends, the footprint circle SHRINKS. Show this with two faint
cone outlines: a wide one at 30 m and a narrow one at 10 m, with a small arrow
labelled "footprint shrinks as you descend".

## Top-down inset (small panel, top right)
- A dot labelled "where the drone THINKS it is".
- A second dot labelled "where the drone ACTUALLY is", about 20 mm away.
- A shaded circle around the believed position, radius labelled "sigma - the
  error bar, 10 to 40 m".
- The pad drawn as a tiny square somewhere inside that circle.

## Caption to include
"The drone knows roughly where it is, but 'roughly' is 10-40 metres. The pad is
only visible below 30.6 m, and the lower it goes the less ground it can see.
Descend too early in the wrong place and the pad leaves the frame."

## Style
Clean technical illustration, flat colours, clear labels, no 3D rendering.
