# Diagram 3 — The bug I found in the simulation (lottery vs. persistent error)

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Purpose
Show two different ways a position error can behave over time, and why the
difference completely changes the best strategy.

## Draw
Two stacked timeline panels. Each has a horizontal time axis (0 to 10 seconds)
and shows the drone's BELIEVED pad position as a series of dots.

### Top panel - "ORIGINAL SIMULATION: fresh random error every 0.5 s"
- The true pad drawn as a green star on a horizontal centre line.
- Twenty dots scattered at different distances from the star, jumping wildly
  between each time step - some 40 m out, some almost on top of the star.
- Circle one dot that lands very close to the star. Label it "a lucky reading".
- Annotation: "Best strategy: HOVER AND WAIT for a lucky reading."
- Annotation in red: "This is a lottery, not a navigation problem."

### Bottom panel - "REALITY: the error persists"
- Same green star.
- Twenty dots that all sit roughly 15 m to the right of the star, drifting only
  slightly. None of them lands near the star.
- Annotation: "No lucky readings ever come."
- Annotation in green: "Best strategy: SEARCH the area, or use the error bar to
  decide how wide to look."

## Bottom bar - the evidence
A small two-row table:
"                    dive straight down    hover and wait
 fresh error each time      86.5%              93.5%
 error persists             71.8%              71.8%   <- identical"
Caption under table: "When the error persists, hovering stops helping at all.
That proved the original simulation was rewarding luck, not skill."

## Style
Clean scientific figure. Green star for truth, blue dots for beliefs.
