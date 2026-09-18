# Diagram 2 — The three versions being compared (the experiment)

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Draw
Three identical drone "control boxes" side by side, to show that only ONE input
differs between them. This is the whole experiment in one picture.

## Layout
Three vertical panels, identical in every way except the middle input arrow.

### Panel 1 - "none"
Inputs flowing into a box labelled "landing policy":
- "altitude"
- "believed distance to pad"
- "can I see the marker? yes/no"
- "time elapsed"
Output arrow: "how fast to descend / where to fly"

### Panel 2 - "staleness"
Exactly the same inputs, PLUS one extra arrow drawn in ORANGE:
- "seconds since last position update" with a small stopwatch icon
Label this panel's extra input: "PerchRL's method - a stopwatch"

### Panel 3 - "sigma"
Exactly the same inputs, PLUS one extra arrow drawn in GREEN:
- "position error bar, in metres (e.g. +/- 21 m)" with a small ruler icon
Label this panel's extra input: "GeoAnchor's method - a calibrated error bar"

## Make this unmissable
Put a banner across the top: "IDENTICAL IN EVERY WAY EXCEPT ONE INPUT".
Draw the shared inputs in grey in all three panels so the orange and green
arrows are the only things that stand out.

## Caption to include
"A stopwatch tells you the estimate is OLD. An error bar tells you the estimate
is WRONG BY 21 METRES. The experiment asks whether that difference matters."

## Style
Flat diagram, boxes and arrows, generous whitespace. No drone artwork needed.
