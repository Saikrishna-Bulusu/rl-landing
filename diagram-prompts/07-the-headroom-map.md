# Diagram 7 — The headroom map: measure the ceiling before you train

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Purpose
Explain the single most reusable idea from this project: before spending hours
training a policy, measure how much a PERFECT decision could possibly be worth.
If that number is zero, no amount of training can help, and knowing that in
advance saves the hours.

## Draw
A horizontal bar chart with four rows. Each row is one operating condition of a
drone landing task. Each row has TWO bars drawn on the same axis (0% to 100%
"deliveries completed"):

- a solid grey bar = what a simple hand-written rule achieves
- a hollow/outlined bar extending further right = what an agent that is TOLD
  THE CORRECT ANSWER achieves (call this "the oracle")

The gap between the end of the grey bar and the end of the outlined bar is the
HEADROOM. Label that gap on every row with a small bracket and the number.

The four rows, top to bottom:

| condition | rule | oracle | headroom | annotation to print beside the row |
|---|---|---|---|---|
| tight battery, moderate error | 59.0% | 59.5% | 0.5 points | "nothing to learn — both are beaten by flight time" |
| normal battery, moderate error | 99.5% | 100.0% | 0.5 points | "already solved — the rule is enough" |
| normal battery, LARGE error | 90.0% | 100.0% | **10.0 points** | "room to win — train HERE" |
| generous battery, moderate error | 100.0% | 100.0% | 0.0 points | "too easy — nothing left to gain" |

Highlight the third row: give it a coloured background band or a thicker
outline, so the eye lands on it immediately.

## The one sentence to print underneath
"There are two ways to have no headroom — the task is too hard, or it is
already solved. Learning only pays in the band between them."

## Style
Clean, analytical, like a results figure in a journal paper. Off-white
background. Grey for the rule bars, a single accent colour (deep green) for the
oracle outline and the highlighted row. Sans-serif labels. No drone
illustrations, no 3D, no gradients, no clip art. Landscape, roughly 16:9.
