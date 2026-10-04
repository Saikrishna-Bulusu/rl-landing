# Diagram 10 — One trained policy, three ways to fly it

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Purpose
A trained decision-maker does not output one answer. It outputs ODDS over a
menu of manoeuvres. How you turn those odds into an action matters as much as
the training. Show three read-outs of the SAME odds and what each does to a
drone that is waiting above a delivery pad.

## Draw
Landscape, three stacked horizontal lanes, plus a small panel on the far left.

**Left panel — "What the policy says right now"**
A small vertical bar chart titled "odds at this moment":
- "hold height" — 92% (tall bar)
- "climb at 3 m/s" — 8% (short bar)
Caption under it: "the menu has only 0 m/s or 3 m/s — nothing in between"

**Lane 1 — "Top choice: always take the tallest bar"** (colour: pale green)
A row of small action tiles: hold, hold, hold, hold, hold…
To the right, a tiny altitude-vs-time sketch.
Add a second, smaller example beneath it, in italics: "in a moment where
'climb' edges ahead at 52%": tiles climb, climb, climb, climb… and an altitude
line that shoots up through a red dashed line labelled "120 m legal ceiling".
Result tag: "91.7% clean — ceiling broken on 3.4% of flights"

**Lane 2 — "Roll the dice"** (colour: medium green)
Tiles: hold, hold, hold, climb, hold, hold, hold, hold, hold, climb…
Altitude sketch: a gentle staircase going slowly up.
Result tag: "96.1% clean — but a different random sequence every flight"

**Lane 3 — "Fly the average"** (colour: dark green)
One formula tile: "0.92 × 0 m/s + 0.08 × 3 m/s = 0.24 m/s"
Tiles: 0.24, 0.24, 0.24, 0.24… (all identical)
Altitude sketch: a smooth, gentle straight ramp up.
Result tag: "96.1% clean — the same gain, no dice"

Across the top of all three lanes, a thin reference strip: "the planner it
learned from: 94.3% clean" (colour: orange).

## The one sentence to print underneath
"The trained policy's advantage was a speed that is not on the menu. Taking
only its favourite option throws that away; flying the average keeps it — with
no randomness."

## Numbers source (for the caption, small print)
"Corridor world, 2,048 validation flights, one tuning run. Final multi-seed
test results are reported separately."

## Style
Flat, clean, diagrammatic. Off-white background, three shades of green for the
lanes, orange for the planner strip, red only for the ceiling line. White
rounded tiles with thin borders, sans-serif. No drone pictures, no brains, no
gradients. Landscape, roughly 16:9.
