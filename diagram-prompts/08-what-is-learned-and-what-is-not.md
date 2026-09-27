# Diagram 8 — The hybrid: which half is learned, which half is arithmetic

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Purpose
Show that a drone landing system is not one problem. Part of it has a textbook
right answer and should never be learned. Part of it has no formula and is
where learning could earn its place. And a third part must never be learned at
all, for safety.

## Draw
Three horizontal bands stacked vertically, each full width, clearly separated.
Left of each band, a bold label. Inside each band, 2–4 rounded boxes listing
what belongs there.

**Band 1 — "Solved by arithmetic. Do not learn this."** (colour: calm blue)
Boxes:
- "Where is the pad? — average every position fix" (add small text under it:
  "cuts the error from 22 m to 9 m")
- "Fly the aircraft — PX4's own position controller"
- "Centre on the marker — proportional control with braking"

**Band 2 — "No formula exists. This is where learning could pay."** (colour:
deep green)
Boxes:
- "How far to search when the pad is not where I expected"
- "When to stop looking and go home"
- "How much battery to hold back"
Add a small honest footnote inside this band, in italics: "tested, and it has
not beaten the arithmetic yet"

**Band 3 — "Never learned. Hard rules, outside the policy."** (colour: dark
charcoal or deep red)
Boxes:
- "Is this marker the one in MY delivery? — check the ID"
- "Never land on a marker I have not actually seen"
Add a small footnote inside this band, in italics: "kept outside the policy so
no training outcome can erode them"

## The one sentence to print underneath
"Asking a neural network to rediscover an average is the wrong use of the tool.
Ask it only for the judgement that has no formula."

## Style
Flat, clean, diagrammatic. Off-white background, three clearly distinct band
colours, white rounded boxes with thin borders. Sans-serif. No drone pictures,
no icons of brains or robots, no gradients. Landscape, roughly 16:9.
