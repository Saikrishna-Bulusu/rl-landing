# Diagram 11 — One decision layer, three test beds (the paper's system figure)

Ask Gemini to draw this. Everything it needs is below; it has no other context.

## Purpose
The paper's Figure 2: how a delivery drone's last-mile decisions are made, what is learned and what is
not, and where each part is tested. A reader should see in five seconds that the learned part is small
and boxed in: it only CHOOSES among options a planner has already checked, and hard safety rules sit
outside it.

## Draw
A left-to-right pipeline of five boxes, joined by arrows, then a row of three "test bed" panels beneath.

**Top row, left to right:**

1. **Sensing** (a tall box, three stacked items inside):
   - "Position: GPS, or GeoAnchor (camera-to-map fixes)"
   - "Map: LiDAR survey (houses, trees) + voxel map of what the survey missed"
   - "Cameras: landing camera (ArUco pad), forward camera, six-camera rig → bird detector → tracker"
2. **Belief** (small box): "where I am, how sure I am (σ), what is around me: people, helicopter,
   other drones, the magpie"
3. **Planner** (box): "14 candidate moves, each rolled 20 s ahead against the forecast; flags any that
   break a rule (CASA 30 m from people, helipad duty, separation, strike)"
4. **Learner** (box drawn in the ACCENT colour, the only one): "picks one of the 14 — a small neural
   network that first copies the planner, then practises in simulation (PPO)"
5. **Flight controller** (box): "PX4 offboard velocity (x500) · Tello `rc` command"

Between box 4 and box 5, draw a narrow vertical bar across the arrow labelled **"Hard safety layer
(not learned)"**, with three short lines on it:
   - "land only on the customer's ArUco ID"
   - "dodge reflex for a diving bird"
   - "once locked on the pad, commit and land"

**Bottom row, three panels side by side, each with a small title and two lines of text:**

| panel | title | lines |
|---|---|---|
| left | "Fast GPU simulator (Logan, QLD)" | "trains the learner: 400,000 flights per run" / "tests: 4,096 flights per decider, paired" |
| middle | "PX4 + Gazebo" | "live flights: real autopilot, rendered cameras" / "weather, magpie, the rig's detector in the loop" |
| right | "DJI Tello in a 3 × 2 × 2 m cage" | "real flights: tiny houses and trees, ArUco pads" / "downward view through a 45° mirror" |

Draw thin dashed lines from the Learner box down to all three panels, labelled "same decision layer".

## The one sentence to print underneath
"Learning only chooses among checked options; what it may never do is enforced outside it."

## Style
Clean, analytical, like a system figure in a robotics paper (IEEE / ACRA). White background. All boxes
light grey with dark grey outlines, EXCEPT the Learner box in one accent colour (deep orange, #d35400)
and the hard-safety bar in dark red outline. Sans-serif labels, no drone illustrations, no icons, no 3D,
no gradients, no clip art. Landscape, width about 2.5 × height (it will span both columns of the paper).
