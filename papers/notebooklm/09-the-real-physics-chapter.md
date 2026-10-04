# Chapter Two: Moving to Real Physics

This covers what happened after the first set of experiments, and why. It is
written for someone who has not seen the code.

## Where we were

The first experiments answered the question — does telling a drone its error bar
help it land? — using a **simplified simulation**. The drone was treated as a
dot moving through the air. There was no spinning of propellers, no tilting, no
weight, no air. Just a point that moved when told to move.

That simulation gave a clear answer across hundreds of experiments: the error bar
made no measurable difference.

## The objection that restarted everything

A fair question was raised: **why use fake physics when a real physics simulator
is available?** A dot-in-the-air cannot tell you whether a real two-kilogram
drone with four spinning propellers would behave the same way.

The original justification had been speed — real physics was supposed to be about
a hundred times slower, making the experiments impossible.

**That justification was checked and it was wrong.** Measured properly:

- simplified simulation: about 45,000 simulated moments per second
- real physics simulation: about 7,000 per second

That is roughly six times slower, not a hundred. Six times is affordable. The
entire reason for avoiding real physics did not survive being measured.

## Building a real drone

The next step was to build the actual aircraft in the physics simulator. The
target was a Holybro X500 — the drone the team genuinely flies.

The first attempt used estimated specifications. Then the real numbers were read
directly from the drone's configuration files already on the laptop:

```
                        estimated      real       error
weight                    1.60 kg    2.06 kg     28% too light
arm length                0.250 m    0.246 m     close
thrust-to-weight ratio       2.30       1.69     36% too optimistic
```

The thrust-to-weight figure is the one that mattered. It describes how much
lifting power the drone has compared to its own weight. A value of 2.3 means a
sporty, agile aircraft. The real 1.69 is far more sluggish, with much less margin
for aggressive manoeuvres. Every control setting tuned against the estimate had
been tuned for a livelier aircraft than actually exists.

**Lesson: read the specification file before assuming the specification.**

## Teaching it to fly: four bugs

A physics simulator gives you an aircraft that obeys physics. It does not give
you an aircraft that flies. Something must decide how fast each of the four
propellers should spin, moment by moment. Building that revealed four faults.

**1. The supplied "velocity controller" was not one.** Asked to descend at 0.3
metres per second, it achieved 1% of that. Internally it aimed at a point one
fiftieth of a second ahead — a target six millimetres away — which is far too
small a nudge to move a two-kilogram aircraft.

**2. The steering controls were mislabelled.** The channel named "roll" actually
controlled pitch, and backwards. The channel named "pitch" controlled roll, also
backwards. Commanding them by their names meant every correction made the error
worse. The aircraft flipped completely over within a second of any turn.

This was found by applying a small test command and watching what actually moved,
rather than trusting the labels.

**3. A "fix" that made things worse.** The steering sensitivity was increased by
a factor of 381, reasoning that a heavier aircraft needs stronger corrections.
Sideways speed appeared to improve — from 0.76 to 1.46 metres per second — which
looked like success.

The aircraft was tumbling. It was moving quickly because it was **falling**, not
flying. Tilt reached 180 degrees and it dropped from ten metres to under one.

**A speed measurement cannot tell the difference between flying somewhere and
falling there.** The original settings had been correct all along.

**4. Attitude commands sent to a position interface.** The test environment was
configured to accept target positions, while being fed steering commands. It
read the numbers as coordinates and obediently flew to the point (0,0,0) — the
ground. Several rounds were spent blaming the controller for this.

## A fifth mistake, repeated

Every controller test used a **constant** command: fly north at half a metre per
second, hold it, measure. That kind of test can never reveal an unstable control
loop, because an unstable loop only misbehaves when it has to **stop**.

Once tested properly — fly to a point and halt — the controller was visibly
oscillating: overshoot the target, swing back past it, overshoot again, and
tumble. Constant-command testing had certified a controller that could not
actually arrive anywhere.

Once corrected, it flew to targets with about three centimetres of error, tilting
under ten degrees.

## A genuine physics discovery

The fixed controller descended beautifully — and then refused to land.

It stopped about 30 centimetres above the ground and hovered there indefinitely.
This is **ground effect**: close to a surface, a hovering aircraft's own
downwash bounces back and provides extra lift. Real helicopters and drones do
exactly this.

Measurements showed the aircraft settling at whatever height balanced the
commanded descent:

```
commanded descent   settles at   touching ground?
   0.15 m/s          0.60 m           no
   0.30 m/s          0.42 m           no
   0.50 m/s          0.32 m           no
   0.80 m/s          0.24 m           no
```

At none of these was it actually touching down. Cutting the motors dropped it to
9 centimetres with four points of ground contact — genuinely resting.

**So the last twenty centimetres cannot be flown. They require deliberately
cutting power once the aircraft is low and steady** — which is precisely what
real drone autopilots do before shutting down. That logic was added, and the
aircraft now lands properly: on target, at about 0.2 metres per second.

For comparison, the old simplified simulation had every landing arriving at
**2.0 metres per second** — because nothing in it cared about impact speed. On a
real airframe that is a breakage, not a landing.

## Three more self-inflicted problems

With the aircraft flying, training began. It failed. The cause was three errors
in the scoring system used to teach the drone.

**Error one: tilting was punished too heavily.** A small penalty was applied for
every moment the drone was tilted. Tilting is how a drone moves sideways, so this
was meant to discourage wild flailing. But applied every moment, a gentle
ten-degree lean accumulated to a penalty of **-79** across a landing, against a
reward of only **+40 to +100** for landing successfully.

So the drone worked out something perfectly rational: never tilt at all. Descend
straight down, land fourteen metres from the target, and keep the tilt savings.
It could **see** the landing pad and deliberately refused to approach it.

**Error two: no reward for getting closer.** Once the marker was in view, nothing
rewarded reducing the distance to it. There was no pull toward the target.

**Error three: loitering paid as well as landing.** Hovering in view of the
marker earned about +123 across an episode. Landing earned +40 to +100. Not
landing was the better-paying strategy.

Each was found by adding up what a behaviour earns across a whole flight, rather
than looking at the individual rules. The rules looked sensible. The economics
did not.

**Lesson: when copying a working system, copy what behaviours are WORTH, not
just the list of rules.** A fourth error came from over-correcting error three
— cutting the loitering reward so far that finding the marker became worthless.

## The problem that has no easy fix

With everything corrected, a new problem appeared, and this one is not a bug.

Running the identical setup six times, changing nothing but the random starting
conditions, produced:

```
success rates:  20%   14%   1%   0%   52%   44%
```

Some attempts learned the task. Others failed completely. This is ordinary in
this kind of machine learning, but the spread here is extreme.

More training was tried. At four times the budget, every failure recovered —
nothing collapsed to zero any more. But the spread did not shrink:

```
                mean    spread
short training   22%      22
4x training      55%      21      best
8x training      41%      22      worse than 4x
```

Eight times the training was **worse** than four times, and one attempt that had
reached 61% fell back to 9%. It learned the task and then unlearned it.

## Why this matters for the question

To detect a difference between two versions, the difference has to be larger than
the random variation between attempts. With variation this large:

```
to detect a 20-point difference:   19 attempts per version
to detect a 10-point difference:   76 attempts per version
to detect a  6-point difference:  212 attempts per version
```

The error-bar effects measured in the simplified simulation were **0 to 6 points**.
Detecting that here would take over two hundred attempts per version — roughly
eight hundred training runs, several days of computing.

**So the real-physics simulation, as it stands, cannot answer the question it was
built to answer.** It is not that the answer is no. It is that the measuring
instrument is too imprecise to read.

The current step is testing one remaining idea: running thirty-two simultaneous
practice environments instead of eight. More simultaneous practice means less
noisy learning, which is the standard remedy for exactly this symptom — and the
laptop has been running at a fraction of its capacity throughout.

## What the real-physics work delivered regardless

Even if the error-bar measurement proves impossible here, this chapter produced
things the simplified simulation never could:

- a correctly-specified X500 aircraft matching the team's own configuration files
- a controller that flies to a point, stops, and lands gently at 0.1-0.3 metres
  per second instead of crashing at 2.0
- an environment with genuine ground effect, obstacles, and a real camera model
- a demonstrable landing that can be shown as video rather than described

Those are what a transfer to the full flight simulator actually requires.

## The pattern across the whole project

Ten confident results have now been withdrawn. Every one came from an unfair
comparison, an unchecked assumption, or a self-inflicted scoring error. **Not one
came from a mistake in the physics or the mathematics.**

The defences that emerged, in order of how much they have saved:

1. Give every alternative the same tuning effort.
2. Twenty repeats, not eight, and always state the uncertainty range.
3. Before a long run, prove the learner can change the outcome at all.
4. Check results partway through. Healthy processes are not a healthy experiment.
5. When something appears to help, test a version with the information removed.
6. Measure the assumption you wrote down before spending compute on it.
7. Test a controller by making it stop, not by making it go.
