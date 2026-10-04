# The Results

All figures below are from real data, not synthetic. Where a result was later
withdrawn, it is shown alongside what replaced it, because the pattern of
withdrawals is itself one of the findings.

## The precondition: is the question even answerable?

Before training anything, a check: does the error bar carry usable signal at all?
If a policy that simply dives straight down succeeds equally whether the error bar
is large or small, then the error bar is uninformative and no amount of training
will produce a result.

**Diving succeeded 98 percent of the time when the error bar was small, and 42
percent when it was large.** A 56-point gap. The question was answerable.

## Experiment one: the point-mass environment

Twenty independent training runs per version, at three different levels of error
time-correlation. The measure reported is success on the hardest third of
episodes, since every version succeeds when uncertainty is low and the versions
can only separate where it is high.

```
                 none    staleness   sigma    hand-written rule
correlation 0.36  69.3      69.7      70.4          74.8
correlation 0.71  58.5      60.1      58.2          81.9
correlation 0.80  52.1      55.2      57.0          83.5
```

Differences between the three learned versions, with statistical significance:

```
                    sigma vs none        sigma vs stopwatch
correlation 0.36     +1.1  p=0.76          +0.7  p=0.84
correlation 0.71     -0.4  p=0.89          -2.0  p=0.52
correlation 0.80     +4.9  p=0.32          +1.8  p=0.52
```

No effect reaches significance at any operating point.

Stated as confidence intervals, which is the honest way to express a null result:
any advantage for the error bar larger than roughly five to fifteen percentage
points is excluded at 95 percent confidence. Meanwhile a simple hand-written rule
beats every learned version by 5, 22, and 27 points respectively.

This is a well-powered negative, not an absence of evidence.

## Experiment two: the cage environment

Sixteen training runs per version in full physics, landing inside a three-metre
cage on a pad with a 25-centimetre success radius.

```
version      success    what the extra channel actually contains
none           7.8%     nothing
clock         10.1%     CONTROL: bare elapsed time
random        11.5%     CONTROL: a number unrelated to the error
sigma         13.2%     the real error bar
staleness     14.9%     a stopwatch
```

At first reading this looks like a win: the error bar beat the empty version by
5.4 points with p=0.011, which passes the usual significance threshold.

## The control experiment that overturned it

Two extra versions were added specifically to try to destroy that result.

The **random** version receives a number with exactly the same distribution as the
error bar — same median, same range — but drawn independently, so it correlates
+0.016 with the drone's actual error, against the real error bar's +0.549. It is,
by construction, meaningless.

```
sigma vs random    +1.7 points    p=0.397    not significant
```

**The error bar is not distinguishable from a meaningless number.**

Worse for the original interpretation: the best-performing version of all five is
the stopwatch at 14.9 percent — and the stopwatch was separately proved to carry
zero information. Its value is byte-identical in every single episode, and
correlates +0.003 with the actual error. The only genuinely informative channel,
the error bar, ranked third of five.

So the apparent success was an artefact of the network simply having one more
input to condition on, not of that input carrying uncertainty information.

An analogy: students given a lucky hat scored better than students given nothing.
But students given a *random* hat scored just as well. The hat is not teaching
physics.

## The diagnostic: did the policy ignore the error bar?

The natural explanation for a null result is that the network learned to ignore an
input it found useless. That was tested directly, with no new training: take the
trained policies, feed them identical situations differing only in the error bar
value, and measure how much the commanded action changes.

```
point-mass, 10 trained policies
  error bar channel      action change 0.2841
  altitude               action change 0.3159   (error bar is 0.90x this)
  distance to pad        action change 0.3669   (error bar is 0.77x this)
```

The policy responds to the error bar almost as strongly as to altitude — an input
it cannot possibly perform the task without. So it did not ignore it.

```
cage, 16 policies per version
  error bar channel (informative)    0.1507
  random channel (meaningless)       0.1733
  difference -0.0227, p=0.124        no difference
```

But it responds just as strongly to a meaningless number. The network does not
learn to discount uninformative inputs; it conditions on whatever it is given.

**The conclusion: the policy read the error bar and acted on it. Acting on it
changed behaviour without improving outcomes.**

That relocates the failure precisely. It is not a perception failure — the channel
is read. It is not a capacity failure — the network conditions on it fine.

## The explanation, and the most important number in the project

Measuring the task itself, rather than the policies, produced the answer.

```
camera footprint radius at the detection ceiling = 17.7 metres
fraction of real GeoAnchor fixes with error below 17.7 metres = 75.9%
```

**Three quarters of real position fixes already place the drone inside the
camera's field of view.** A policy that simply dives succeeds 96 percent of the
time.

There is nothing for uncertainty-awareness to improve, because the localisation
is already accurate enough that the decision does not have to be made carefully.

How much worse would it need to be?

```
error scale   median error   dive succeeds   hold-and-wait succeeds
      1.0x        11.6 m         96.4%             100.0%
      1.5x        17.4 m         62.8%              95.2%
      2.0x        23.2 m         29.6%              60.0%
      3.0x        34.8 m          3.6%              12.0%
```

The task only becomes a genuine decision problem when localisation is roughly 1.5
to 2 times worse than GeoAnchor actually achieves.

This converts the whole project from a null result into a quantified statement:
**a calibrated error bar would begin to earn its keep only if localisation
accuracy degraded by about half as much again.**
