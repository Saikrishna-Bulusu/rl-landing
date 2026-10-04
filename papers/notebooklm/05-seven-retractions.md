# Seven Wrong Answers, and What Caused Each

Over the course of this work, seven confident results were produced and then
withdrawn. Every single one was caused by an unequal comparison or an unchecked
assumption. **Not one was caused by a bug in the physics or the mathematics.**

That pattern is arguably the most transferable thing the project produced, so
each case is recorded here with its cause and its fix.

## One: "The hand-written rule beats the AI by 11.7 points"

The hand-written rules were given a twelve-candidate tuning sweep. The learning
algorithm was given hand-guessed settings and no tuning at all. The rules won.

This measured tuning effort, not method quality. It is exactly the bias described
in a paper called *Leveling the Playing Field*, which re-ran published
"AI beats classical control" results with the tuning asymmetry removed and found
most of the gaps vanished — except here the bias was pointed at the AI instead.

**Fix:** equal twelve-candidate budgets for both sides. The gap closed to a tie
(73.9 versus 74.8). The original settings turned out to be badly wrong; one
version improved from 54.8 percent at two million training steps to 66.3 percent
at six hundred thousand.

**Lesson:** if you tune one method carefully and the other carelessly, you are
measuring your own effort.

## Two: "The error bar beats the stopwatch, p=0.007"

With eight training runs per version, the error bar led by 10.4 points at p=0.007,
with what appeared to be a large effect size.

With twenty runs, the identical comparison gave **+1.8 points at p=0.52**.

It was noise. Eight runs happened to include two lucky ones.

**Fix:** twenty seeds minimum, and report confidence intervals rather than bare
significance.

**Lesson:** eight repeats cannot support a claim in a high-variance setting. The
effect sizes that looked "large" were inflated by the same small sample.

## Three: "The error bar makes training more consistent"

After the eight-seed run, the error bar version showed roughly four times lower
run-to-run variance, which looked like a more robust finding than the mean
difference.

With twenty runs and properly tuned settings, the ordering **reversed** — the
error bar version had the highest variance.

**Cause:** the original comparison used undertuned settings, which inflated the
variance of the other versions.

**Lesson:** a secondary statistic computed from a flawed primary experiment
inherits the flaw.

## Four: "The error correlation is 0.36"

Measured correctly between consecutive dataset photographs. But the drone does
not receive photographs at the dataset's spacing — it receives position fixes
every half second, and closer-in-time readings are more alike.

Converting properly using the measured 7.3-metre frame spacing and a realistic
approach speed gives **0.71**. The raw figure corresponded to flying at about
seventeen metres per second, faster than the simulated aircraft can fly.

**Lesson:** a number measured under one set of conditions is not automatically
valid under another. This assumption had actually been *written down* as a caveat
and then not checked for several hours of compute.

## Five: "The cage results show no difference between versions"

Eighty-four training runs all scored between 17.0 and 17.2 percent — a spread of
0.1 percentage points across twelve different settings and sixteen random seeds.

**Cause:** the cage's control interface expects raw motor speeds, from zero to
about 21,700 revolutions per minute. The learning algorithm emits values roughly
between minus three and plus three. Against a hover requirement of 14,476 RPM,
every action it could physically produce amounted to **0.02 percent of the thrust
needed to stay airborne**.

The drone fell out of the sky on every episode regardless of what the policy did,
and landed within the target radius by luck 17 percent of the time. The policy was
never in the loop at all. Every other check passed — the environment built, reset,
stepped, returned sensible observations, and trained without error.

**Fix:** normalise the control interface, and add a check that asserts the
actions the policy can emit actually span descend, hover, and climb.

**Lesson:** verifying that code runs is not verifying that the agent can affect
the outcome. The diagnostic symptom was visible only in the results: near-identical
scores across genuinely different settings.

## Six: "The error bar significantly improves cage landing, p=0.011"

Covered in detail in the results document. A control version fed a deliberately
meaningless number scored statistically the same, and the best-performing version
of all carried zero information.

**Lesson:** whenever an extra input appears to help, test a same-shaped input with
the information stripped out before believing it.

## Seven: "Removing the episode difficulty structure preserves the measurement"

In redesigning the environment, a mechanism giving each landing a restricted band
of difficulty was removed, because it appeared to halve the relationship being
measured.

That reasoning was wrong. The comparison was against the *pooled* correlation
across all scenes, when the correct target was the *within-scene* correlation.
Removing the mechanism gave every landing access to easy fixes, and the task
saturated: all four versions scored 97 to 100 percent, including the meaningless
control.

**Fix:** measure the real within-scene structure and make each episode a real
scene. A hard scene contains *zero* easy fixes, which is precisely the property
that had been destroyed.

**Lesson:** when a check fails, diagnose which quantity is actually wrong before
removing the mechanism it is complaining about.

## The pattern

Physics bugs announce themselves. Unfair comparisons and unchecked assumptions do
not — they produce clean, plausible, confidently wrong numbers that survive every
test except the one nobody ran.

The practical defences that emerged:

- Give every alternative the same tuning budget, always.
- Twenty repeats, not eight, and report confidence intervals.
- Before any long run, prove the agent can change the outcome — two extreme
  settings must produce different scores.
- Check results partway through, not just at the end. Healthy jobs are not a
  healthy experiment.
- Whenever an input appears to help, run a control with the information removed.
- Measure the assumption you wrote down, before spending compute on it.
