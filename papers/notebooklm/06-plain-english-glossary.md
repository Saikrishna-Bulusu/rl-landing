# Plain English Glossary

Every technical term used in this project, explained without jargon.

**Sigma, or the error bar.** A number, in metres, saying how wrong the position
estimate probably is. "You are here, give or take twenty-one metres."

**Calibrated.** The error bar is honest. If it claims twenty-one metres, the real
error really is about twenty-one metres — not five, and not sixty. An uncalibrated
error bar is worse than none at all, because it makes the system confident at
exactly the wrong moments.

**GeoAnchor.** The system that works out where the drone is by matching its camera
view against satellite imagery, when GPS is unavailable.

**Policy.** The landing software — the thing that decides how fast to descend and
where to fly. In this project it is a small neural network.

**PPO.** The training method. The software attempts thousands of landings and
gradually improves. Nobody programs the rules; it works them out from experience.

**Arm.** One version being tested. Three main versions here, differing by a single
input.

**Ablation.** Removing one thing to see whether it mattered. Like cooking a dish
without salt to find out what the salt was doing.

**Hyperparameter.** A setting chosen before training begins — how fast to adjust,
how large the network is. Like oven temperature: not the recipe, but it changes
the outcome substantially.

**Seed.** One repeat of the experiment. Training involves randomness, so the same
setup run twice gives different results. Reporting a single seed is like reporting
one coin flip.

**p-value.** The probability of seeing a gap this large **by pure chance**, if
there were genuinely no difference. A p-value of 0.006 means roughly a one-in-170
chance of a fluke. Below 0.05 is the conventional threshold for "probably real."

**Effect size.** How *large* a difference is, separately from how *confident* we
are it exists. A difference can be statistically certain but too small to care
about, or large but uncertain.

**Confidence interval.** A range of values consistent with the data. The honest
way to report a null result: rather than "we found nothing," say "any effect
larger than six points is ruled out."

**Bonferroni correction.** When you run several statistical tests at once, one of
them will look significant by chance. This tightens the threshold to compensate.

**Autocorrelation.** How much one reading resembles the next. Zero means every
reading is independent random noise. One means the reading never changes.

**Rank correlation.** A correlation measured on the *ordering* of values rather
than the values themselves. Used here because the error distribution has a heavy
tail, and a single enormous error would otherwise dominate an ordinary
correlation.

**Gaussian copula.** A technique for adding realistic time-structure to replayed
data without distorting its distribution. A hidden, smoothly drifting number is
used to select *which* real measurement to replay, rather than to generate a
value directly.

**ENCE.** A score for "is the error bar honest?" Zero is perfect. Groups fixes by
their claimed uncertainty and compares claimed against observed within each group.

**Conformal calibration.** A method for producing a coverage guarantee — "the truth
is within 1.97 times the error bar, 90 percent of the time" — that makes no
assumption about the shape of the error distribution. Important here, since the
errors are emphatically not bell-shaped.

**Baseline.** A simple hand-written rule, not AI. The thing the learned method must
beat to justify its existence.

**Control arm.** A version deliberately fed useless information, included to check
whether an apparent improvement is real or just an artefact of having an extra
input. The scientific equivalent of a placebo.

**Saturation.** When a task is so easy that every version succeeds and no
comparison is possible. A saturated experiment measures nothing.

**Floor effect.** The opposite: the task is so hard that every version fails, and
again no comparison is possible.

**Detection ceiling.** The altitude above which the landing marker is too few
pixels across to be read — about 30.6 metres here, for a one-metre marker at 640
pixels and a 60-degree field of view.

**Camera footprint.** The circle on the ground the camera can see. Its radius is
proportional to altitude, which means it *shrinks* as the drone descends — so
diving the moment the marker appears can push it back out of frame.
