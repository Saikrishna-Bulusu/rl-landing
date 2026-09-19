# Reading Guide

These eight documents describe a research project testing whether a drone landing
policy benefits from being told how uncertain its position estimate is. They are
self-contained: no code, repository, or external material is required.

Suggested order:

1. **Project overview** — the question, why it matters, what makes it hard.
2. **How the simulation works** — the environments and the design decisions
   behind them, including the replay-not-simulate principle.
3. **The calibration result** — the measurement contribution, which stands
   independent of everything else.
4. **The results** — every experiment, including the one that was overturned by
   its own control.
5. **Seven retractions** — seven confident results that were withdrawn, and the
   cause of each.
6. **Plain English glossary** — every technical term, defined without jargon.
7. **All key numbers** — a reference sheet of every figure.
8. **What this means** — the defensible claims, and the claims explicitly not
   supported.

Two notes on interpretation.

Numbers are reported as medians and percentiles, not averages. The position error
distribution has a heavy tail, and a single catastrophic fix would dominate any
mean.

A "negative result" here is not an absence of evidence. The experiments were
powered to exclude effects above a stated size, and those bounds are reported
alongside every null.

Status at time of writing: one further experiment is still running — a sweep over
how much localisation accuracy would have to degrade before the error bar begins
to pay. Preliminary indications are consistent with the results reported here.
