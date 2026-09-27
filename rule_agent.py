#!/usr/bin/env python3
"""
rule_agent.py -- the analytical baselines, as agents inside the same env.

WHY THEY LIVE HERE AND NOT IN THE GAZEBO MISSION

Gazebo can run about twenty missions an hour. That is enough to demonstrate a
pipeline and nowhere near enough to claim a difference -- this project has
already had an 8-seed p=0.007 turn into p=0.52 at 20 seeds. Putting the rule
into the point-mass environment lets the same comparison run thousands of
times, and Gazebo is then asked only to confirm that the winner survives real
firmware.

THREE BASELINES, NOT ONE

  SigmaRule       search a box of radius k*sigma around the LATEST fix.
                  This is what flies in px4_mission.py today.

  SigmaRuleMean   identical, but centred on the running MEAN of all fixes.
                  This exists because it is the first thing a reviewer will
                  say: "if averaging helps, just average in the rule". If this
                  matches the policy, then the value was in the averaging and
                  not in the learning, and that is the honest finding.

  FixedRule       ignore sigma entirely, always search a fixed radius. The
                  control for whether sigma is doing anything at all.

Each sees exactly the observation vector the policy sees. Nothing reads env
internals, so no baseline is handicapped by being given less information.
"""

import math

import numpy as np


class _Base:
    """Common decoding of the shared observation vector."""

    SWEEP_TURNS = 12.0            # phase units to grow the spiral to full size

    def __init__(self, env, k=2.5):
        self.k = float(k)
        self.h_detect = float(env.h_detect)     # a design constant, not privilege
        self.sr_max = float(env.SEARCH_R_MAX)
        self.vz_max = float(env.VZ_MAX)
        self.spiral_rate = 0.9 * env.DT
        self.reset()

    def reset(self):
        self.phase = 0.0

    @staticmethod
    def decode(o):
        return dict(h=o[0] * 60.0, sigma=o[3] * 50.0, acq=o[4] > 0.5)

    def radius(self, s):
        raise NotImplementedError

    def offset(self, o):
        """Where to look, relative to the fix the guidance is centred on."""
        return 0.0, 0.0

    def act(self, o):
        d = self.decode(o)
        # EXPANDING spiral, not an orbit at fixed radius. Commanding a constant
        # r makes the aircraft circle at that range and never cross the middle,
        # so a larger box scored WORSE than a smaller one -- which would have
        # made sigma-scaling look harmful for a reason that is an artefact of
        # the search pattern, not a property of sigma. Ramping r from zero
        # outward covers the disk, and matches the expanding square flown in
        # Gazebo.
        rmax = 0.0 if d["acq"] else min(self.radius(d["sigma"]), self.sr_max)
        r = rmax * min(1.0, self.phase / self.SWEEP_TURNS)
        a1 = 2.0 * r / self.sr_max - 1.0

        if d["acq"]:
            vz = -1.0                       # marker in view: go down
        elif d["h"] > self.h_detect:
            vz = -1.0                       # too high to see anything: descend
        else:
            vz = 0.0                        # at the ceiling: hold and sweep

        self.phase += self.spiral_rate
        # a[2] is a bearing under steer=True and a damping gain otherwise; a
        # spiral bearing is the sane reading of both, so the same agent is
        # valid in either configuration.
        a2 = math.sin(self.phase)
        return np.array([vz, a1, a2], np.float32)


class SigmaRule(_Base):
    """radius = k * sigma. What px4_mission.py flies today."""

    def radius(self, s):
        return self.k * s


class SigmaRuleMean(SigmaRule):
    """
    Same box, but aimed at the running mean of the fixes rather than the newest.

    The mean is not free to use: the guidance centres the search on the latest
    fix, so this has to steer by the offset between them, which observation
    channels 6 and 7 carry in units of sigma.
    """

    def act(self, o):
        a = super().act(o)
        if len(o) >= 9:
            d = self.decode(o)
            off = np.array([o[6], o[7]], float) * d["sigma"]
            n = float(np.linalg.norm(off))
            if n > 1e-6 and not d["acq"]:
                # aim at the mean: bearing toward it, radius equal to its range
                a[2] = math.atan2(off[1], off[0]) / math.pi
                a[1] = 2.0 * min(n, self.sr_max) / self.sr_max - 1.0
        return a


class FixedRule(_Base):
    """Ignores sigma. The control for whether sigma matters at all."""

    def radius(self, s):
        return self.k


class PolicyAgent:
    """A trained policy, behind the same interface."""

    def __init__(self, path):
        from stable_baselines3 import PPO
        self.m = PPO.load(path, device="cpu")

    def reset(self):
        pass

    def act(self, o):
        a, _ = self.m.predict(o, deterministic=True)
        return a


def rollout(agent, env, seed):
    agent.reset()
    o, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        o, r, term, trunc, info = env.step(agent.act(o))
        total += r
        if term or trunc:
            return dict(outcome=info.get("outcome", "rtl"),
                        success=bool(info["success"]), miss_m=info["miss_m"],
                        t_s=info["t_s"], sigma_m=info["sigma_m"], reward=total)


# ---------------------------------------------------------------------------
# Agents for the obstacle environment (v5).
#
# Two oracles, deliberately, because they measure different things:
#
#   PadOracle      knows the pad exactly. Upper bound on what ANY improvement
#                  is worth, information included. If this is flat against the
#                  rule, stop -- nothing can help.
#   ObstacleOracle knows the TREES exactly but estimates the pad exactly as the
#                  rule does. This isolates the value of handling obstacles
#                  better, which is the decision we are actually asking a policy
#                  to learn. It is the tighter and more honest number.
# ---------------------------------------------------------------------------

class AvoidRule(SigmaRuleMean):
    """
    The strongest hand-written baseline: aim at the averaged fix, but steer off
    that bearing when a sector reading says something is close, and do not
    descend while a tree is within a body-length.

    This exists so the comparison is not a straw man. Beating a rule that
    ignores the trees would prove nothing.
    """

    # 6.0 m was pathological: trees sit within 22 m of the pad, so approaching
    # it almost always tripped the threshold, and the rule spent every episode
    # veering off and refusing to descend. It scored 32% against 98.8% for the
    # SAME rule with the avoidance switched off -- a baseline that bad proves
    # nothing about what a policy beats.
    CLEAR_M = 3.0

    def __init__(self, env, k=0.5):
        super().__init__(env, k)
        self.sense_r = float(env.SENSE_R)
        self.n_sec = int(env.N_SECTORS)

    def act(self, o):
        a = super().act(o)
        if len(o) < 9 + self.n_sec:
            return a
        rng = np.asarray(o[-self.n_sec:], float) * self.sense_r
        near = float(rng.min())
        if near < self.CLEAR_M:
            # steer toward the most open sector instead of straight at the pad
            best = int(np.argmax(rng))
            a[2] = ((best + 0.5) / self.n_sec * 2.0 - 1.0)
            a[0] = max(a[0], 0.0)          # stop descending next to a trunk
        return a


class PadOracle:
    """Told the pad. Upper bound on the value of any improvement at all."""

    def __init__(self, env):
        self.e = env

    def reset(self):
        pass

    def act(self, o):
        e = self.e
        # THE ACTION IS AN OFFSET FROM THE BELIEVED PAD, NOT A HEADING FROM THE
        # AIRCRAFT. Computing (pad - position) and handing it over as if it
        # were a heading made this "oracle" fly to a point unrelated to the
        # pad -- it scored 0% with perfect information, which is what exposed
        # it. Any agent here steers by saying where to look RELATIVE TO THE
        # LATEST FIX, so knowing the pad means commanding exactly the vector
        # that corrects the fix.
        off = e.pad - e._believed_pad()
        n = float(np.linalg.norm(off))
        a1 = 2.0 * min(n, e.SEARCH_R_MAX) / e.SEARCH_R_MAX - 1.0
        a2 = math.atan2(off[1], off[0]) / math.pi
        # Descend WHILE crossing, down to whatever the canopy allows, and go
        # all the way once overhead. The first version held altitude until it
        # was laterally over the pad and only then descended, which with no
        # trees means starting a 50 m descent with 13 s left -- it scored 0%,
        # worse than the plain rule. An oracle that loses to the thing it is
        # supposed to bound is a broken measuring stick, not a hard problem.
        d_pad = float(np.linalg.norm(e.pad - e.xy))
        tall = max([h for _, _, h in getattr(e, "obst", [])], default=0.0)
        want = (tall + 2.0) if d_pad > 3.0 else 0.0
        vz = float(np.clip((want - e.h) * 0.5, -1.0, 1.0))
        return np.array([vz, a1, a2], np.float32)


class ObstacleOracle(SigmaRuleMean):
    """
    Estimates the pad like the rule does, but knows every tree exactly.
    Isolates how much better obstacle handling alone is worth.
    """

    def __init__(self, env, k=0.5):
        super().__init__(env, k)
        self.e = env

    def act(self, o):
        a = super().act(o)
        e = self.e
        tall = max([h for _, _, h in getattr(e, "obst", [])], default=0.0)
        n = float(np.linalg.norm(e._fix_mean() - e.xy))
        want = (tall + 2.0) if n > 3.0 else 0.0
        a[0] = float(np.clip((want - e.h) * 0.5, -1.0, 1.0))
        return a


class CanopyRule(SigmaRuleMean):
    """
    Land NEAR the pad rather than on top of it when the column is fouled.

    The failure this fixes, measured in crash_geometry.py: the pad sits in a
    clear column about 4.7 m across, the aircraft's belief about where the pad
    is sits 12 m out, and so a quarter of descents are committed over a point
    that has canopy above it. Those crash 72% of the time against 35% for the
    rest.

    Descending is therefore not a thing to do whenever the marker is visible.
    It is a thing to do only where the sky is open, and the sector ranges say
    where that is: a reading far below SENSE_R means something is beside us at
    OUR OWN altitude, which is the definition of being in the canopy layer.
    When that happens, hold height and slide the aim point toward the most
    open bearing instead of pressing down through a tree.

    Two knobs, both physical: how much clearance counts as open, and how far
    to step aside. Neither is tuned against the eval set.
    """

    CLEAR_M = 4.0        # a sector closer than this is an obstruction
    STEP_M = 6.0         # how far off the pad to slide when one is found

    def act(self, o):
        a = super().act(o)
        if len(o) < 17:
            return a                     # no range sensor: nothing to add
        rng = np.asarray(o[9:17], float) * 30.0     # SENSE_R
        near = float(rng.min())
        if near >= self.CLEAR_M:
            return a                     # open sky: the ordinary rule stands

        # Something is at our altitude. Do not descend into it, and aim at the
        # clearest bearing instead of at the believed pad.
        s = int(np.argmax(rng))
        th = (s + 0.5) / len(rng) * 2 * math.pi
        # CLIMB, do not hold. Holding at canopy height and sliding sideways
        # translates THROUGH canopies -- the crash test fires on horizontal
        # proximity inside the canopy band, so the evasive move was itself the
        # hazard. Going over the top first is the only safe way across.
        a[0] = +1.0
        a[2] = ((th + math.pi) % (2 * math.pi) - math.pi) / math.pi
        a[1] = 2.0 * min(self.STEP_M, self.sr_max) / self.sr_max - 1.0
        return a
