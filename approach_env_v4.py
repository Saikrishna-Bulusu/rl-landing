#!/usr/bin/env python3
"""
approach_env_v4.py -- the four changes that give a policy something to win at.

v3 lost a head-to-head against a one-line rule (search a box of radius
2.5*sigma): 3/4 deliveries against 3/3, and 60% slower. Rather than accept
that, this asks whether v3 was ever being asked the right question. Four
things were wrong, and each is a separate flag so the ablation can say which
of them actually mattered.

1. TIME WAS FREE, SO IT DAWDLED.

   v3 paid +0.5 every step the marker was in view against a -0.02 step cost.
   At DT=0.2 that is +2.4/s net, so forty seconds of loitering in view earned
   +96 -- within four points of what DELIVERING the package pays. The policy
   being slower than the hand rule was not a weakness of RL. It was doing
   exactly what it was paid to do. Now the in-view bonus is small and every
   step costs real energy.

2. IT SAW ONE FIX AT A TIME, WHICH THROWS AWAY THE STRUCTURE.

   GeoAnchor's error is AR(1) with rho=0.707, not fresh noise: it is a drifting
   bias whose mean is zero. So averaging fixes shrinks it -- the variance of a
   mean of N is sigma^2 (1+rho)/(N(1-rho)), about 0.58 sigma^2 at N=10, a 24%
   reduction in error for free. A rule keyed on the LATEST fix cannot use that.
   A policy shown the running mean can. This is the cleanest argument for RL
   here: not that it controls better, but that it can exploit structure the
   analytical rule discards.

3. IT COULD SIZE THE SEARCH BUT NOT AIM IT.

   a[2] damped a fixed-rate spiral. The pattern itself was hardcoded, so the
   policy could say how far to look and never where. Now it sets the bearing
   directly and can bias toward where the fixes have been drifting.

4. ITS EPISODES WERE NOT THE MISSION.

   v3 spawned roughly a sigma from the pad and had only to descend. The real
   aircraft cruises 120 m at altitude before it may look at all, and that
   transit is the single largest piece of the battery. Including it is what
   makes the reserve bind -- and a budget that binds is the only regime where
   trading search extent against getting home is a decision at all.
"""

import math

import numpy as np
from gymnasium import spaces

from approach_env_v3 import ApproachEnvV3


class ApproachEnvV4(ApproachEnvV3):

    # (1) time costs energy; loitering in view no longer pays like delivering
    IN_VIEW = 0.10               # was 0.50
    ENERGY_PER_STEP = 0.10       # was 0.02, and it is now the dominant drain

    # (2b) THE ERROR DIRECTION WANDERS. MEASURED, NOT ASSUMED.
    #
    # v1-v3 drew one bearing per episode and nudged it by 0.02 rad per fix, so
    # the error was a fixed bias pointing one way for the whole flight. Under
    # that model averaging fixes is USELESS -- it converges on the mean
    # magnitude of a skewed distribution, which is worse than the latest
    # sample. The self-check below caught exactly that.
    #
    # GeoAnchor's own error vectors say otherwise. Across 2993 clean frames the
    # MAGNITUDE persists (lag-1 +0.09 to +0.55, which is what rho models) but
    # the DIRECTION does not: the mean cosine between consecutive error
    # directions is +0.25, and |mean error vector| / sd is 0.08 to 0.33, so
    # errors substantially cancel when averaged. For a Gaussian bearing step,
    # E[cos] = exp(-s^2/2), so s = sqrt(-2 ln 0.25).
    DIR_PERSIST = 0.25
    BEARING_STEP = math.sqrt(-2.0 * math.log(0.25))

    # (4) the mission: cruise in from a depot rather than starting on top
    START_RANGE = (100.0, 140.0)
    AREA = 220.0                 # the box has to hold a 120 m transit leg
    PAD_SPAN = 30.0              # ...with the pad near the middle of it
    VXY_MAX = 8.0                # matches the x500 cruise flown in Gazebo
    # a[1] maps to 0..this. 25 m could not cover a scaled sigma, whose
    # median is 24 m and tail reaches 140 m. Widened in the class constant
    # the base actually reads -- an earlier attempt rescaled the ACTION
    # instead, which the base's clip to [-1,1] silently undid.
    SEARCH_R_MAX = 60.0

    def __init__(self, *a, history=True, steer=True, transit=True,
                 energy=True, **kw):
        self.history, self.steer = bool(history), bool(steer)
        self.transit, self.energy = bool(transit), bool(energy)
        super().__init__(*a, **kw)
        if self.history:
            n = self.observation_space.shape[0] + 3
            self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)

    # ---- (2) fix history -------------------------------------------------
    def _new_fix(self):
        super()._new_fix()
        b = self._believed_pad()
        self._fix_sum = getattr(self, "_fix_sum", np.zeros(2)) + b
        self._fix_n = getattr(self, "_fix_n", 0) + 1

    def _fix_mean(self):
        n = getattr(self, "_fix_n", 0)
        return self._fix_sum / n if n else self._believed_pad()

    def _obs(self):
        o = super()._obs()
        if not self.history:
            return o
        # Where the AVERAGE of every fix so far sits relative to the newest one,
        # in units of sigma, plus how many fixes that average rests on. That is
        # the whole of the extra information, and it is exactly what a rule
        # keyed on the latest fix cannot see.
        d = (self._fix_mean() - self._believed_pad()) / max(self.sigma, 1e-6)
        n = getattr(self, "_fix_n", 0)
        return np.concatenate([o, np.array(
            [np.clip(d[0], -3, 3), np.clip(d[1], -3, 3),
             min(n, 40) / 40.0], np.float32)]).astype(np.float32)

    # ---- (3) the policy aims the search ----------------------------------
    def _search_target(self, a, search_r, chase):
        if not self.steer:
            return super()._search_target(a, search_r, chase)
        if self.steer == 2:
            # CARTESIAN. The polar form below makes the best known strategy --
            # "aim at the running mean" -- awkward to express: the policy must
            # compose a radius and a bearing that together reproduce a vector
            # it can already see in observation channels 6 and 7. In Cartesian
            # form that strategy is a linear function of those two channels,
            # which is about the easiest thing a network can represent. If the
            # policy does better here, the earlier shortfall was the action
            # space, not the method.
            off = np.clip(a[1:3], -1, 1) * self.SEARCH_R_MAX
            self.target = self._believed_pad() + off
            return self.target
        # a[2] is a bearing now, not a damping gain. Offset from the newest fix
        # so that steering toward the running mean is something the policy has
        # to learn from the observation rather than something baked in here.
        th = math.pi * float(np.clip(a[2], -1, 1))
        self.target = self._believed_pad() + search_r * np.array(
            [math.cos(th), math.sin(th)])
        return self.target

    # ---- (1) reward ------------------------------------------------------
    def _reward(self, acq, miss):
        if not self.energy:
            return super()._reward(acq, miss)
        r = -self.ENERGY_PER_STEP
        if acq:
            r += self.IN_VIEW
            r += 0.3 * (-self.vz / self.VZ_MAX)
            r -= 0.02 * miss
        else:
            reach = self.h * self.fov_tan
            trust = np.clip(reach / max(3.0 * self.sigma, 1e-6), 0.0, 1.0)
            gate = 1.0 if self.h >= self.h_detect else (2.0 * trust - 1.0)
            r += 0.25 * gate * (-self.vz / self.VZ_MAX)
        if acq and not self.was_acquired:
            r += 10.0
            self.was_acquired = True
        return r

    # ---- (4) start the episode at the depot -------------------------------
    def reset(self, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options)
        self._fix_sum, self._fix_n = np.zeros(2), 0
        self._new_fix()
        if self.transit:
            rng = self.rng
            # Keep the pad near the middle so a full transit leg fits inside
            # the box; the base class scatters it across half the area, which
            # silently clipped the start to 85 m instead of 120.
            self.pad = rng.uniform(-self.PAD_SPAN, self.PAD_SPAN, 2)
            th = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(*self.START_RANGE)
            self.xy = np.clip(self.pad + d * np.array([np.cos(th), np.sin(th)]),
                              -self.AREA, self.AREA)
            self._fix_sum, self._fix_n = np.zeros(2), 0
            self._new_fix()
            self.target = self._believed_pad()
        return self._obs(), i


def demo(pool="sigma_pool.npz"):
    """Checks the four changes are real, not just present."""
    import numpy as np
    kw = dict(pool=pool, seed=0, t_max=60, rho=0.707, sigma_scale=1.75)

    e = ApproachEnvV4(arm="sigma", **kw)
    o, _ = e.reset(seed=0)
    assert o.shape == (9,), f"history should add 3 channels, got {o.shape}"

    # (1) loitering in view must no longer pay like delivering
    old = ApproachEnvV3(arm="sigma", **kw)
    loiter_v3 = (0.5 - 0.02) / old.DT * 40          # 40 s in view
    loiter_v4 = (e.IN_VIEW - e.ENERGY_PER_STEP) / e.DT * 40
    print(f"1. 40 s loitering in view: v3 earns {loiter_v3:+.0f}, "
          f"v4 earns {loiter_v4:+.0f}, delivery pays {e.R_DELIVER:+.0f}")
    assert loiter_v3 > 0.8 * e.R_DELIVER, "v3 loiter should rival delivery"
    assert loiter_v4 < 0.2 * e.R_DELIVER, "v4 loiter must not rival delivery"

    # (2) the running mean must actually be a better pad estimate than the
    #     latest fix -- if it is not, the extra channels are noise
    errs_latest, errs_mean = [], []
    for s in range(60):
        e.reset(seed=s)
        for _ in range(60):
            e.step(np.array([0.0, -1.0, 0.0]))
        errs_latest.append(np.linalg.norm(e._believed_pad() - e.pad))
        errs_mean.append(np.linalg.norm(e._fix_mean() - e.pad))
    a, b = float(np.mean(errs_latest)), float(np.mean(errs_mean))
    print(f"2. pad estimate error: latest fix {a:.1f} m, running mean {b:.1f} m "
          f"({100*(a-b)/a:+.0f}%)")
    assert b < a, "averaging fixes must beat the latest fix, or history is useless"

    # (3) bearing must change where it looks
    e.reset(seed=1)
    t = []
    for bearing in (-1.0, 0.0, 1.0):
        e.reset(seed=1)
        e.step(np.array([0.0, 1.0, bearing]))
        t.append(e.target.copy())
    spread = max(np.linalg.norm(t[i] - t[j]) for i in range(3) for j in range(3))
    print(f"3. bearing moves the look-point by up to {spread:.0f} m")
    assert spread > 10.0, "steering is inert"

    # (4) the transit must actually cost something
    near = ApproachEnvV4(arm="sigma", transit=False, **kw); near.reset(seed=2)
    far = ApproachEnvV4(arm="sigma", transit=True, **kw); far.reset(seed=2)
    dn = np.linalg.norm(near.xy - near.pad); df = np.linalg.norm(far.xy - far.pad)
    cost = df / far.VXY_MAX
    print(f"4. start range: without transit {dn:.0f} m, with transit {df:.0f} m "
          f"-> {cost:.0f} s of a {far.T_MAX:.0f} s budget "
          f"({100*cost/far.T_MAX:.0f}%)")
    assert df >= far.START_RANGE[0] - 1, "transit start was clipped"
    assert cost > 0.15 * far.T_MAX, "transit must cost a real share of the budget"
    print("\nall four changes verified")


if __name__ == "__main__":
    demo()
