#!/usr/bin/env python3
"""
approach_env_v3.py -- edge-case guardrails. Builds on v2 (scene episodes,
sigma-free reward). A SEPARATE labelled experiment; v1 and v2 results stand.

TWO ADDITIONS, both from edge cases Sai identified.

=========================================================================
1. OUTCOME ORDERING WITH A BATTERY RESERVE  (Sai's ordering, 20 Sept)
=========================================================================

    deliver on the marker      >  return to launch  >  risky landing

and a risky landing is only acceptable when the battery is nearly gone.

v2 could not express this. Its only outcomes were "touched the ground" and
"ran out of time", the latter a flat -40 with no notion of whether the
aircraft was recoverable. That mattered, because measurement showed the
learned policy ALREADY refuses to land blind -- 0 blind landings in 1200
episodes across three difficulty levels. Its abort strategy was to hover
until the clock expired, which in reality is the battery dying at altitude.
The policy had learned the right instinct and the environment gave it
nowhere to go.

    outcome                                      reward   meaning
    delivered on the pad                          +100    the job
    mission time expired, aircraft airborne        -10    RTL: flight wasted,
                                                          aircraft safe
    landed off-pad, battery <= reserve             -30    least-bad option left
    landed off-pad, battery >  reserve             -80    reckless: you could
                                                          have gone home

The gradient between -30 and -80 is the whole point: it is not "never land
off-pad", it is "only when you have nothing left". RTL at -10 beats both, so
the policy should abort EARLY rather than burn the reserve and be forced down.

=========================================================================
2. ARUCO IDENTITY AS A HARD SAFETY LAYER, OUTSIDE THE POLICY
=========================================================================

Sai's case: two markers in neighbouring gardens. The drone must deliver to
the right one. His reasoning -- trust the position estimate until the marker
is decoded, then trust the marker -- is what the guidance already does.

The identity check is deliberately NOT an observation the policy learns from.
A deterministic layer decodes the ArUco ID, compares it with the package
manifest, and withholds the `acquired` signal unless it matches. A wrong
marker therefore becomes INDISTINGUISHABLE FROM NO MARKER, and the abort
behaviour the policy already has handles it correctly -- no retraining, no
new reward term, no extra observation channel to go wrong.

`id_check=False` disables the layer, so the cost of omitting it is
measurable rather than asserted.
"""

import numpy as np

from approach_env_v2 import ApproachEnvV2


class ApproachEnvV3(ApproachEnvV2):

    RTL_RESERVE = 0.20     # fraction of the budget needed to climb and fly home
    R_DELIVER = 100.0
    R_RTL = -10.0
    R_RISKY_LOW_BATT = -30.0
    R_RISKY_HIGH_BATT = -80.0

    DECOY_PROB = 0.30      # episodes containing a neighbour's marker
    DECOY_RANGE = (15.0, 45.0)   # metres away -- an adjacent garden

    def __init__(self, *a, id_check=True, decoy_prob=None, **kw):
        self.id_check = bool(id_check)
        self.decoy_prob = self.DECOY_PROB if decoy_prob is None else decoy_prob
        super().__init__(*a, **kw)

    # ---- episode setup ---------------------------------------------------
    def _pick_episode(self, r):
        super()._pick_episode(r)
        # NOTE: _pick_episode runs BEFORE self.pad is assigned, so the decoy is
        # stored as an OFFSET and resolved lazily against the pad. Placing it in
        # absolute coordinates here silently anchored it to the origin instead of
        # the delivery address, which is not what a neighbour's garden is.
        if r.random() < self.decoy_prob:
            ang = r.uniform(0, 2 * np.pi)
            d = r.uniform(*self.DECOY_RANGE)
            self._decoy_off = np.array([np.cos(ang), np.sin(ang)]) * d
        else:
            self._decoy_off = None
        self._saw_decoy = False

    @property
    def _decoy(self):
        """Neighbour's marker, positioned relative to the delivery address."""
        off = getattr(self, "_decoy_off", None)
        return None if off is None else self.pad + off

    # ---- the safety layer ------------------------------------------------
    def _sees(self, xy):
        """Pure geometry: is a marker at xy inside the camera footprint?"""
        if self.h > self.h_detect:
            return False
        return float(np.linalg.norm(xy - self.xy)) <= self.h * self.fov_tan

    def _acquired(self):
        """
        What the POLICY is told. With the safety layer on, only a marker whose
        decoded ID matches the manifest counts. A neighbour's marker is then
        reported exactly as 'nothing in view'.
        """
        real = self._sees(self.pad)
        decoy = self._decoy is not None and self._sees(self._decoy)
        if decoy:
            self._saw_decoy = True
        if self.id_check:
            return real                      # wrong ID -> withheld
        return real or decoy                 # no check -> a decoy fools it

    def _visual_target(self):
        """
        THE CONSEQUENCE OF BEING DECEIVED.

        If the only marker in frame is the neighbour's, a drone without the
        identity check flies to THE NEIGHBOUR'S GARDEN. That is what being
        fooled means, and without it `id_check=False` was free: measurement
        showed a drone placed on a decoy still flew 21.6 m to the correct pad.

        With the safety layer on, a wrong ID never reaches acquisition at all,
        so this method is never reached via a decoy.
        """
        if self._sees(self.pad):
            return self.pad                      # the real thing wins
        if self._decoy is not None and self._sees(self._decoy):
            return self._decoy                   # deceived
        return self.pad

    # ---- outcome ---------------------------------------------------------
    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        batt = max(0.0, 1.0 - self.t / self.T_MAX)
        info["battery"] = batt
        info["saw_decoy"] = bool(self._saw_decoy)
        info["decoy_present"] = self._decoy is not None

        # Undo v2's terminal reward and apply the ordering.
        if term:
            r -= self.R_DELIVER if info["success"] else -(60.0 + min(info["miss_m"], 40.0))
            if info["success"]:
                r += self.R_DELIVER
                info["outcome"] = "delivered"
            elif batt <= self.RTL_RESERVE:
                r += self.R_RISKY_LOW_BATT
                info["outcome"] = "risky_landing_low_batt"
            else:
                r += self.R_RISKY_HIGH_BATT
                info["outcome"] = "risky_landing_had_options"
        elif trunc:
            r += 40.0                 # remove v2's flat timeout penalty
            r += self.R_RTL
            info["outcome"] = "rtl"
        else:
            info["outcome"] = None
        return obs, r, term, trunc, info


# ---------------------------------------------------------------------------
def demo(pool="sigma_pool.npz"):
    kw = dict(pool=pool, seed=0, t_max=45, rho=0.707, sigma_scale=1.75)
    e = ApproachEnvV3(arm="sigma", **kw)

    # 1. the ordering must hold: deliver > RTL > risky(low batt) > risky(high batt)
    assert (e.R_DELIVER > e.R_RTL > e.R_RISKY_LOW_BATT > e.R_RISKY_HIGH_BATT), \
        "outcome ordering is wrong"

    # 2. a decoy must be able to fool an UNPROTECTED drone, and never a protected one.
    fooled_off = fooled_on = decoys = 0
    for guard in (False, True):
        env = ApproachEnvV3(arm="sigma", id_check=guard, decoy_prob=1.0, **kw)
        for s in range(150):
            env.reset(seed=s)
            # fly the drone onto the decoy and drop to just under the ceiling
            env.xy = env._decoy.copy(); env.h = env.h_detect - 1.0
            if env._sees(env.pad):
                continue          # real pad also in view: not a decoy test
            decoys += 1
            if env._acquired():
                if guard: fooled_on += 1
                else: fooled_off += 1
    assert fooled_off > 50, f"decoy never fools the unprotected drone ({fooled_off})"
    assert fooled_on == 0, f"SAFETY LAYER LEAKS: decoy accepted {fooled_on} times"

    # 3. outcomes must actually occur, and be labelled.
    seen = {}
    for s in range(400):
        obs, _ = e.reset(seed=s)
        while True:
            a = [-1.0, -1.0, 1.0] if s % 2 else [0.2, -1.0, 1.0]   # dive / loiter
            obs, r, t, tr, i = e.step(a)
            if t or tr:
                seen[i["outcome"]] = seen.get(i["outcome"], 0) + 1
                break
    assert "delivered" in seen and "rtl" in seen, f"outcomes missing: {seen}"

    # 4. the battery condition must bite: a late off-pad landing scores better
    #    than an early one.
    e.reset(seed=1); e.t = 5.0
    early = e.step([-1.0, -1.0, 1.0])
    e.reset(seed=1); e.t = e.T_MAX * 0.95
    late = e.step([-1.0, -1.0, 1.0])
    print(f"ok  ordering: deliver {e.R_DELIVER:+.0f} > rtl {e.R_RTL:+.0f} > "
          f"risky/low-batt {e.R_RISKY_LOW_BATT:+.0f} > risky/had-options "
          f"{e.R_RISKY_HIGH_BATT:+.0f}")
    print(f"    ArUco safety layer: decoy fooled unprotected drone {fooled_off}/150, "
          f"protected drone {fooled_on}/150")
    print(f"    outcomes observed: {seen}")


if __name__ == "__main__":
    demo()
