#!/usr/bin/env python3
"""
cage_env.py -- STAGE 2. The same sigma ablation, in the MuJoCo Tello cage.

WHAT THIS ADDS TO THE EXISTING AVIARY

`TelloLandingAviary` hands the policy `rel_xy`, the TRUE vector from drone to
pad. There is no localisation error in it at all, so it cannot express the
question this project asks. This subclass replays GeoAnchor's real error into
that channel and adds the same three ablation arms as the outdoor study:

    none       believed pad position only
    staleness  + exp(-lambda * t_since_fix)
    sigma      + the calibrated sigma, in cage metres

THE SIGN, WHICH IS EASY TO GET BACKWARDS

GeoAnchor estimates the DRONE's position, not the pad's. The pad is a known
coordinate. So if the drone believes it sits at (true + err), it computes the
pad as being (rel_xy - err) away. Getting this backwards made the outdoor
guidance target run away and produced 1-5% success across the board.

SIGMA SCALING, AND WHY IT IS A JUDGEMENT CALL

Outdoor sigma is 8.75-80 m in an operating box of half-extent 80 m. The cage is
3.0 x 2.0 x 3.0 m. Scaling by the ratio of operating half-extents

    SIGMA_SCALE = 1.5 / 80 = 0.019

keeps "how much of the searchable area is the error" constant, which is the
quantity that actually drives the search-vs-commit decision. It puts median
sigma at ~0.26 m against a 0.25 m landing radius -- tight but achievable, with
the worst fixes (1.5 m) placing the believed pad at the cage wall.

Two rejected alternatives, recorded so this is not re-litigated:
  - scale by success radius (0.75 m -> 0.25 m): gives sigma ~4.6 m in a 3 m
    cage. Absurd; the believed pad is almost always outside the cage.
  - scale by altitude (50 m -> 2 m): gives sigma ~0.55 m. Defensible, harder.
`sigma_scale` is a constructor argument so this can be swept rather than argued.

NOTE ON DUPLICATED CODE

The AR(1) copula fix model below is duplicated from `approach_env._new_fix`
rather than shared. That is deliberate and temporary: a 130-run overnight
experiment is currently importing `approach_env`, and editing it mid-flight
would change the code under jobs that have not started yet. Unify into a shared
`replay.py` once that finishes.
"""

import math
import os
import sys

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.expanduser("~/RL_projects/MuJoCo-drones-gym"))
from tello_landing_aviary import TelloLandingAviary   # noqa: E402


class CageLandingEnv(TelloLandingAviary):

    SIGMA_SCALE = 1.5 / 80.0      # cage half-extent / outdoor half-extent
    FIX_DT = 0.5                  # GeoAnchor fix interval (s)

    # Thrust authority as a fraction of hover RPM. a=0 hovers, a=+1 is
    # 1.3x hover (climb), a=-1 is 0.7x hover (descend).
    ACT_BAND = 0.30

    def __init__(self, pool="sigma_pool.npz", arm="sigma", rho=None,
                 sigma_scale=None, lam=0.5, pad_moves=False, **kw):
        """
        pad_moves defaults to FALSE here, unlike the parent aviary.

        A drifting pad is a second, unrelated source of uncertainty. This study
        is about LOCALISATION uncertainty, so letting the pad wander would
        confound the very thing being measured -- and it would do so unequally,
        since a policy that tracks a moving pad well looks good for reasons
        that have nothing to do with sigma. Turn it back on only to study pad
        tracking, which is a different question.
        """
        assert arm in ("none", "staleness", "sigma"), arm
        self.arm = arm
        self.lam = lam
        self.sigma_scale = self.SIGMA_SCALE if sigma_scale is None else sigma_scale

        d = np.load(pool, allow_pickle=True)
        self.pool_sigma = np.asarray(d["sigma"], float)
        self.pool_err = np.asarray(d["err"], float)
        order = np.argsort(self.pool_sigma)
        self.sigma_bins = np.array_split(order, 10)
        self.bin_err_sorted = [np.sort(self.pool_err[b]) for b in self.sigma_bins]
        self.rho = float(rho) if rho is not None else (
            float(d["rho"]) if "rho" in d.files else 0.0)

        super().__init__(pad_moves=pad_moves, **kw)

        # The parent DECLARES shape (18,) but _computeObs returns 26 -- the
        # base aviary contributes 20 elements, not the 12 its comment assumes.
        # Derive the space from a real observation so it cannot drift again.
        self._fix_rng = np.random.default_rng(0)
        self._reset_fix_state()
        probe = self._computeObs()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, probe.shape, np.float32)

        # THE ACTION SPACE MUST BE NORMALISED.
        #
        # The parent exposes raw motor RPM: Box(0, 21713.7). A PPO Gaussian
        # policy emits roughly [-3, +3], which clips to [0, 3] RPM against a
        # HOVER_RPM of 14476 -- 0.02% of the thrust needed to stay airborne.
        # The drone therefore falls out of the sky on EVERY step regardless of
        # what the policy does, and every run scores an identical ~17% (the
        # rate at which a falling drone lands within 0.25 m by luck).
        # 84 training runs were wasted on this before it was caught.
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

    # ---- replayed localisation error ------------------------------------
    def _reset_fix_state(self):
        r = self._fix_rng
        b = int(r.integers(len(self.sigma_bins)))
        self._bin_err = self.bin_err_sorted[b]
        self.sigma = float(np.median(self.pool_sigma[self.sigma_bins[b]])) \
            * self.sigma_scale
        self._bearing = float(r.uniform(0, 2 * np.pi))
        self._z = float(r.normal())
        self.t_fix = 0.0
        self._new_fix()

    def _new_fix(self):
        """AR(1) Gaussian copula, so P(error | sigma) stays exactly empirical."""
        r = self._fix_rng
        self._z = (self.rho * self._z +
                   np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * r.normal())
        u = 0.5 * (1.0 + math.erf(self._z / np.sqrt(2.0)))
        e = self._bin_err
        mag = float(e[int(round(u * (len(e) - 1)))]) * self.sigma_scale
        self._bearing += r.normal(0.0, 0.02)
        self.err_xy = mag * np.array([np.cos(self._bearing),
                                      np.sin(self._bearing)])
        self.t_fix = 0.0

    # ---- gym -------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._fix_rng = np.random.default_rng(seed)
        out = super().reset(seed=seed, options=options)
        self._reset_fix_state()
        obs = self._computeObs()
        return obs, (out[1] if isinstance(out, tuple) and len(out) > 1 else {})

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        rpm = self.HOVER_RPM * (1.0 + self.ACT_BAND * a)
        obs, rew, term, trunc, info = super().step(rpm)
        self.t_fix += 1.0 / self.CTRL_FREQ
        if self.t_fix >= self.FIX_DT:
            self._new_fix()
        info["sigma_m"] = float(self.sigma)
        info["err_m"] = float(np.linalg.norm(self.err_xy))
        return self._computeObs(), rew, term, trunc, info

    def _computeObs(self):
        obs = np.asarray(super()._computeObs(), np.float32).flatten()
        # Base layout: [0:20] drone state, [20:22] rel_xy, [22] altitude,
        # [23:25] pad velocity, [25] horizontal distance.
        believed = obs[20:22] - self.err_xy.astype(np.float32)
        obs[20:22] = believed
        obs[25] = np.linalg.norm(believed)

        if self.arm == "sigma":
            unc = self.sigma / (self.SIGMA_SCALE * 50.0)
        elif self.arm == "staleness":
            unc = float(np.exp(-self.lam * self.t_fix))
        else:
            unc = 0.0
        return np.concatenate([obs, [unc]]).astype(np.float32)


# ---------------------------------------------------------------------------
def demo():
    """One runnable check: sign, arm wiring, and that the obs space is honest."""
    pool = "sigma_pool.npz"
    e = CageLandingEnv(pool=pool, arm="sigma", render_mode=None)
    obs, _ = e.reset(seed=0)

    assert e.observation_space.contains(obs), \
        "observation space does not match the observation actually returned"
    assert obs.shape == (27,), obs.shape

    # Sigma must land in a range that is sane for a 3 x 2 m cage.
    sigmas = []
    for s in range(60):
        e.reset(seed=s)
        sigmas.append(e.sigma)
    lo, hi = min(sigmas), max(sigmas)
    assert 0.05 < lo and hi < 2.0, f"scaled sigma out of cage range: {lo}..{hi}"

    # THE SIGN. The believed pad must sit at (true - err), not (true + err).
    e.reset(seed=3)
    true_rel = e._pad_xy - e.pos[0][:2]
    bel_rel = e._computeObs()[20:22]
    assert np.allclose(bel_rel, true_rel - e.err_xy, atol=1e-4), \
        "believed pad has the wrong sign -- see the docstring"

    # Arms differ in exactly one channel and nothing else.
    envs = {a: CageLandingEnv(pool=pool, arm=a, render_mode=None)
            for a in ("none", "staleness", "sigma")}
    o = {a: v.reset(seed=1)[0] for a, v in envs.items()}
    assert o["none"][-1] == 0.0 and o["sigma"][-1] > 0 and o["staleness"][-1] > 0
    for a in ("sigma", "staleness"):
        assert np.allclose(o[a][:-1], o["none"][:-1], atol=1e-5), \
            f"{a} arm differs outside the uncertainty channel"

    # THE ACTION INTERFACE MUST LET THE POLICY FLY.
    #
    # This is the check whose absence cost 84 training runs. It asserts that
    # the actions a policy can actually emit span descend / hover / climb. An
    # un-normalised action space passes every other test in this file and then
    # silently produces a constant score, because the policy is never in the
    # loop at all.
    def _z_after(act, n=150):
        e2 = CageLandingEnv(pool=pool, arm="none", render_mode=None)
        e2.reset(seed=0)
        z0 = float(e2.pos[0][2])
        for _ in range(n):
            e2.step(np.full(4, act, dtype=np.float64))
        return z0, float(e2.pos[0][2])

    assert tuple(e.action_space.low) == (-1.0,) * 4, \
        f"action space is not normalised: {e.action_space}"
    z0, z_hover = _z_after(0.0)
    z0, z_up = _z_after(1.0)
    z0, z_down = _z_after(-1.0)
    assert abs(z_hover - z0) < 0.30, \
        f"action 0 should roughly hover, went {z0:.2f} -> {z_hover:.2f} m"
    assert z_up > z0 + 0.20, f"action +1 should climb, went {z0:.2f} -> {z_up:.2f}"
    assert z_down < z0 - 0.20, f"action -1 should descend, went {z0:.2f} -> {z_down:.2f}"

    print(f"    flight authority: a=-1 -> {z_down:.2f} m, a=0 -> {z_hover:.2f} m, "
          f"a=+1 -> {z_up:.2f} m  (from {z0:.2f} m)")
    print(f"ok  obs={obs.shape[0]}-dim  cage sigma {lo:.2f}-{hi:.2f} m "
          f"vs {e.LAND_RADIUS:.2f} m landing radius  rho={e.rho:.3f}")
    print(f"    median cage sigma {np.median(sigmas):.2f} m, "
          f"scale {e.sigma_scale:.4f} (= 1.5/80)")


if __name__ == "__main__":
    demo()
