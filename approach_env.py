#!/usr/bin/env python3
"""
approach_env.py -- the approach-to-touchdown decision problem, with GeoAnchor's
real localization error replayed into it.

THE PROBLEM THIS ENV IS

A delivery drone is at ~50 m over a suburb. GeoAnchor tells it where it is, with
an honest sigma of 10-40 m. The ArUco pad in the customer's backyard is a
handful of pixels at that altitude -- invisible. So there is a gap between
"I know where I am to +/- 20 m" and "I can see the pad", and crossing it is a
decision:

  - descend now, or hold and take another fix?
  - fly straight to the believed point, or sweep a pattern around it?
  - and if the ground arrives and the pad was never found, that is a delivery
    into someone's pool.

A policy given only a position estimate cannot tell sigma=3 m from sigma=40 m --
it behaves identically in both. That is the hypothesis this env exists to test.

DESIGN, AND WHY

  Point-mass dynamics, not MuJoCo. WaveLander (arXiv 2607.01281) trains its
  landing policy on "a single prismatic joint constrained to vertical
  translation" with a randomized-gain velocity tracker -- no quadrotor dynamics
  at all. At 50 m under a velocity controller, attitude contributes nothing to a
  search-vs-commit decision, and this trains in minutes on a laptop instead of
  hours. The MuJoCo cage aviary stays where it earns its keep: the Tello check.

  Hierarchical, per WaveLander. The policy emits 2 numbers. A P controller flies
  the aircraft. The learned surface is as small as it can be while still able to
  express the decision.

  Observation is 6-dim. Resist growth.

  Uncertainty is REPLAYED, not simulated. Sampling sigma from an invented
  distribution would break the thing being measured: calibration does not
  transfer for free, and randomizing sigma's marginal does not preserve its
  relationship to the actual error. Real (sigma, error) pairs from
  results/s8_train keep the joint distribution correct by construction.

ABLATION ARMS -- the whole point, mirroring PerchRL's rho_t ablation
    "none"       policy sees the position estimate only
    "staleness"  policy also sees exp(-lambda * t_since_fix)   <- PerchRL's proxy
    "sigma"      policy also sees the calibrated sigma, in metres
Same replayed errors in all three, so the only difference is what the policy was
told about them.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ApproachEnv(gym.Env):
    metadata = {"render_modes": []}

    # ---- world ----------------------------------------------------------
    AREA = 80.0          # half-extent of the operating box (m)
    H0 = (45.0, 65.0)    # spawn altitude range (m AGL)
    H_GROUND = 1.0       # touchdown altitude (m)
    PAD_R = 0.75         # success radius at touchdown (m)

    # ---- vehicle (sub-1 kg delivery quad) -------------------------------
    VZ_MAX = 2.0         # commanded descent/climb, m/s
    VXY_MAX = 5.0        # lateral speed cap, m/s
    TAU = (0.30, 0.80)   # first-order velocity lag, randomized per episode

    # ---- camera ---------------------------------------------------------
    # These lines set the whole difficulty of the task. Do not tune them to make
    # training easier -- they are the physics of seeing a 1 m marker.
    FOV_DEG = 60.0
    PX = 640
    MIN_PX = 20.0        # marker needs ~20 px across to decode reliably
    PAD_SIZE = 1.0       # marker edge length (m)

    # ---- episode --------------------------------------------------------
    DT = 0.2             # control period (s) -> 5 Hz decisions
    T_MAX = 120.0        # seconds
    FIX_DT = 0.5         # GeoAnchor fix interval (s) -> 2 Hz

    def __init__(self, pool="sigma_pool.npz", arm="sigma", seed=None,
                 sigma_scale=1.0, lam=0.5, t_max=None, vxy_max=None):
        """
        t_max / vxy_max override the class defaults. They are the two honest
        difficulty knobs: loiter time is really a battery reserve, and lateral
        speed is really an airframe limit. If a sigma-blind sweep saturates on
        the real pool, tighten THESE -- do not weaken the baseline, and do not
        touch the camera constants, which are physics.
        """
        super().__init__()
        assert arm in ("none", "staleness", "sigma"), arm
        self.arm = arm
        self.lam = lam
        self.sigma_scale = sigma_scale
        if t_max is not None:
            self.T_MAX = float(t_max)
        if vxy_max is not None:
            self.VXY_MAX = float(vxy_max)
        self.rng = np.random.default_rng(seed)

        d = np.load(pool, allow_pickle=True)
        self.pool_sigma = np.asarray(d["sigma"], float)
        self.pool_err = np.asarray(d["err"], float)
        self.pool_scene = np.asarray(d["scene"]).astype(str)

        # Group pool rows into sigma deciles. Within an episode sigma is FIXED
        # -- it is a property of the scene, altitude band and matcher, not
        # something that re-rolls twice a second -- and each fix draws its error
        # magnitude from the real errors of rows with that sigma. So the
        # conditional distribution P(error | sigma) stays exactly the empirical
        # one, which is the property the whole replay design exists to preserve.
        order = np.argsort(self.pool_sigma)
        self.sigma_bins = np.array_split(order, 10)

        fov = np.radians(self.FOV_DEG)
        self.fov_tan = np.tan(fov / 2.0)
        # Above this altitude the marker is smaller than MIN_PX and cannot be
        # decoded no matter where the drone is. ~30 m for a 1 m pad at 640 px.
        self.h_detect = self.PAD_SIZE * self.PX / (fov * self.MIN_PX)

        # a[0] vertical velocity, a[1] how hard to chase the newest fix
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

    # ---- localization ---------------------------------------------------
    def _new_fix(self):
        """
        Draw the next replayed fix. Sigma is fixed for the episode; the error
        magnitude is a fresh draw from the real errors that accompany that sigma.

        The bearing drifts only slightly. Map-match error direction is set by
        scene structure, not white noise, and a fast-rotating bearing would let
        a drone that simply descends sweep a free search pattern -- which would
        silently destroy the experiment by making the sigma-blind policy good.
        """
        j = int(self.rng.choice(self._bin))
        self._bearing += self.rng.normal(0.0, 0.02)
        mag = float(self.pool_err[j]) * self.sigma_scale
        self.err_xy = mag * np.array([np.cos(self._bearing),
                                      np.sin(self._bearing)])
        self.t_fix = 0.0

    def _believed_xy(self):
        """What the aircraft thinks its position is. Wrong by the replayed error."""
        return self.xy + self.err_xy

    def _believed_pad(self):
        """
        Where the aircraft thinks the pad is, expressed in TRUE coordinates.

        The pad is a delivery coordinate and is known exactly. What is uncertain
        is where the aircraft itself is. Believing it sits at xy + err, it
        computes the pad as (pad - belief) away, so steering there actually
        takes it to pad - err. That offset is the whole problem.
        """
        return self.pad - self.err_xy

    def _acquired(self):
        """
        Is the marker visible right now?

        Two conditions, both physical: low enough that it spans MIN_PX, and
        inside the camera footprint. Neither alone is sufficient, which is what
        makes altitude a real decision rather than 'always descend'.
        """
        if self.h > self.h_detect:
            return False
        return np.linalg.norm(self.pad - self.xy) <= self.h * self.fov_tan

    # ---- gym ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng

        self.pad = r.uniform(-self.AREA * 0.5, self.AREA * 0.5, 2)
        self.h = float(r.uniform(*self.H0))
        # Start displaced from the pad by roughly a typical sigma, so the task
        # begins in the regime where uncertainty actually matters.
        self.xy = self.pad + r.normal(0, 15.0, 2)
        self.vz = 0.0
        self.vxy = np.zeros(2)
        self.tau = float(r.uniform(*self.TAU))
        self.t = 0.0
        self.was_acquired = False

        # Pick this episode's conditions: one sigma decile, held for the episode.
        self._bin = self.sigma_bins[int(r.integers(len(self.sigma_bins)))]
        self.sigma = float(np.median(self.pool_sigma[self._bin])) * self.sigma_scale
        self._bearing = r.uniform(0, 2 * np.pi)
        self._new_fix()
        self.target = self._believed_pad()     # the low-passed guidance target

        return self._obs(), {}

    def _obs(self):
        # What the aircraft believes its horizontal distance to the pad is.
        d_bel = float(np.linalg.norm(self._believed_pad() - self.xy))

        # The three arms differ ONLY in this channel.
        if self.arm == "sigma":
            unc = self.sigma / 50.0
        elif self.arm == "staleness":
            unc = float(np.exp(-self.lam * self.t_fix))
        else:
            unc = 0.0

        return np.array([
            self.h / 60.0,
            self.vz / self.VZ_MAX,
            min(d_bel, 100.0) / 100.0,
            unc,
            float(self._acquired()),
            min(self.t, self.T_MAX) / self.T_MAX,
        ], np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        vz_cmd = a[0] * self.VZ_MAX
        chase = (a[1] + 1.0) * 0.5              # 0..1

        # --- lateral guidance: a controller, not the policy -----------------
        #
        # a[1] is how hard to chase the newest fix. This replaced an expanding
        # search radius, which measurement showed was inert: sweeping r=0 scored
        # exactly as well as the best tuned sweep, because the camera footprint
        # (17.3 m at the ceiling) already dwarfs the disk worth sweeping.
        #
        # The real lever is that the believed pad JUMPS. Each fix redraws the
        # error magnitude, so at high sigma the guidance target moves ~51 m
        # (p90) between 0.2 s control steps while the aircraft can fly 1.0 m.
        # Chasing that is chasing noise, and it burns the whole speed budget
        # thrashing instead of sitting still inside the footprint. Damping the
        # chase is the decision, and its optimum is sigma-dependent by
        # construction: snap to the fix when sigma is small, hold station when
        # it is large.
        if self._acquired():
            target = self.pad          # marker in view: fly to it, trust nothing else
            self.target = target.copy()
        else:
            self.target = (1.0 - chase) * self.target + chase * self._believed_pad()
            target = self.target

        # ponytail: plain P controller with a velocity cap. A full PID adds an
        # integral term that only matters for steady-state offset against wind,
        # which this env does not model. Upgrade when wind is added.
        v_des = (target - self.xy) / max(self.tau, 1e-3)
        n = np.linalg.norm(v_des)
        if n > self.VXY_MAX:
            v_des *= self.VXY_MAX / n

        # First-order velocity lag, same shape as WaveLander's low-level tracker
        k = self.DT / self.tau
        self.vxy += k * (v_des - self.vxy)
        self.vz += k * (vz_cmd - self.vz)
        self.xy = np.clip(self.xy + self.vxy * self.DT, -self.AREA, self.AREA)
        self.h = float(np.clip(self.h + self.vz * self.DT, 0.0, 120.0))

        self.t += self.DT
        self.t_fix += self.DT
        if self.t_fix >= self.FIX_DT:
            self._new_fix()

        acq = self._acquired()
        miss = float(np.linalg.norm(self.pad - self.xy))
        r = self._reward(acq, miss)

        terminated = self.h <= self.H_GROUND
        truncated = (not terminated) and self.t >= self.T_MAX
        if terminated:
            r += 100.0 if miss <= self.PAD_R else -(60.0 + min(miss, 40.0))
        elif truncated:
            r -= 40.0                            # out of fuel, still airborne

        info = {"miss_m": miss, "acquired": bool(acq),
                "ever_acquired": bool(self.was_acquired),
                "sigma_m": float(self.sigma), "h_m": self.h, "t_s": self.t,
                "success": bool(terminated and miss <= self.PAD_R)}
        return self._obs(), float(r), terminated, truncated, info

    def _reward(self, acq, miss):
        """
        Continuous gates rather than switching rules, per WaveLander: descent is
        rewarded when the situation is good and penalised when it is not, so
        hold / descend / climb falls out of the sign instead of an `if` ladder.
        """
        r = -0.02                                # time pressure, small

        if acq:
            r += 0.5                             # marker in view
            r += 0.3 * (-self.vz / self.VZ_MAX)  # descending is good now
            r -= 0.02 * miss
        else:
            # Blind. Descending below the detection ceiling without the marker
            # spends altitude you cannot cheaply get back -- gate it on how much
            # of the position error the camera footprint could cover.
            reach = self.h * self.fov_tan
            trust = np.clip(reach / max(3.0 * self.sigma, 1e-6), 0.0, 1.0)
            if self.h >= self.h_detect:
                gate = 1.0                       # above the ceiling: descend
            else:
                gate = 2.0 * trust - 1.0         # below: only if plausibly covered
            r += 0.25 * gate * (-self.vz / self.VZ_MAX)

        if acq and not self.was_acquired:
            r += 10.0                            # first acquisition
            self.was_acquired = True
        return r


# ---------------------------------------------------------------------------
def demo(pool=None):
    """One runnable check: the physics and the ablation wiring, not the policy.

    With no argument it builds a synthetic pool so this runs with no GeoAnchor
    data present. Pass a path to run the precondition against the real pool --
    which is the run that actually decides whether the study is viable.
    """
    import os
    import tempfile

    if pool is not None:
        p = pool
    else:
        n = 600
        rng = np.random.default_rng(0)
        sig = rng.uniform(5, 45, n)
        err = np.abs(rng.normal(0, sig))
        sc = rng.choice([f"Scene_{i:02d}" for i in range(5)], n)
        p = os.path.join(tempfile.mkdtemp(), "pool.npz")
        np.savez(p, sigma=sig, err=err, scene=sc)

    e = ApproachEnv(pool=p, arm="sigma", seed=0)

    # The detection ceiling is what makes this a decision problem. If it drifts
    # above the spawn altitude the task becomes trivial.
    assert 25.0 < e.h_detect < 35.0, e.h_detect
    assert e.h_detect < e.H0[0], "marker visible from spawn -- task is trivial"

    obs, _ = e.reset(seed=1)
    assert obs.shape == (6,) and np.isfinite(obs).all()

    # Arms must differ in exactly one channel and nothing else.
    envs = {a: ApproachEnv(pool=p, arm=a, seed=0) for a in
            ("none", "staleness", "sigma")}
    o = {a: v.reset(seed=1)[0] for a, v in envs.items()}
    assert o["none"][3] == 0.0 and o["sigma"][3] > 0 and o["staleness"][3] > 0
    for a in ("sigma", "staleness"):
        assert np.allclose(np.delete(o[a], 3), np.delete(o["none"], 3)), \
            f"{a} arm differs outside the uncertainty channel"

    # THE PRECONDITION FOR THE WHOLE STUDY.
    #
    # Commit-and-dive is the sigma-blind strategy. It must succeed when sigma is
    # small and fail when sigma is large -- that gap is the only thing a
    # sigma-aware policy can exploit. If diving works equally well at every
    # sigma, the uncertainty channel carries nothing and no amount of training
    # will produce a result. Check it before spending a GPU-week.
    lo, hi = [], []
    for s in range(120):
        e.reset(seed=s)
        sigma0 = e.sigma
        while True:
            # max descent, tracking every fix: the sigma-blind strategy
            _, _, term, trunc, i = e.step([-1.0, 1.0])
            if term or trunc:
                (hi if sigma0 > 25.0 else lo).append(i["success"])
                break
    lo_r, hi_r = float(np.mean(lo)), float(np.mean(hi))
    assert len(lo) > 10 and len(hi) > 10, "pool has no spread in sigma"
    assert lo_r - hi_r > 0.15, (
        f"diving succeeds {lo_r:.0%} at low sigma and {hi_r:.0%} at high -- "
        "no sigma-dependent gap, so there is nothing for the policy to learn")

    # And the replayed error must actually move the believed position.
    e.reset(seed=3)
    assert np.linalg.norm(e._believed_xy() - e.xy) > 0.5

    print(f"ok  h_detect={e.h_detect:.1f} m  footprint@30m={30 * e.fov_tan:.1f} m")
    print(f"    dive succeeds {lo_r:.0%} at low sigma vs {hi_r:.0%} at high "
          f"-- {lo_r - hi_r:.0%} gap for a sigma-aware policy to close")


if __name__ == "__main__":
    import sys
    demo(sys.argv[1] if len(sys.argv) > 1 else None)
