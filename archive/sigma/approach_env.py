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

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def decode_curve(path="detect_curve_60m.json", marker="0"):
    """
    p(decode) against apparent marker size, FITTED TO THE MEASURED COUNTS.

    MIN_PX = 34 is a cliff: below it the marker is invisible, above it perfect.
    The measured curve is a ramp, not a cliff -- 0% at 10-15 px, 2% at 15-20,
    38% at 20-25, 84% at 25-30, 100% by 35 px.

    detect_curve.json could not show that: it was flown from 26 m, so 25 px was
    its floor and everything below was extrapolation. detect_curve_60m.json is
    the same flight from 60 m, which actually samples the tail.

    KNOWN LIMITATION. These rates are measured within 2 m of directly overhead.
    In flight the 1.20 m marker decodes at 14 px when it sits near the edge of
    the frame, where a 99.7 deg rectilinear projection stretches it. So this
    curve understates detection off-axis, and the env's footprint is a circle
    of uniform detectability inside it.

    So fit a logistic to the binomial counts in detect_curve.json rather than
    threshold on them. The 400-2000 px bin is excluded: its 45% is the marker
    OUTGROWING the frame on short final, a different failure that the real
    aircraft answers with the finer marker in its ladder and that this
    single-marker env does not model.

    Returns (x0, s) for p(px) = 1 / (1 + exp(-(px - x0) / s)).
    """
    import json
    from scipy.optimize import minimize

    bins = [b for b in json.load(open(path))[marker] if b["px_hi"] <= 400]
    x = np.array([0.5 * (b["px_lo"] + b["px_hi"]) for b in bins], float)
    n = np.array([b["n"] for b in bins], float)
    k = np.round(np.array([b["rate"] for b in bins], float) * n)

    def nll(th):
        p = 1.0 / (1.0 + np.exp(-(x - th[0]) / max(abs(th[1]), 1e-3)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))

    r = minimize(nll, [30.0, 1.5], method="Nelder-Mead")
    return float(r.x[0]), float(abs(r.x[1]))


class ApproachEnv(gym.Env):
    metadata = {"render_modes": []}

    # ---- world ----------------------------------------------------------
    AREA = 80.0          # half-extent of the operating box (m)
    H0 = (45.0, 65.0)    # spawn altitude range (m AGL)
    H_GROUND = 1.0       # touchdown altitude (m)
    PAD_R = 0.75         # success radius at touchdown (m)

    # ---- vehicle (sub-1 kg delivery quad) -------------------------------
    VZ_MAX = 2.0         # commanded descent/climb, m/s
    SEARCH_R_MAX = 25.0  # a[1] maps to 0..this, metres
    VXY_MAX = 5.0        # lateral speed cap, m/s
    # FIRST-ORDER VELOCITY LAG, MEASURED ON THE REAL FIRMWARE.
    #
    # measure_tau.py commanded PX4 in Gazebo a velocity step and timed the 63.2%
    # crossing (px4_response.json):
    #
    #     lateral  0.69 s, settling at 2.02 m/s against a 2.00 m/s command
    #     vertical 0.23 s, settling at 1.52 m/s against a 1.50 m/s command
    #
    # Two things were wrong before. The lag was a single number for both axes,
    # but the real aircraft changes descent rate THREE TIMES faster than it
    # changes ground speed -- it only has to change thrust, not tilt the whole
    # airframe -- so the env was making vertical decisions look more sluggish
    # and lateral ones less sluggish than they are. And it was drawn uniformly
    # from 0.30-0.80, whose mean of 0.55 is a noticeably more agile aircraft
    # than the 0.69 we actually fly.
    TAU_LAT = 0.69
    TAU_VER = 0.23

    # How fast the ERROR DIRECTION wanders between fixes. 0.02 rad makes the
    # direction effectively fixed for a whole episode, which is what v1-v3
    # assumed; v4 overrides it with the value measured from GeoAnchor's own
    # per-frame error vectors. Left here so earlier results reproduce exactly.
    BEARING_STEP = 0.02

    # ---- camera ---------------------------------------------------------
    # These lines set the whole difficulty of the task. Do not tune them to make
    # training easier -- they are the physics of seeing a 1 m marker.
    # CAMERA, matched to the x500_mono_cam_down actually flown in Gazebo
    # (intrinsics read off the camera_info topic, not assumed: fx = 539.94).
    FOV_DEG = 99.7
    PX = 1280
    MIN_PX = 34.0        # 20 px is where decoding BEGINS to fail; 34 is where
                         # it is reliable, measured against this camera
    PAD_SIZE = 1.20      # the COARSE marker -- the one acquisition depends on

    # How detection is modelled. "threshold" is the original cliff at MIN_PX;
    # "curve" samples from the logistic fitted to the measured counts.
    #
    # "curve" IS THE MEASURED ONE AND THE DEFAULT IS STILL "threshold". That
    # needs justifying. The cliff is wrong in both directions -- it says a
    # 1.20 m marker is invisible above 19.1 m when it in fact decodes half the
    # time at 27.2 m, and certain below it when it is not -- and re-running
    # bigger_marker.py under both, the crash reduction credited to a 2.40 m pad
    # fell from 49.2 pp to 30.5 pp. So "curve" should win.
    #
    # It cannot be flipped as a one-liner. Under "curve" this env's own demo()
    # guard fails: diving succeeds 100% at low sigma and 98% at high, so the
    # sigma-dependent gap the whole study rests on disappears at these defaults.
    # A more capable detector makes the v1 scenario trivial, and the fix is to
    # retune its altitudes -- a separate job with its own re-baselining, not a
    # default change smuggled in here. v4/v5 configure their own difficulty and
    # pass "curve" explicitly, which is how the marker sweep was run.
    DETECT = "threshold"

    # ---- episode --------------------------------------------------------
    DT = 0.2             # control period (s) -> 5 Hz decisions
    T_MAX = 120.0        # seconds
    FIX_DT = 0.5         # GeoAnchor fix interval (s) -> 2 Hz

    def __init__(self, pool="sigma_pool.npz", arm="sigma", seed=None,
                 sigma_scale=1.0, lam=0.5, t_max=None, vxy_max=None, rho=None):
        """
        t_max / vxy_max override the class defaults. They are the two honest
        difficulty knobs: loiter time is really a battery reserve, and lateral
        speed is really an airframe limit. If a sigma-blind sweep saturates on
        the real pool, tighten THESE -- do not weaken the baseline, and do not
        touch the camera constants, which are physics.
        """
        super().__init__()
        # "truesigma" is not a system anyone could build -- it hands the policy
        # the EXACT magnitude of the error on the current fix, which requires
        # the ground truth the aircraft is trying to find. It exists to settle
        # one question: our sigma predictor is only about 31% accurate (ENCE
        # 0.313), so a null result for sigma could always be blamed on the
        # predictor rather than on the idea. If a PERFECT uncertainty number
        # also gives no benefit, that excuse is gone and the null is final.
        # "qNN" is truesigma DEGRADED to a known rank-correlation NN/100 with
        # the real error -- q100 is perfect ordering, q40 is badly noisy. It
        # exists because the obvious way to ask "how good must sigma be?" --
        # building pools whose sigma column is re-coupled to the error -- also
        # changes WHICH ERRORS the environment draws, since episodes are
        # sampled by sigma decile. That silently changes the task along with
        # the information and the comparison means nothing. Degrading the
        # OBSERVED CHANNEL instead holds the pool, the errors and the episode
        # distribution fixed, exactly as the truesigma arm does.
        assert (arm in ("none", "staleness", "sigma", "truesigma")
                or (arm.startswith("q") and arm[1:].isdigit())), arm
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
        # Sorted once so a fix is an O(1) empirical-quantile lookup.
        self.bin_err_sorted = [np.sort(self.pool_err[b]) for b in self.sigma_bins]

        # Temporal correlation of the localisation error. Measured from the
        # real frame ordering by calibrate.lag1_rank_corr and carried in the
        # pool; rho=0 (iid fixes) was the original assumption and it is the one
        # extreme where "loiter and wait for a lucky fix" dominates everything.
        # rho=1 is the other, where loitering is worthless. Real data is between.
        self.rho = float(rho) if rho is not None else float(d["rho"]) \
            if "rho" in d.files else 0.0

        fov = np.radians(self.FOV_DEG)
        self.fov_tan = np.tan(fov / 2.0)
        # Above this altitude the marker spans fewer than MIN_PX and cannot be
        # decoded no matter where the drone is.
        #
        # This used to divide by the FOV in radians, a small-angle
        # approximation. At the old 60 deg that was 10% optimistic and did not
        # matter; at the real camera's 99.7 deg it is 37% optimistic, which
        # would have told the policy it could acquire the pad from 26 m when
        # the aircraft actually loses it at 19 m. Use the tangent.
        self.h_detect = self.PAD_SIZE * self.PX / (2.0 * self.fov_tan
                                                   * self.MIN_PX)
        # Focal length in pixels, so apparent size is PAD_SIZE * F_PX / h.
        # Same 539.94 that was read off the camera_info topic in Gazebo.
        self.f_px = self.PX / (2.0 * self.fov_tan)
        self._curve = decode_curve() if self.DETECT == "curve" else None
        if self._curve is not None:
            # h_detect is what the RULE is designed around -- "descend until you
            # can reliably see". Leaving it at the MIN_PX value while the physics
            # became a ramp would handicap the rule for a reason that has nothing
            # to do with the question, so move it to the altitude where the
            # measured curve reaches 90%.
            x0, sc = self._curve
            self.h_detect = self.PAD_SIZE * self.f_px / (x0 - sc * np.log(1/0.9 - 1))

        # a[0] vertical velocity, a[1] search radius, a[2] chase gain.
        #
        # Both lateral levers are present because which one matters is set by
        # rho. At rho=0 the error re-rolls every fix, so holding still and
        # waiting beats searching and a sweep is provably inert (r=0 ties the
        # best tuned sweep). At rho=1 the error is a fixed bias no amount of
        # waiting removes, and only spatial search can find the pad. Real data
        # sits between, so neither lever can be dropped a priori.
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

    # ---- localization ---------------------------------------------------
    def _new_fix(self):
        """
        Draw the next replayed fix. Sigma is fixed for the episode; the error
        magnitude comes from the real errors that accompany that sigma, drawn
        through an AR(1) GAUSSIAN COPULA so that consecutive fixes are
        correlated the way real ones are.

        Why a copula rather than an AR(1) on the magnitude itself: the latent
        z_t is standard normal at stationarity, so u_t = Phi(z_t) is exactly
        Uniform(0,1) and mag_t is exactly the empirical quantile function of the
        bin evaluated there. P(error | sigma) therefore stays EXACTLY the
        measured distribution -- the invariant this whole replay design exists
        to protect -- while gaining a time axis it previously assumed away.
        An AR(1) on the raw heavy-tailed magnitude would distort that marginal.

        The bearing drifts only slightly. Map-match error direction is set by
        scene structure, not white noise, and a fast-rotating bearing would let
        a drone that simply descends sweep a free search pattern -- which would
        silently destroy the experiment by making the sigma-blind policy good.
        """
        self._z = (self.rho * self._z +
                   np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * self.rng.normal())
        u = 0.5 * (1.0 + math.erf(self._z / np.sqrt(2.0)))
        e = self._bin_err
        mag = float(e[int(round(u * (len(e) - 1)))]) * self.sigma_scale
        self._bearing += self.rng.normal(0.0, self.BEARING_STEP)
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
        # Inside the camera footprint is necessary under either model.
        if np.linalg.norm(self.pad - self.xy) > self.h * self.fov_tan:
            return False
        if self._curve is None:
            return self.h <= self.h_detect
        x0, sc = self._curve
        span = self.PAD_SIZE * self.f_px / max(self.h, 1e-6)
        return self._roll < 1.0 / (1.0 + np.exp(-(span - x0) / sc))

    def _visual_target(self):
        """
        Where the camera says to fly, once something is acquired. The base env
        has exactly one marker, so this is the pad. Subclasses that model more
        than one marker in the world override it.
        """
        return self.pad

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
        self.tau = self.TAU_LAT          # kept: the guidance law uses it
        self.t = 0.0
        self.phase = r.uniform(0, 2 * np.pi)   # search spiral phase
        self.was_acquired = False
        # ONE draw per step, not one per call. _acquired() is asked three times
        # a step (the observation, the reward, the termination check); drawing
        # separately would let the marker be visible to the reward and invisible
        # to the observation in the same instant.
        self._roll = float(r.random())

        # Pick this episode's conditions: one sigma decile, held for the episode.
        b = int(r.integers(len(self.sigma_bins)))
        self._bin = self.sigma_bins[b]
        self._bin_err = self.bin_err_sorted[b]
        self.sigma = float(np.median(self.pool_sigma[self._bin])) * self.sigma_scale
        self._bearing = r.uniform(0, 2 * np.pi)
        self._z = float(r.normal())          # AR(1) latent, at stationarity
        self._new_fix()
        self.target = self._believed_pad()     # the low-passed guidance target

        return self._obs(), {}

    def _obs(self):
        # What the aircraft believes its horizontal distance to the pad is.
        d_bel = float(np.linalg.norm(self._believed_pad() - self.xy))

        # The three arms differ ONLY in this channel.
        if self.arm == "sigma":
            unc = self.sigma / 50.0
        elif self.arm == "truesigma":
            unc = float(np.linalg.norm(self.err_xy)) / 50.0
        elif self.arm.startswith("q"):
            unc = self._degraded_sigma() / 50.0
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
        search_r = (a[1] + 1.0) * 0.5 * self.SEARCH_R_MAX
        chase = (a[2] + 1.0) * 0.5              # 0..1

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
            # Fly to whichever marker was actually acquired -- NOT unconditionally
            # to the true pad. Hardcoding self.pad here meant a drone fooled by a
            # neighbour's ArUco still flew to the correct address, so being
            # deceived cost nothing and the ArUco safety-layer experiment measured
            # nothing. Subclasses override _visual_target() to model deception.
            target = self._visual_target()
            self.target = target.copy()
        else:
            target = self._search_target(a, search_r, chase)

        # ponytail: plain P controller with a velocity cap. A full PID adds an
        # integral term that only matters for steady-state offset against wind,
        # which this env does not model. Upgrade when wind is added.
        v_des = (target - self.xy) / max(self.tau, 1e-3)
        n = np.linalg.norm(v_des)
        if n > self.VXY_MAX:
            v_des *= self.VXY_MAX / n

        # First-order velocity lag, per axis, from the measured firmware.
        self.vxy += (self.DT / self.TAU_LAT) * (v_des - self.vxy)
        self.vz += (self.DT / self.TAU_VER) * (vz_cmd - self.vz)
        self.xy = np.clip(self.xy + self.vxy * self.DT, -self.AREA, self.AREA)
        self.h = float(np.clip(self.h + self.vz * self.DT, 0.0, 120.0))

        self._roll = float(self.rng.random())
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

    def _degraded_sigma(self):
        """
        The true error, pushed through a Gaussian copula so its RANK agrees
        with the truth only `rho` of the time, then mapped onto the real
        sigma distribution's own quantiles so the number the policy sees has
        a realistic scale whatever its quality.
        """
        from scipy.stats import norm
        rho = int(self.arm[1:]) / 100.0
        e = float(np.linalg.norm(self.err_xy))
        u = float(np.clip(np.mean(self.pool_err < e), 1e-4, 1 - 1e-4))
        z = rho * norm.ppf(u) + np.sqrt(max(1 - rho ** 2, 0.0)) * self.rng.normal()
        return float(np.quantile(self.pool_sigma,
                                 float(np.clip(norm.cdf(z), 1e-4, 1 - 1e-4))))

    def _search_target(self, a, search_r, chase):
        """
        Where to look when the marker is NOT in view.

        A hook, because this is the decision under study and subclasses need to
        replace it without copying the whole guidance block around it. The base
        behaviour is a fixed-rate spiral offset from the newest fix, damped by
        a[2]; v4 replaces the fixed spiral with a bearing the policy chooses.
        """
        self.phase += 0.9 * self.DT
        raw = self._believed_pad() + search_r * np.array(
            [np.cos(self.phase), np.sin(self.phase)])
        self.target = (1.0 - chase) * self.target + chase * raw
        return self.target

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
    # Was "25 < h_detect < 35", which stopped being true when the small-angle
    # approximation was replaced by the tangent and h_detect fell to 19.1 m. It
    # had been failing under every detector since. What it meant to check is
    # that the marker is NOT visible from the spawn altitude -- that is the next
    # line -- and that the number is physical rather than degenerate.
    assert 5.0 < e.h_detect < 60.0, e.h_detect
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
            _, _, term, trunc, i = e.step([-1.0, -1.0, 1.0])
            if term or trunc:
                (hi if sigma0 > 25.0 else lo).append(i["success"])
                break
    lo_r, hi_r = float(np.mean(lo)), float(np.mean(hi))
    assert len(lo) > 10 and len(hi) > 10, "pool has no spread in sigma"
    assert lo_r - hi_r > 0.15, (
        f"diving succeeds {lo_r:.0%} at low sigma and {hi_r:.0%} at high -- "
        "no sigma-dependent gap, so there is nothing for the policy to learn")

    # The copula must preserve the empirical marginal AND hit the target
    # autocorrelation. If the marginal drifts, P(error | sigma) is no longer the
    # measured distribution and the replay design has quietly become a
    # simulation -- the exact failure this env exists to avoid.
    ec = ApproachEnv(pool=p, arm="sigma", seed=0, rho=0.6)
    ec.reset(seed=1)
    mags = []
    for _ in range(20000):
        ec._new_fix()
        mags.append(np.linalg.norm(ec.err_xy))
    m = np.asarray(mags)
    rk = lambda x: np.argsort(np.argsort(x)).astype(float)
    ac = float(np.corrcoef(rk(m[:-1]), rk(m[1:]))[0, 1])
    assert abs(ac - 0.6) < 0.08, f"copula autocorr {ac:.3f} != rho 0.6"
    for q in (50, 90):
        got, want = np.percentile(m, q), np.percentile(ec._bin_err, q)
        assert abs(got - want) <= 0.1 * want + 1.0, \
            f"copula distorted the p{q} marginal: {got:.2f} vs {want:.2f}"

    # And the replayed error must actually move the believed position.
    e.reset(seed=3)
    assert np.linalg.norm(e._believed_xy() - e.xy) > 0.5

    print(f"ok  h_detect={e.h_detect:.1f} m  footprint@30m={30 * e.fov_tan:.1f} m")
    print(f"    dive succeeds {lo_r:.0%} at low sigma vs {hi_r:.0%} at high "
          f"-- {lo_r - hi_r:.0%} gap for a sigma-aware policy to close")


if __name__ == "__main__":
    import sys
    demo(sys.argv[1] if len(sys.argv) > 1 else None)
