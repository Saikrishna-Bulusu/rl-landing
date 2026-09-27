#!/usr/bin/env python3
"""
approach_env_v2.py -- varying-sigma follow-up. SEPARATE experiment; every v1
result stands as the "constant sigma" condition.

AN EPISODE IS A SCENE. That is the whole design, and it is measured, not invented.

v1 held sigma constant for a whole landing. The data says otherwise: within a
single scene, sigma spans p10->p90 ratios of 2-7x. But the first attempt at a
fix (giving each episode a contiguous 40% window of POOLED sigma ranks) failed
in two ways at once, and both are worth recording:

  - it attenuated rank-corr(sigma, err) from +0.341 to +0.211, because a window
    of the pooled ordering is not what a real scene looks like;
  - removing the window then let every episode draw from the entire pool, so
    EVERY landing contained an easy fix and all four arms saturated at 97-100%
    including the meaningless `random` control. A saturated task measures
    nothing.

The measurement that settles it:

    scene       n   sigma p10  p50   p90    sd   within-scene    fixes < 10 m
    Scene_09  149      11.2   11.9  20.1   8.5      +0.317           6.7%
    Scene_10  202      11.4   15.2  29.1   7.0      +0.510           0.0%
    Scene_16   52       9.1   11.4  13.0   2.1      +0.028          36.5%
    Scene_22   37      14.7   27.9  52.2  15.3      +0.621           0.0%
    pooled rank-corr +0.341   mean within-scene +0.369

A hard scene contains NO easy fixes. That is the property the band destroyed and
the property that makes the task a decision rather than a lottery. So an episode
now samples a real SCENE, and its fixes are drawn from that scene's own rows.
Everything follows from the data: within-scene sigma spread, within-scene
sigma->error relationship, between-scene difficulty, and the easy-fix supply.

Also fixed here, carried over from the first v2 attempt:

  THE REWARD NO LONGER LEAKS SIGMA. v1's `_reward` computed its blind-descent
  gate from `self.sigma` for EVERY arm, so even `none` was shaped toward
  sigma-aware behaviour. That is a plausible reason the sigma OBSERVATION bought
  nothing in v1. The gate now uses a fixed reference, identical for all arms.

  EVALUATE BY SCENE, NOT BY FIRST-FIX SIGMA. `train.py` labels an episode by the
  sigma at reset. Once sigma varies that is nearly meaningless -- measured
  corr(first-fix sigma, episode-mean sigma) = +0.179. Episodes are now grouped
  by SCENE, whose difficulty is a real, reportable property.

LIMITATION, STATE IT: there are only 4 usable scenes, so episodes come in 4
difficulty flavours. Variety within a flavour comes from different fix
sequences, not different scenes. With more scenes on disk this would improve.
"""

import math

import numpy as np

from approach_env import ApproachEnv


class ApproachEnvV2(ApproachEnv):

    def __init__(self, *a, **kw):
        arm = kw.get("arm", "sigma")
        if arm == "random":
            kw["arm"] = "sigma"          # parent validates the arm name
        super().__init__(*a, **kw)
        self.arm = arm

        # Group the pool BY SCENE. Each scene's rows are sorted by sigma so an
        # AR(1) latent can walk that scene's own sigma distribution.
        self.scenes = []
        for s in sorted(set(self.pool_scene)):
            m = self.pool_scene == s
            if m.sum() < 20:
                continue
            o = np.argsort(self.pool_sigma[m])
            self.scenes.append({"name": str(s),
                                "sigma": self.pool_sigma[m][o],
                                "err": self.pool_err[m][o]})
        assert self.scenes, "no scene has enough fixes"
        self.SIGMA_REF = float(np.median(self.pool_sigma))

    # ---- localisation ----------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info or {})
        info["scene"] = self._scene["name"]
        return self._obs(), info

    def _pick_episode(self, r):
        self._scene = self.scenes[int(r.integers(len(self.scenes)))]
        # CONTROL arm: a meaningless per-episode constant with the same
        # distribution as the sigma channel, independent of everything.
        self._fake = float(self.pool_sigma[int(r.integers(len(self.pool_sigma)))])

    def _new_fix(self):
        """One fix = one real row FROM THIS EPISODE'S SCENE."""
        sc = self._scene
        self._z = (self.rho * self._z +
                   np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * self.rng.normal())
        u = 0.5 * (1.0 + math.erf(self._z / np.sqrt(2.0)))
        i = int(np.clip(round(u * (len(sc["sigma"]) - 1)), 0, len(sc["sigma"]) - 1))

        self.sigma = float(sc["sigma"][i]) * self.sigma_scale
        mag = float(sc["err"][i]) * self.sigma_scale
        self._bearing += self.rng.normal(0.0, self.BEARING_STEP)
        self.err_xy = mag * np.array([np.cos(self._bearing),
                                      np.sin(self._bearing)])
        self.t_fix = 0.0

    # ---- observation -----------------------------------------------------
    def _obs(self):
        o = super()._obs()
        if self.arm == "random":
            o = o.copy()
            o[3] = np.float32(getattr(self, "_fake", 0.0) / 50.0)
        return o

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        info["scene"] = self._scene["name"]
        return obs, r, term, trunc, info

    # ---- reward ----------------------------------------------------------
    def _reward(self, acq, miss):
        """v1's reward, except the blind-descent gate uses a CONSTANT."""
        r = -0.02
        if acq:
            r += 0.5
            r += 0.3 * (-self.vz / self.VZ_MAX)
            r -= 0.02 * miss
        else:
            reach = self.h * self.fov_tan
            trust = np.clip(reach / max(3.0 * self.SIGMA_REF, 1e-6), 0.0, 1.0)
            gate = 1.0 if self.h >= self.h_detect else (2.0 * trust - 1.0)
            r += 0.25 * gate * (-self.vz / self.VZ_MAX)
        if acq and not self.was_acquired:
            r += 10.0
            self.was_acquired = True
        return r


_orig_reset = ApproachEnv.reset


def _v2_reset(self, seed=None, options=None):
    if isinstance(self, ApproachEnvV2):
        self._pick_episode(np.random.default_rng(seed) if seed is not None
                           else self.rng)
    return _orig_reset(self, seed=seed, options=options)


ApproachEnv.reset = _v2_reset


# ---------------------------------------------------------------------------
def demo(pool="sigma_pool.npz"):
    """Runnable checks, each one guarding a failure this project already hit."""
    e = ApproachEnvV2(pool=pool, arm="sigma", seed=0, t_max=45, rho=0.707)
    rank = lambda x: np.argsort(np.argsort(x)).astype(float)

    # 1. sigma must VARY within an episode (the point of v2).
    e.reset(seed=1)
    seen = [e.sigma]
    for _ in range(200):
        e.step([0.0, -1.0, 1.0]); seen.append(e.sigma)
    assert np.std(seen) > 0.3, f"sigma barely varies: sd={np.std(seen):.3f}"

    # 2. HARD SCENES MUST NOT CONTAIN EASY FIXES. This is the check whose
    #    absence let every episode saturate at 97-100%.
    per = {}
    for s in range(400):
        _, info = e.reset(seed=s)
        v = [e.sigma]
        for _ in range(60):
            _, _, t, tr, _ = e.step([0.0, -1.0, 1.0]); v.append(e.sigma)
            if t or tr: break
        per.setdefault(info["scene"], []).append(min(v))
    frac_easy = {k: float(np.mean(np.array(v) < 10.0)) for k, v in per.items()}
    assert min(frac_easy.values()) < 0.10, (
        "every scene supplies an easy fix -- the task will saturate: "
        + str({k: round(v, 2) for k, v in frac_easy.items()}))

    # 3. the within-scene sigma->error relationship must survive the replay.
    sg, er = [], []
    for s in range(250):
        e.reset(seed=s)
        for _ in range(30):
            e.step([0.0, -1.0, 1.0])
            sg.append(e.sigma); er.append(np.linalg.norm(e.err_xy))
    c = np.corrcoef(rank(np.array(sg)), rank(np.array(er)))[0, 1]
    assert c > 0.20, f"sigma->error relationship lost in replay: {c:+.3f}"

    # 4. the reward must not leak sigma to any arm.
    e.reset(seed=5); e.sigma = 5.0;  lo = e._reward(False, 20.0)
    e.sigma = 80.0;                  hi = e._reward(False, 20.0)
    assert abs(lo - hi) < 1e-9, f"reward still leaks sigma: {lo:.4f} vs {hi:.4f}"

    # 5. arms differ in exactly one channel.
    envs = {a: ApproachEnvV2(pool=pool, arm=a, seed=0, t_max=45, rho=0.707)
            for a in ("none", "staleness", "sigma", "random")}
    o = {a: v.reset(seed=1)[0] for a, v in envs.items()}
    assert o["none"][3] == 0.0
    for a in ("sigma", "staleness", "random"):
        assert o[a][3] > 0, f"{a} channel is dead"
        assert np.allclose(np.delete(o[a], 3), np.delete(o["none"], 3)), \
            f"{a} arm differs outside the uncertainty channel"

    print(f"ok  episodes are real scenes: {[s['name'] for s in e.scenes]}")
    print(f"    sigma varies within an episode: sd {np.std(seen):.2f} m")
    print(f"    rank-corr(sigma, err) in replay: {c:+.3f}  (pool +0.341)")
    print("    fraction of episodes with an easy (<10 m) fix, per scene:")
    for k, v in sorted(frac_easy.items()):
        print(f"      {k:<10} {100*v:5.1f}%")
    print(f"    reward is sigma-free for all arms (SIGMA_REF={e.SIGMA_REF:.1f} m)")


if __name__ == "__main__":
    demo()
