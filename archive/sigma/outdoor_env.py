#!/usr/bin/env python3
"""
outdoor_env.py -- the delivery-scale landing task, in MuJoCo, on a real x500.

WHAT THIS REPLACES

Every sigma result so far came from `approach_env`, a numpy point mass with no
physics engine: no attitude, no thrust limits, no ground effect. It also touched
down at 2.00 m/s on every landing, because nothing in its reward cared about
impact speed. MuJoCo turned out to be only 6.4x slower (7,167 vs 45,679
steps/s), not the 100x the original handoff assumed, so there was never a good
reason to avoid it.

WHAT THE POLICY CONTROLS, AND WHY

The policy does NOT fly the aircraft. It emits two numbers:

    a[0]  vertical rate command   ->  [-V_DESC_MAX, +V_CLIMB_MAX] m/s
    a[1]  lateral approach speed  ->  [0, V_LAT_MAX] m/s toward the believed pad

and a cascaded velocity controller (x500_vel) converts those into thrust and
attitude. This mirrors PX4 offboard velocity control, which is the interface the
Gazebo x500 actually accepts, so the decision the policy learns transfers even
though the low-level control does not.

The alternative -- learning raw motor commands -- would spend most of the
network's capacity learning to hover, swamping the sigma signal that is the
entire subject of the experiment.

AIRFRAME

Real X500 parameters read from the PX4 Gazebo model on this machine, not
estimated: 2.064 kg, arm 0.246 m, thrust/weight 1.69. See x500.py. The
thrust/weight in particular was guessed at 2.3 and is really 1.69, which is a
much less agile aircraft.

Physics is MJC_GND: MuJoCo plus GROUND EFFECT, with aerodynamic drag and
downwash deliberately OFF for this first version. Ground effect is kept because
it is the dominant effect in the last metre -- it holds the aircraft at
0.24-0.60 m with ZERO ground contacts and it will not land by descending alone
(see x500_vel.land). Drag and downwash, plus wind and domain randomisation, are
deferred to a second version so this one isolates whether sigma helps at all
under clean dynamics.
"""

import math
import os
import sys

import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces

import x500
import x500_vel

sys.path.insert(0, os.path.expanduser("~/RL_projects/MuJoCo-drones-gym"))
from multi_drone_mujoco.envs.base_aviary import BaseAviary, _generate_aviary_xml  # noqa: E402
from multi_drone_mujoco.utils.enums import Physics, ActionType, ObservationType  # noqa: E402


class OutdoorLandingEnv(BaseAviary):

    # ---- world ----------------------------------------------------------
    AREA = 60.0              # half-extent of the operating box (m)
    H0 = (40.0, 55.0)        # spawn altitude (m AGL)
    PAD_R = 0.75             # success radius (m)

    # ---- camera: the physics of seeing the marker -----------------------
    # Do not tune these to make training easier. They set the whole structure.
    FOV_DEG = 60.0
    PX = 640
    MIN_PX = 20.0
    PAD_SIZE = 1.0

    # ---- envelope -------------------------------------------------------
    V_DESC_MAX = 2.0         # policy may command up to this descent rate
    V_CLIMB_MAX = 1.0
    V_LAT_MAX = 3.0
    T_MAX = 90.0             # mission seconds (longer than point-mass: real flight)
    FIX_DT = 0.5
    POLICY_HZ = 5.0          # the POLICY decides at 5 Hz; the CONTROLLER flies at 48


    # ---- landing quality -------------------------------------------------
    TOUCH_VZ_GOOD = 0.3      # a landing at or below this is clean
    TOUCH_VZ_BAD = 1.5       # at or above this, treat as an airframe-breaking impact
    # Tilt below this is free. Set to 45 deg on Sai's reasoning, and the data
    # agrees more strongly than expected: the velocity controller CAPS tilt at
    # 20 deg by design, so the aircraft physically cannot flail. Measured in
    # flight: median 6.2 deg, p99 20.4 deg, max 20.9 deg, and 0.00% of moments
    # above 45 deg. At the previous 20 deg threshold the penalty fired on 4.38%
    # of moments -- punishing the drone for using its full, safe, controller-
    # approved authority. At 45 deg it never fires, which is right: the hazard
    # it guards against is already impossible.
    TILT_FREE = np.radians(45.0)

    def __init__(self, pool="sigma_pool.npz", arm="sigma", rho=None,
                 sigma_scale=1.0, lam=0.5, n_trees=0, seed=None,
                 ctrl_freq=48, sim_freq=240):
        assert arm in ("none", "staleness", "sigma", "random"), arm
        self.arm = arm
        self.lam = lam
        self.sigma_scale = sigma_scale
        self.n_trees = int(n_trees)
        self._rng = np.random.default_rng(seed)

        d = np.load(pool, allow_pickle=True)
        self.pool_sigma = np.asarray(d["sigma"], float)
        self.pool_err = np.asarray(d["err"], float)
        self.pool_scene = np.asarray(d["scene"]).astype(str)
        self.rho = float(rho) if rho is not None else (
            float(d["rho"]) if "rho" in d.files else 0.0)
        self.scenes = []
        for s in sorted(set(self.pool_scene)):
            m = self.pool_scene == s
            if m.sum() < 20:
                continue
            o = np.argsort(self.pool_sigma[m])
            self.scenes.append({"name": str(s), "sigma": self.pool_sigma[m][o],
                                "err": self.pool_err[m][o]})

        fov = np.radians(self.FOV_DEG)
        self.fov_tan = np.tan(fov / 2.0)
        self.h_detect = self.PAD_SIZE * self.PX / (fov * self.MIN_PX)

        super().__init__(drone_model=x500.X500_MODEL, num_drones=1,
                         physics=Physics.MJC_GND,
                         act_type=ActionType.ATTITUDE,
                         obs_type=ObservationType.KIN,
                         ctrl_freq=ctrl_freq, sim_freq=sim_freq,
                         initial_xyzs=np.array([[0.0, 0.0, self.H0[1]]]))

        # The parent's reset() calls _computeObs(), so every field the
        # observation touches must exist before that ever runs.
        self.pad = np.zeros(2)
        self.err_xy = np.zeros(2)
        self.sigma = float(np.median(self.pool_sigma)) * sigma_scale
        self._fake = self.sigma
        self._scene = self.scenes[0]
        self._bearing = 0.0
        self._z = 0.0
        self.t = 0.0
        self.t_fix = 0.0
        self.was_acquired = False

        # ACTION REPEAT. The controller must run at 48 Hz to fly the aircraft,
        # but the policy is making a slow decision -- descend or wait, chase the
        # fix or hold -- and does not need to re-decide 48 times a second.
        #
        # Without this the episode is up to 4320 policy steps while a PPO
        # rollout is 4096-8192, so a rollout covers UNDER ONE EPISODE and the
        # terminal landing reward is essentially invisible to the update. That
        # is why the preflight returned 0% for both extreme settings.
        #
        # At 5 Hz an episode is ~450 policy steps and a rollout covers ~18 of
        # them. It also matches the point-mass env, which decided at 5 Hz.
        self._repeat = max(1, int(round(self.CTRL_FREQ / self.POLICY_HZ)))

        # One constant reference sigma for the blind-descent gate, identical
        # for every arm so the reward cannot leak sigma to arms that should not
        # see it -- the confound that had to be removed from approach_env v1.
        self.SIGMA_REF = float(np.median(self.pool_sigma)) * sigma_scale

        self.ctrl = x500_vel.VelocityController(self.M, self.HOVER_RPM, self.MAX_RPM)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self._build_scene()

    # ---- scene -----------------------------------------------------------
    def _tree_xml(self, i, x, y, h, r):
        return (f'  <geom name="tree{i}_t" type="cylinder" pos="{x} {y} {h*0.35:.2f}" '
                f'size="{r*0.18:.2f} {h*0.35:.2f}" rgba="0.35 0.25 0.15 1"/>\n'
                f'  <geom name="tree{i}_c" type="sphere" pos="{x} {y} {h*0.8:.2f}" '
                f'size="{r:.2f}" rgba="0.13 0.42 0.15 1"/>\n')

    def _build_scene(self, trees=None):
        """Rebuild the MuJoCo model with a pad and (optionally) trees."""
        xml = _generate_aviary_xml(1, x500.X500_MODEL,
                                   np.array([[0.0, 0.0, self.H0[1]]]),
                                   np.zeros((1, 3)), False, False,
                                   1.0 / self.SIM_FREQ)
        scene = (
            # The DRAWN pad must match PAD_SIZE, the marker the detection model
            # assumes. They disagreed (1.2 m drawn vs 1.0 m modelled), which
            # would have made every rendered frame subtly lie about the task.
            f'  <geom name="pad" type="box" pos="0 0 0.01" '
            f'size="{self.PAD_SIZE/2:.3f} {self.PAD_SIZE/2:.3f} 0.01"'
            ' rgba="0.95 0.95 0.95 1"/>\n'
            f'  <geom name="padx" type="box" pos="0 0 0.02" '
            f'size="{self.PAD_SIZE*0.46:.3f} {self.PAD_SIZE*0.10:.3f} 0.005"'
            ' rgba="0.95 0.45 0.05 1"/>\n'
            f'  <geom name="pady" type="box" pos="0 0 0.02" '
            f'size="{self.PAD_SIZE*0.10:.3f} {self.PAD_SIZE*0.46:.3f} 0.005"'
            ' rgba="0.95 0.45 0.05 1"/>\n')
        for i, (x, y, h, r) in enumerate(trees or []):
            scene += self._tree_xml(i, x, y, h, r)
        xml = xml.replace("</worldbody>", scene + "\n  </worldbody>")
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        mujoco.mj_forward(self.model, self.data)
        self._trees = list(trees or [])

    # ---- localisation replay --------------------------------------------
    def _new_fix(self):
        sc = self._scene
        self._z = (self.rho * self._z +
                   np.sqrt(max(1.0 - self.rho ** 2, 0.0)) * self._rng.normal())
        u = 0.5 * (1.0 + math.erf(self._z / np.sqrt(2.0)))
        i = int(np.clip(round(u * (len(sc["sigma"]) - 1)), 0, len(sc["sigma"]) - 1))
        self.sigma = float(sc["sigma"][i]) * self.sigma_scale
        mag = float(sc["err"][i]) * self.sigma_scale
        self._bearing += self._rng.normal(0.0, 0.02)
        self.err_xy = mag * np.array([np.cos(self._bearing), np.sin(self._bearing)])
        self.t_fix = 0.0

    def _believed_pad(self):
        """Pad in TRUE coordinates as the aircraft believes it. See approach_env."""
        return self.pad - self.err_xy

    def _acquired(self):
        p = self.pos[0]
        if p[2] > self.h_detect:
            return False
        return float(np.linalg.norm(self.pad - p[:2])) <= p[2] * self.fov_tan

    # ---- gym --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        r = self._rng

        trees = []
        for _ in range(self.n_trees):
            while True:
                x, y = r.uniform(-self.AREA * 0.6, self.AREA * 0.6, 2)
                if np.hypot(x, y) > 6.0:       # never on top of the pad
                    break
            trees.append((round(float(x), 2), round(float(y), 2),
                          float(r.uniform(4.0, 9.0)), float(r.uniform(1.2, 2.6))))
        self._build_scene(trees)

        self.pad = np.zeros(2)                  # pad is the origin of the scene
        b = int(r.integers(len(self.scenes)))
        self._scene = self.scenes[b]
        self._bearing = float(r.uniform(0, 2 * np.pi))
        self._z = float(r.normal())
        self._fake = float(self.pool_sigma[int(r.integers(len(self.pool_sigma)))]) \
            * self.sigma_scale
        self.t = 0.0
        self.t_fix = 0.0
        self.was_acquired = False
        self.ctrl.reset()
        self._new_fix()
        super().reset(seed=seed)

        # start displaced by roughly a typical error, above the detection ceiling
        h0 = float(r.uniform(*self.H0))
        off = r.normal(0, 12.0, 2)
        self._set_state(np.array([off[0], off[1], h0]))
        return self._obs(), {"scene": self._scene["name"]}

    def _set_state(self, xyz):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drone0_joint")
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr:adr + 3] = xyz
        self.data.qpos[adr + 3:adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        dadr = self.model.jnt_dofadr[jid]
        self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._updateAndStoreKinematicInformation()

    def _obs(self):
        p = self.pos[0]
        d_bel = float(np.linalg.norm(self._believed_pad() - p[:2]))
        if self.arm == "sigma":
            unc = self.sigma / 50.0
        elif self.arm == "staleness":
            unc = float(np.exp(-self.lam * self.t_fix))
        elif self.arm == "random":
            unc = self._fake / 50.0
        else:
            unc = 0.0
        tilt = float(np.abs(self.rpy[0][:2]).max())
        return np.array([
            p[2] / 60.0,
            float(self.vel[0][2]) / self.V_DESC_MAX,
            min(d_bel, 100.0) / 100.0,
            unc,
            float(self._acquired()),
            min(self.t, self.T_MAX) / self.T_MAX,
            tilt / np.radians(30.0),
        ], np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        vz_cmd = a[0] * (self.V_DESC_MAX if a[0] < 0 else self.V_CLIMB_MAX)
        lat_cap = (a[1] + 1.0) * 0.5 * self.V_LAT_MAX

        dt = 1.0 / self.CTRL_FREQ
        acq = False
        for _ in range(self._repeat):
            p = self.pos[0]
            target = self.pad if self._acquired() else self._believed_pad()
            d = target - p[:2]
            v_xy = np.clip(d * 0.8, -lat_cap, lat_cap)
            u = self.ctrl([v_xy[0], v_xy[1], vz_cmd],
                          self.vel[0], self.rpy[0], self.ang_v[0], dt)
            super().step(np.array([u]))
            self.t += dt
            self.t_fix += dt
            if self.t_fix >= self.FIX_DT:
                self._new_fix()
            if self._acquired():
                acq = True
                self.was_acquired = True
            if self.data.ncon > 0 and self.pos[0][2] < 0.5:
                break
            if self.t >= self.T_MAX:
                break

        p = self.pos[0]
        miss = float(np.linalg.norm(self.pad - p[:2]))
        vz = abs(float(self.vel[0][2]))
        on_ground = self.data.ncon > 0 and p[2] < 0.5
        terminated = bool(on_ground)
        truncated = bool(not terminated and self.t >= self.T_MAX)

        rew = self._reward(acq, miss, vz, terminated, truncated)
        info = {"miss_m": miss, "acquired": bool(acq),
                "ever_acquired": bool(self.was_acquired),
                "sigma_m": float(self.sigma), "h_m": float(p[2]),
                "t_s": self.t, "touch_vz": vz if terminated else None,
                "scene": self._scene["name"],
                "success": bool(terminated and miss <= self.PAD_R
                                and vz <= self.TOUCH_VZ_BAD)}
        return self._obs(), rew, terminated, truncated, info

    def _reward(self, acq, miss, vz, terminated, truncated):
        """
        Rebuilt to match the point-mass reward that trained reliably. Three
        terms were missing, and together they made the task nearly unlearnable:

        1. NO DISTANCE SHAPING. `r -= 0.02 * miss` was dropped, so once the
           marker was in view nothing rewarded getting CLOSER to it. Seeds
           landed 3-10 m away and the run looked "collapsed".
        2. The in-view bonus is +0.5, MATCHING THE POINT-MASS ENV EXACTLY.
           An attempt to "fix" hovering by cutting it to +0.2 made things far
           worse: with the 0.02/metre distance penalty, acquiring at a realistic
           10-15 m became worth 0.00 to -0.10 per step, so FINDING THE MARKER
           WAS PUNISHED. Mean success fell from 20.1% to 6.3% across 6 seeds.
           At +0.5 the net stays positive (+0.30/step at 10 m) while landing
           (+40..100 plus +10) still dominates hovering (+68 at 10 m).
        3. NO SIGNAL WHILE BLIND. Above the detection ceiling the only term was
           time pressure, so there was no gradient to descend at all. The
           point-mass env gated descent on whether the camera footprint could
           plausibly cover the error; the same gate is used here, computed from
           a CONSTANT reference sigma so it leaks nothing to any arm.
        """
        r = -0.02
        if acq:
            r += 0.5                                     # in view (point-mass value)
            r += 0.3 * (-float(self.vel[0][2]) / self.V_DESC_MAX)
            r -= 0.02 * miss                             # CLOSE THE DISTANCE
            if not self.was_acquired:
                r += 10.0
        else:
            # Blind. Descending below the ceiling without the marker spends
            # altitude you cannot cheaply recover -- gate it on how much of a
            # TYPICAL error the footprint could cover. SIGMA_REF is a constant,
            # identical for every arm, so this cannot leak sigma.
            h = float(self.pos[0][2])
            reach = h * self.fov_tan
            trust = np.clip(reach / max(3.0 * self.SIGMA_REF, 1e-6), 0.0, 1.0)
            gate = 1.0 if h >= self.h_detect else (2.0 * trust - 1.0)
            r += 0.25 * gate * (-float(self.vel[0][2]) / self.V_DESC_MAX)

        # Only EXCESSIVE tilt is punished; see TILT_FREE.
        excess = max(0.0, float(np.abs(self.rpy[0][:2]).max()) - self.TILT_FREE)
        r -= 3.0 * excess

        if terminated:
            if miss <= self.PAD_R:
                q = np.clip((self.TOUCH_VZ_BAD - vz)
                            / (self.TOUCH_VZ_BAD - self.TOUCH_VZ_GOOD), 0.0, 1.0)
                r += 40.0 + 60.0 * q
            else:
                r -= 60.0 + min(miss, 40.0)
        elif truncated:
            r -= 40.0
        return float(r)

    def _computeObs(self): return self._obs()
    def _computeReward(self): return 0.0
    def _computeTerminated(self): return False
    def _computeTruncated(self): return False
    def _computeInfo(self): return {}


# ---------------------------------------------------------------------------
def demo(pool="sigma_pool.npz"):
    """Runnable checks, each guarding a failure this project has already hit."""
    kw = dict(pool=pool, rho=0.707, sigma_scale=1.0)
    e = OutdoorLandingEnv(arm="sigma", **kw)

    # 1. THE BLIND PHASE MUST EXIST. If the marker is visible from spawn there
    #    is no search-versus-commit decision and the task is trivial.
    assert e.h_detect < e.H0[0], \
        f"marker visible from spawn ({e.h_detect:.1f} vs {e.H0[0]}) -- task is trivial"

    # 2. THE POLICY MUST BE ABLE TO FLY. An un-normalised action space cost 84
    #    cage runs; the drone must reach descend / hover / climb.
    hs = {}
    for a0, label in ((-1.0, "descend"), (0.0, "hover"), (1.0, "climb")):
        e.reset(seed=1)
        z0 = float(e.pos[0][2])
        for _ in range(120):
            e.step([a0, -1.0])
        hs[label] = float(e.pos[0][2]) - z0
    assert hs["descend"] < -1.0, f"cannot descend: {hs}"
    assert hs["climb"] > 0.3, f"cannot climb: {hs}"
    assert abs(hs["hover"]) < 1.5, f"cannot hold altitude: {hs}"

    # 3. LANDINGS MUST BE GENTLE. The point-mass policy touched down at 2.00 m/s
    #    on every single landing because nothing in its reward cared.
    vzs = []
    for s in range(14):
        e.reset(seed=100 + s)
        while True:
            _, _, t, tr, i = e.step([-0.5, 1.0])
            if t or tr:
                if t:
                    vzs.append(i["touch_vz"])
                break
    assert vzs, "nothing ever landed"
    assert max(vzs) < 1.0, f"hard impacts: {['%.2f' % v for v in vzs]}"

    # 4. SIGMA MUST VARY WITHIN AN EPISODE (the v2 fix) and the joint must hold.
    e.reset(seed=3)
    sg = [e.sigma]
    for _ in range(200):
        e.step([-0.2, 0.0]); sg.append(e.sigma)
    assert np.std(sg) > 0.3, f"sigma barely varies: sd={np.std(sg):.3f}"

    # 5. ARMS DIFFER IN EXACTLY ONE CHANNEL.
    envs = {a: OutdoorLandingEnv(arm=a, **kw)
            for a in ("none", "staleness", "sigma", "random")}
    o = {a: v.reset(seed=7)[0] for a, v in envs.items()}
    assert o["none"][3] == 0.0
    for a in ("sigma", "staleness", "random"):
        assert o[a][3] > 0, f"{a} channel is dead"
        assert np.allclose(np.delete(o[a], 3), np.delete(o["none"], 3), atol=1e-5), \
            f"{a} arm differs outside the uncertainty channel"

    # 6. TREES MUST SPAWN, AND MUST NOT BE ON THE PAD.
    et = OutdoorLandingEnv(arm="sigma", n_trees=6, **kw)
    et.reset(seed=2)
    assert len(et._trees) == 6, et._trees
    for x, y, _, _ in et._trees:
        assert np.hypot(x, y) > 6.0, f"tree on the pad at ({x},{y})"
    n_geom = et.model.ngeom
    et2 = OutdoorLandingEnv(arm="sigma", n_trees=0, **kw); et2.reset(seed=2)
    assert n_geom > et2.model.ngeom, "trees did not reach the model"

    print(f"ok  x500 {e.M:.3f} kg, T/W {e.THRUST2WEIGHT_RATIO:.2f}, "
          f"h_detect {e.h_detect:.1f} m, footprint {e.h_detect*e.fov_tan:.1f} m")
    print(f"    flight authority: descend {hs['descend']:+.1f} m, hover "
          f"{hs['hover']:+.1f} m, climb {hs['climb']:+.1f} m over 2.5 s")
    print(f"    touchdown speeds over {len(vzs)} landings: "
          f"median {np.median(vzs):.2f} m/s, worst {max(vzs):.2f} m/s")
    print(f"    sigma varies within an episode: sd {np.std(sg):.2f} m")
    print(f"    trees: {len(et._trees)} spawned, "
          f"{n_geom - et2.model.ngeom} geoms added, none within 6 m of the pad")


if __name__ == "__main__":
    demo()
