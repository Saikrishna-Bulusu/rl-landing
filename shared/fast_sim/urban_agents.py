#!/usr/bin/env python3
"""
urban_agents.py -- the three kinds of decider the headroom question needs.

  Rule        what a careful practitioner would write: fly the averaged-fix aim,
              obey the CASA duties as reactive if-then logic, sidestep drones by
              closest-point-of-approach. No foresight, no knowledge of patterns.

  MPC(info)   ONE planner, three levels of information, so that differences
              between them are information and nothing else:
                "obs"    only what the sensors show now, extrapolated straight
                "prior"  + imagined intruders drawn from the world's pattern
                         (the corridor, the hub radials) where the sensors are
                         blind -- the knowledge a thousand past flights contain
                "full"   the true future of every intruder in this episode.
                         An oracle. It is handed the env on purpose.

READING THE GAPS
  obs  - rule   what better planning buys, with no extra information
  prior - obs   what KNOWING THE PATTERN buys  <- the thing RL could learn
  full - prior  what knowing THIS flight's future buys <- no amount of past
                flights teaches that; only better sensing does
In the random world the pattern is uninformative, so prior - obs should vanish
there. That is the control.
"""
import math

import torch

import urban_env as U


def _unit(v, eps=1e-6):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=eps)


class Rule:
    name = "rule"
    A_BRAKE, SEARCH_ALT, PAD_BLOCK_R, EVADE_R = 1.2, 35.0, 35.0, 12.0
    SEP_MARGIN, LOOK = 40.0, 10.0
    BIRD_MARGIN, BIRD_LOOK = 3.0, 4.0
    V_BLIND = 0.5          # committed descent onto a locked pad it cannot see, m/s (px4_mission)
    PEOPLE_MEM_S = 30.0    # someone seen near the pad is assumed there this long, unless seen gone
    SLOW_R, V_SLOW = 80.0, 5.0   # stage 3 only: terminal-area speed cap, m and m/s

    def reset(self, env):
        self.n, self.dev = env.n, env.dev
        self.t_search = torch.zeros(self.n, device=self.dev)
        self.t_people = torch.full((self.n,), -U.FAR, device=self.dev)
        self.survey_at, self.known_rel = env.survey_at, env.known_rel
        self.voxel = bool(env.p_new)          # stage 3: look ahead on the voxel map (see obstacles())
        # a local 3D planner on the map (env.local3d: one drone, e.g. PX4 in Gazebo): the rule flies its
        # path instead of the straight line; env.tops(xy) is the map as column tops (survey + voxel map)
        self.local = None
        if getattr(env, "local3d", False):
            import local_planner as LP
            self.local, self.tops = LP.LocalPlanner(), env.tops

    # ---- where to go, before any intruder is considered ---------------------------
    def guidance(self, o, land_when_clear=True):
        track = o["track"]
        bel = o["pad_rel_bel"]
        d_bel = torch.linalg.norm(bel, dim=-1)
        # COMMIT ON LOCK (Sai, 24 Sept): once its own marker has been read, the pad is where
        # that sighting put it. Losing sight -- fog, a dodge -- means steer back to it and keep
        # going down, slower; never fall back to the navigation guess and spiral away.
        locked = o["ever_seen"]
        # searching: at the believed pad, marker never read -> expanding spiral
        self.t_search = torch.where(~track & ~locked & (d_bel < 60.0), self.t_search + U.DT, torch.zeros_like(self.t_search))
        r = (1.5 * self.t_search).clamp(max=50.0); phi = 0.3 * self.t_search
        spiral = bel + r[:, None] * torch.stack([torch.cos(phi), torch.sin(phi)], -1)
        tgt = torch.where(track[:, None], o["pad_rel_vis"], torch.where(locked[:, None], o["pad_rel_fix"],
                          torch.where((d_bel < 5.0)[:, None] | (self.t_search > 0)[:, None], spiral, bel)))
        d = torch.linalg.norm(tgt, dim=-1)
        # Stage 3: an 8 m depth camera needs ~5.5 m to stop from 8 m/s plus the reaction, so in the
        # terminal area -- where a crane next door would be -- fly 5 m/s (stop + react ~4.8 m).
        cap = torch.where(d_bel < self.SLOW_R, self.V_SLOW, U.V_XY) if self.voxel else torch.full_like(d, U.V_XY)
        vxy = _unit(tgt) * torch.minimum(torch.sqrt(2 * self.A_BRAKE * d), cap)[:, None]

        z = o["pos_z"]
        # everyone counts: anyone seen within 35 m of the pad blocks the descent
        g = o["ground_rel"][..., :2]
        pad = o["pad_rel_vis"]
        near = (o["ground_seen"] & (torch.linalg.norm(g - pad[:, None], dim=-1) < self.PAD_BLOCK_R)).any(-1)
        # Fog hides people as it hides the marker. Someone seen near the pad is remembered until
        # the yard is seen clear without them (the marker in view = nothing in between). Without
        # this every patch that hid them looked like an empty yard: in fog 20% of flights came
        # within 30 m of a person (0% with it). Clear weather: people and marker show together.
        self.t_people = torch.where(near, torch.full_like(self.t_people, o["t"]),
                                    torch.where(o["seen"], torch.full_like(self.t_people, -U.FAR), self.t_people))
        block = near | (o["t"] - self.t_people < self.PEOPLE_MEM_S)
        over = (track | locked) & (d < 2.0)
        alt = torch.where(d_bel > 60.0, torch.full_like(z, U.CRUISE), torch.full_like(z, self.SEARCH_ALT))
        vz = (0.6 * (alt - z)).clamp(-U.VZ_DN, U.VZ_UP)
        down = torch.where(track, torch.where(z > 8.0, -2.5, -0.8), -self.V_BLIND) * torch.ones_like(z)
        vz_any = torch.where(over, down, vz)                    # would land whoever is there
        vz = torch.where(over & ~block, down, vz)
        vz = torch.where(over & block & (z < 32.0), torch.full_like(z, U.VZ_UP), vz)
        if self.local is not None:
            vxy, vz = self.follow(o, tgt, alt, over, vxy, vz)
        self.goal = torch.cat([tgt, torch.where(over, torch.zeros_like(z), alt)[:, None] - z[:, None]], -1)
        self.v_any = torch.cat([vxy, vz_any[:, None]], -1)
        if not land_when_clear:
            return self.v_any, block
        return torch.cat([vxy, vz[:, None]], -1), block

    def follow(self, o, tgt, alt, over, vxy, vz):
        """The rule's target at the rule's height, reached along the local planner's path: head for its carrot
        at the rule's speed. Over the pad the descent stays the rule's (vertical, on the marker)."""
        import numpy as np
        import local_planner as LP
        if bool(over[0]):
            self.local.path = None
            return vxy, vz
        me = np.r_[o["bel_xy"][0].cpu().numpy(), float(o["pos_z"][0])]
        path = self.local.plan(me, np.r_[me[:2] + tgt[0].cpu().numpy(), float(alt[0])], self.tops)
        if path is None:                      # boxed in: the rule's own line; obstacles() still stops it
            return vxy, vz
        d = self.local.carrot(me, path) - me
        h, sp = math.hypot(d[0], d[1]), float(torch.linalg.norm(vxy[0]))
        sp = min(sp, LP.V_SENSE)                            # never faster than the depth camera can stop for
        if self.local.clear_ahead < LP.C_CLEAR:             # threading past something on the map: slower still
            sp = min(sp, LP.V_NEAR)
        if h > 0.3:
            vxy = torch.tensor([[d[0] / h * sp, d[1] / h * sp]], device=vxy.device, dtype=vxy.dtype)
        vz = torch.tensor([float(np.clip(0.6 * d[2], -U.VZ_DN, U.VZ_UP))], device=vz.device, dtype=vz.dtype)
        return vxy, vz

    # ---- CASA: move away from a helicopter within 1.4 km of its landing site -----
    def helipad(self, o, v):
        hls = o["hls_rel_bel"]
        r = torch.linalg.norm(hls, dim=-1)
        duty = o["obliged"]
        inside = duty & (r <= U.HLS_R + 20.0)
        away = -_unit(hls)
        z = o["pos_z"]
        # Climb above the canopy BEFORE running. The escape used to go sideways from 15 m,
        # clear of every roof in the footprint map; on the LiDAR map every crash the rule
        # had (37 of 4,096 flights) was this escape, at 8 m/s and 16 m, into 20 m trees.
        high = z >= self.SEARCH_ALT
        flee = torch.cat([away * U.V_XY * high[:, None],
                          torch.where(high, torch.zeros_like(z), torch.full_like(z, U.VZ_UP))[:, None]], -1)
        v = torch.where(inside[:, None], flee, v)
        # outside but under duty: do not re-enter -- strip the inward component
        edge = duty & ~inside & (r <= U.HLS_R + 150.0)
        inward = (v[:, :2] * _unit(hls)).sum(-1).clamp(min=0.0)
        v2 = v.clone(); v2[:, :2] = v[:, :2] - inward[:, None] * _unit(hls)
        return torch.where(edge[:, None], v2, v)

    # ---- never over a person, and climb clear of anyone within 30 m ---------------
    def people(self, o, v):
        g = o["ground_rel"]
        dh = torch.linalg.norm(g[..., :2], dim=-1).masked_fill(~o["ground_seen"], U.FAR)
        dmin, j = dh.min(-1)
        near = dmin < self.EVADE_R
        away = -_unit(g[torch.arange(self.n, device=self.dev), j][:, :2])
        v = v.clone()
        v[:, :2] = torch.where(near[:, None], away * 3.0, v[:, :2])
        d3 = torch.linalg.norm(g, dim=-1).masked_fill(~o["ground_seen"], U.FAR).min(-1).values
        low = (d3 < U.PEOPLE_R + 3.0) & (o["pos_z"] < 32.0)
        floor = U.VZ_UP * low.float()               # climb if low near someone; never descend over them
        v[:, 2] = torch.where(near | low, torch.maximum(v[:, 2], floor), v[:, 2])
        return v

    # ---- sidestep other drones by closest point of approach ------------------------
    def drones(self, o, v):
        rel, dv, seen = o["drone_rel"], o["drone_vel"], o["drone_seen"]
        ang = torch.tensor([0, 30, -30, 60, -60, 90, -90], device=self.dev, dtype=torch.float32) * math.pi / 180
        c, s = torch.cos(ang), torch.sin(ang)
        cand = torch.stack([c[None] * v[:, None, 0] - s[None] * v[:, None, 1],
                            s[None] * v[:, None, 0] + c[None] * v[:, None, 1],
                            v[:, None, 2].expand(-1, len(ang))], -1)                   # (n,7,3)
        cand = torch.cat([cand, 0.3 * v[:, None], v[:, None] + torch.tensor([0, 0, 2.0], device=self.dev),
                          v[:, None] - torch.tensor([0, 0, 2.0], device=self.dev)], 1)  # (n,10,3)
        rv = dv[:, None] - cand[:, :, None]                                           # (n,C,D,3)
        tstar = (-(rel[:, None] * rv).sum(-1) / (rv * rv).sum(-1).clamp(min=1e-6)).clamp(0, self.LOOK)
        dcpa = torch.linalg.norm(rel[:, None] + rv * tstar[..., None], dim=-1)
        dcpa = dcpa.masked_fill(~seen[:, None], U.FAR).min(-1).values                 # (n,C)
        ok = dcpa >= self.SEP_MARGIN
        first = torch.where(ok.any(-1), ok.float().argmax(-1), dcpa.argmax(-1))
        return cand[torch.arange(self.n, device=self.dev), first]

    # ---- a diving bird: off its line at full speed, climbing (what worked for Lyons et al.) --
    # A careful practitioner checks the escape first: the side we are on, the other side, or
    # straight up -- the first that clears the bird, has no tree or roof in the next 2 s (the
    # survey map) and no one within 12 m of where it ends up. Dodging blind cost the rule 4.9%
    # crashes into trees and 4.1% flights over a person.
    def birds(self, o, v):
        rel, bv = o["bird_rel"], o["bird_vel"]
        u = _unit(bv[:, :2])
        side = -rel[:, :2] - (-rel[:, :2] * u).sum(-1, keepdim=True) * u        # the side we are on
        side = torch.where(torch.linalg.norm(side, dim=-1, keepdim=True) > 0.1, _unit(side),
                           torch.stack([-u[:, 1], u[:, 0]], -1))
        up = torch.full_like(v[:, :1], U.VZ_UP)
        esc = torch.stack([torch.cat([side * U.V_XY, up], -1), torch.cat([-side * U.V_XY, up], -1),
                           torch.cat([torch.zeros_like(side), up], -1)], 1)            # (n,3,3)
        cand = torch.cat([v[:, None], esc], 1)                                          # (n,4,3)
        rv = bv[:, None] - cand
        tstar = (-(rel[:, None] * rv).sum(-1) / (rv * rv).sum(-1).clamp(min=1e-6)).clamp(0, self.BIRD_LOOK)
        cpa = torch.linalg.norm(rel[:, None] + rv * tstar[..., None], dim=-1)            # (n,4)
        miss = cpa >= self.BIRD_MARGIN
        clear = miss.clone()
        me = torch.where(o["track"][:, None], o["pad_xy"] - o["pad_rel_vis"], o["bel_xy"])
        for t in (0.5, 1.0, 1.5, 2.0):
            clear &= self.map_at(me[:, None], cand[..., :2] * t) < (o["pos_z"][:, None] + cand[..., 2] * t - 1.0)
        g = o["ground_rel"][..., :2].masked_fill(~o["ground_seen"][..., None], U.FAR)   # (n,G,2)
        end = cand[..., :2] * 2.0                                                        # (n,4,2)
        clear &= (torch.linalg.norm(g[:, None] - end[:, :, None], dim=-1) >= self.EVADE_R).all(-1)
        threat = o["bird_seen"] & (torch.linalg.norm(bv, dim=-1) > 1.0) & ~miss[:, 0]
        pick = torch.where(clear[:, 1:].any(-1), clear[:, 1:].float().argmax(-1), cpa[:, 1:].argmax(-1))
        return torch.where(threat[:, None], esc[torch.arange(self.n, device=self.dev), pick], v)

    # ---- stage 3, continuous: brake and climb rather than fly into what the voxel map shows ----
    # Only where the world holds things the survey missed (env.p_new): elsewhere the survey is
    # complete and 50/35 m clear it, and the four-condition study was trained without this layer.
    def obstacles(self, o, v):
        me = torch.where(o["track"][:, None], o["pad_xy"] - o["pad_rel_vis"], o["bel_xy"])
        z, v0 = o["pos_z"], o["vel"]
        tc = torch.tensor([U.TAU_XY, U.TAU_XY, U.TAU_Z], device=v.device)

        def blocked(c):
            # The path the drone will ACTUALLY fly under command c -- velocity lags ~0.7 s, the
            # planner's own rollout model -- for 2 s, every 0.25 s (a 2 m column can't slip
            # between at 8 m/s). Only real obstacles, not bare ground, within 2 m of that path.
            # Checked as if the new direction took hold at once, a 45 deg swerve "cleared" a
            # column the curving real path still hit.
            b = torch.zeros_like(z, dtype=torch.bool)
            for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
                dp = c * t + (v0 - c) * tc * (1 - torch.exp(-t / tc))
                hgt = self.map_at(me, dp[:, :2])
                b |= (hgt > 0.3) & (hgt > (z + dp[:, 2]).clamp(min=0.0) - 2.0)
            return b
        # Steer round it, 45 then 90 deg either side; boxed in, stop and climb. Braking and
        # climbing alone stalled at a crane on the way in: the approach height kept pulling
        # the drone back down into it (98% timeouts).
        stop = torch.cat([torch.zeros_like(v[:, :2]), torch.full_like(z[:, None], U.VZ_UP)], -1)
        out, todo = v, blocked(v)
        for deg in (45.0, -45.0, 90.0, -90.0):
            c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            u = torch.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1], v[:, 2]], -1)
            ok = todo & ~blocked(u)
            out, todo = torch.where(ok[:, None], u, out), todo & ~ok
        return torch.where(todo[:, None], stop, out)

    def map_at(self, me, d):
        """The drone's map at displacement d from where it believes it is (me): the survey, absolute,
        inheriting the navigation error; in stage 3 also what its sensors have seen, relative."""
        h = self.survey_at(me + d)
        return torch.maximum(h, self.known_rel(d)) if self.voxel else h

    def safety(self, o, v):
        v = self.drones(o, self.people(o, self.helipad(o, v)))
        return self.birds(o, self.obstacles(o, v) if self.voxel else v)

    def act(self, o):
        v, _ = self.guidance(o)
        return self.safety(o, v)


class MPC(Rule):
    """
    Receding-horizon planner. Every 0.5 s it scores a handful of candidate
    velocity commands -- the rule's own action always among them -- by rolling
    each forward 20 s against predicted intruders, and flies the cheapest.

    The ONLY thing that differs between mpc_obs, mpc_prior and mpc_full is what
    the intruder predictions are built from. Same candidates, same rollout, same
    costs. That is what lets their differences be read as information.
    """
    H, REPLAN, PROG_S, GAMMA = 20, 2, 5, 0.95
    W = dict(people=30.0, over=30.0, helipad=30.0, sep=60.0, hit=1e4,
             nmac=1e4, hhit=1e5, bld=1e5, ceil=60.0, late=50.0, bird=1e5)
    # bird = bld: a strike loses the airframe like a roof does. At the drones' 1e4 a bird
    # hit 1 s away weighed less than a roof the 20 s constant-command rollout would reach
    # in 15 s -- which the drone would re-plan around long before -- so it declined dodges.
    # Plan against the SAME margins the rule uses (35 m people, 12 m overhead,
    # 40 m drones). Without them mpc_obs landed beside anyone standing 31 m out --
    # legal until they take a step -- and scored 10% people violations to the
    # rule's 0.1%. Margins are how any planner copes with its own predictions
    # being wrong; all three information levels get identical ones.
    M_PEOPLE, M_OVER, M_SEP = U.PEOPLE_R + 5.0, U.OVER_R + 2.0, U.DRONE_SEP + 10.0
    M_BIRD = U.HIT_BIRD + 1.5

    def __init__(self, info="obs", mode="shield", m=8, seed=7):
        """
        mode="shield": fly the RULE's action unless foresight says it leads into a
        violation within the horizon; only then take the least-risky alternative.
        Lexicographic -- risk first, then staying close to the rule -- so there is
        no risk-versus-progress weight to tune. It keeps the rule's timing and
        changes only what the extra information changes, which makes it the clean
        instrument for "what does knowing more buy".
        mode="plan": minimise one weighted cost (progress, deadline, risk). Kept
        because it CAN beat the rule on timing, not only on safety; its weights are
        a judgement call, and it is reported alongside, never instead.
        mode="safeplan": risk first, then time. Never accepts a predicted violation
        when a safe candidate exists, and among the safe ones picks the fastest --
        not the rule's. The strongest oracle here: the shield's safety with the
        planner's freedom over timing.
        """
        assert info in ("obs", "prior", "full") and mode in ("shield", "plan", "safeplan")
        self.info, self.mode, self.m, self.seed = info, mode, (m if info == "prior" else 0), seed
        self.name = f"{mode}_{info}"

    def reset(self, env):
        super().reset(env)
        self.env, self.k = env, 0        # "full" reads the future from env; "prior" only its prior
        self.cmd = torch.zeros(env.n, 3, device=env.dev)
        self.tau = torch.arange(1, self.H + 1, device=env.dev, dtype=torch.float32)
        self.tau0 = torch.arange(0, self.H + 1, device=env.dev, dtype=torch.float32)
        self.disc = self.GAMMA ** self.tau
        if self.info == "prior":         # imagined intruders: own generator, never the env's
            g = torch.Generator(env.dev).manual_seed(self.seed)
            self.imag = env.sample_intruders(env.n * self.m, env.pad_idx.repeat_interleave(self.m), g=g)

    # ---- candidates --------------------------------------------------------------
    def candidates(self, o):
        v_nom, self.block = self.guidance(o)
        v_rule = self.safety(o, v_nom)
        def rot(v, deg):
            c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            return torch.stack([c * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c * v[:, 1], v[:, 2]], -1)
        z = torch.zeros_like(v_nom)
        flee = torch.cat([-_unit(o["hls_rel_bel"]) * U.V_XY, z[:, :1]], -1)
        hold = v_nom.clone(); hold[:, 2] = 0.0
        climb = v_nom.clone(); climb[:, 2] = U.VZ_UP
        half = v_nom.clone(); half[:, :2] *= 0.5
        up = z.clone(); up[:, 2] = U.VZ_UP
        C = torch.stack([v_rule, self.v_any, v_nom, rot(v_nom, 45), rot(v_nom, -45), rot(v_nom, 90),
                         rot(v_nom, -90), rot(v_nom, 180), half, z, up, flee, hold, climb], 1)
        sp = torch.hypot(C[..., 0], C[..., 1]).clamp(min=1e-6)
        C[..., :2] *= (U.V_XY / sp).clamp(max=1.0)[..., None]
        C[..., 2] = C[..., 2].clamp(-U.VZ_DN, U.VZ_UP)
        return C

    def rollout(self, o, C):
        tc = torch.tensor([U.TAU_XY, U.TAU_XY, U.TAU_Z], device=C.device)
        tau = self.tau[:, None]
        e = torch.exp(-tau / tc)                                       # (H,3)
        v0, c = o["vel"][:, None, None], C[:, :, None]
        dp = c * tau + (v0 - c) * tc * (1 - e)                          # (n,P,H,3)
        vel = c + (v0 - c) * e
        z = (o["pos_z"][:, None, None] + dp[..., 2]).clamp(min=0.0)
        return dp, vel, z

    # ---- predictions: the only thing that differs between the three ------------
    def predict(self, o):
        env, n, H = self.env, self.env.n, self.H
        T = o["t"] + self.tau
        if self.info == "full":                   # the oracle: this episode's true future
            Tn = T[None].expand(n, H)
            hp, _, hair = env.heli_at(env.heli, Tn)
            dp_, _, _ = env.drones_at(env.dr, Tn)
            gp, _ = env.ground_at(env.gr, Tn)
            me = env.pos
            b = env.bird                          # its current dive, seen at any range (not its next one)
            bird = torch.where(b["on"][:, None, None], (b["pos"] - me)[:, None] + b["vel"][:, None] * self.tau0[None, :, None],
                               torch.full((n, H + 1, 3), U.FAR, device=me.device))
            parts = [dict(w=1.0, heli=(hp - me[:, None])[:, None], air=hair[:, None],
                          drones=(dp_ - me[:, None, None])[:, None], ground=(gp - me[:, None, None])[:, None],
                          bird=bird[:, None], duty_end=(env.heli["t_arr"] + 30.0) * env.heli["on"])]
            return parts
        tau = self.tau[None, :, None]
        hr = o["heli_rel"][:, None] + o["heli_vel"][:, None] * tau          # straight-line extrapolation
        air = o["heli_seen"][:, None] & (hr[..., 2] + o["pos_z"][:, None] > 0.5)
        dr = o["drone_rel"][:, None] + o["drone_vel"][:, None] * tau[..., None]
        gr = o["ground_rel"][:, None] + o["ground_vel"][:, None] * tau[..., None]
        bird = o["bird_rel"][:, None] + o["bird_vel"][:, None] * self.tau0[None, :, None]    # (n,H+1,3)
        parts = [dict(w=1.0, heli=hr[:, None], air=air[:, None], drones=dr[:, None], ground=gr[:, None],
                      bird=bird[:, None], duty_end=None)]
        if self.info == "prior":
            m, im = self.m, self.imag
            Tm = T[None].expand(n * m, H)
            bel = torch.cat([o["bel_xy"], o["pos_z"][:, None]], -1).repeat_interleave(m, 0)
            now = torch.full((n * m, 1), o["t"], device=bel.device)
            ihp, _, ihair = env.heli_at(im[0], Tm)
            h0, _, a0 = env.heli_at(im[0], now)
            drop = ((torch.linalg.norm(h0[:, 0] - bel, dim=-1) <= env.d_heli) & a0[:, 0]) \
                | (o["heli_seen"] | o["aware"]).repeat_interleave(m)          # only one arrival exists
            ihair = ihair & ~drop[:, None]
            idp, _, _ = env.drones_at(im[1], Tm)
            d0, _, _ = env.drones_at(im[1], now)
            dvis = torch.linalg.norm(d0[:, 0] - bel[:, None], dim=-1) <= env.d_drone   # would be seen now
            idr = torch.where(dvis[:, None, :, None], torch.full_like(idp, U.FAR), idp - bel[:, None, None])
            igp, _ = env.ground_at(im[2], Tm)
            g0, _ = env.ground_at(im[2], now)
            gvis = torch.linalg.norm(g0[:, 0, :, :2] - bel[:, None, :2], dim=-1) <= bel[:, 2:3].clamp(min=1.0) * U.FOV_TAN
            igr = torch.where(gvis[:, None, :, None], torch.full_like(igp, U.FAR), igp - bel[:, None, None])
            sh = lambda x: x.reshape(n, m, *x.shape[1:])
            parts.append(dict(w=1.0 / m, heli=sh(ihp - bel[:, None]), air=sh(ihair), drones=sh(idr),
                              ground=sh(igr), bird=torch.full((n, m, H + 1, 3), U.FAR, device=bel.device),
                              duty_end=None))
        return parts

    # ---- cost -----------------------------------------------------------------------
    def cost(self, o, C):
        dp, vel, z = self.rollout(o, C)                                  # (n,P,H,3)
        n, P = C.shape[:2]
        W = self.W
        # Nothing counts after touchdown: the mission is over and the motors are
        # off. Scoring the rest of the window made a landing plan look like a drone
        # parked beside the street while a car went past, so the perfect-foresight
        # shield refused to land and scored BELOW the rule it was meant to bound.
        airborne = torch.cummin((z > 0.3).int(), -1).values.float()         # (n,P,H)
        disc = self.disc * airborne
        total = torch.zeros(n, P, device=C.device)
        # Progress: how close each plan gets to the goal within PROG_S seconds, with
        # altitude clamped at the ground. The first version scored distance AT
        # t = 5 s on an unclamped rollout, so a descent "arrived" 2 m below the
        # ground, overshot the goal, and scored worse than hovering: the planner
        # sat 1.9 m above the pad until timeout, 256 episodes out of 256.
        g = self.goal
        dpe = torch.cat([dp[..., :2], (z - o["pos_z"][:, None, None])[..., None]], -1)
        near = torch.linalg.norm(g[:, None, None] - dpe[:, :, :self.PROG_S], dim=-1).min(-1).values
        prog = torch.linalg.norm(g, dim=-1)[:, None] - near
        total -= prog
        # The deadline. Under "clean delivery" a timeout fails the mission exactly
        # as completely as a violation does, but a per-second violation price with
        # no clock made the oracle wait out tiny risks until the mission expired.
        # Estimate the time still needed after each plan and charge any overrun.
        left = torch.linalg.norm(g[:, None, :2] - dpe[:, :, self.PROG_S - 1, :2], dim=-1) / 6.0 \
            + (g[:, None, 2] - dpe[:, :, self.PROG_S - 1, 2]).abs() / 2.0 + 10.0
        over_t = (o["t"] + self.PROG_S + left - U.T_MAX).clamp(min=0.0)
        total += W["late"] * over_t
        base = total.clone()
        self.base = base
        # static: buildings on the map, looked up where the drone BELIEVES it is.
        # While the marker is tracked that belief is the marker fix (the pad is a
        # surveyed point, so seeing it IS a position fix); otherwise the GeoAnchor
        # fix, ~9 m off. Using the GeoAnchor fix all the way down made every final
        # descent look like flying into the neighbour's roof, and the first version
        # of this planner delivered 0% because of it.
        me = torch.where(o["track"][:, None], o["pad_xy"] - o["pad_rel_vis"], o["bel_xy"])
        xy = me[:, None, None] + dp[..., :2]
        # the voxel map (survey + what the sensors have seen); only the oracle is handed the true world
        if self.info == "full":                    # the oracle: the true world, where it truly is
            me = self.env.pos[:, :2] if self.env.p_new else me
            hmap = lambda d: self.env.height_at(me.reshape(n, *([1] * (d.dim() - 2)), 2) + d)
        else:
            hmap = lambda d: self.map_at(me.reshape(n, *([1] * (d.dim() - 2)), 2), d)
        total += W["bld"] * ((hmap(dp[..., :2]) > z).float() * disc).sum(-1)
        if self.env.p_new:   # stage 3's thin columns fit between the 1 s samples: every 0.25 s for 3 s
            tc = torch.tensor([U.TAU_XY, U.TAU_XY, U.TAU_Z], device=C.device)
            tf = 0.25 * torch.arange(1, 13, device=C.device, dtype=torch.float32)[:, None]
            cc, v0 = C[:, :, None], o["vel"][:, None, None]
            dpf = cc * tf + (v0 - cc) * tc * (1 - torch.exp(-tf / tc))
            zf = (o["pos_z"][:, None, None] + dpf[..., 2]).clamp(min=0.0)
            total += W["bld"] * ((hmap(dpf[..., :2]) > zf) & (zf > 0.3)).float().sum(-1)
        total += W["ceil"] * ((z > U.CEIL).float() * disc).sum(-1)
        hls = o["hls_rel_bel"][:, None, None] - dp[..., :2]              # (n,P,H,2)
        r = torch.linalg.norm(hls, dim=-1)
        away = (-(hls / r[..., None].clamp(min=1.0)) * vel[..., :2]).sum(-1) >= 1.0
        inside = r <= U.HLS_R
        for part in self.predict(o):
            w = part["w"]
            hr = part["heli"][:, :, None] - dp[:, None]                   # (n,S,P,H,3)
            d3 = torch.linalg.norm(hr, dim=-1); air = part["air"][:, :, None]
            dh, dv_ = torch.linalg.norm(hr[..., :2], dim=-1), hr[..., 2].abs()
            nmac = (dh < U.NMAC_H) & (dv_ < U.NMAC_V) & air
            ds = disc[:, None]                                                 # (n,1,P,H)
            total += w * (W["nmac"] * (nmac.float() * ds).sum(-1)
                          + W["hhit"] * (((d3 < U.HIT_HELI) & air).float() * ds).sum(-1)).sum(1)
            newly = (d3 <= self.env.d_heli) & air
            aware = o["aware"][:, None, None, None] | (torch.cummax(newly.int(), -1).values > 0)
            if part["duty_end"] is not None:
                on = (o["t"] + self.tau)[None, None, None] < part["duty_end"][:, None, None, None]
            else:
                on = air | o["obliged"][:, None, None, None]
            duty = aware & on & inside[:, None] & ~away[:, None]
            total += w * (W["helipad"] * (duty.float() * ds).sum(-1)).sum(1)
            dd = torch.linalg.norm(part["drones"][:, :, None] - dp[:, None, :, :, None], dim=-1)   # (n,S,P,H,D)
            total += w * (W["sep"] * ((dd < self.M_SEP).any(-1).float() * ds).sum(-1)
                          + W["hit"] * ((dd < U.HIT_DRONE).any(-1).float() * ds).sum(-1)).sum(1)
            # a bird: closest approach between the 1 s samples -- a 15 m/s dive fits between two
            rb = part["bird"][:, :, None] - torch.cat([torch.zeros_like(dp[:, :, :1]), dp], 2)[:, None]  # (n,S,P,H+1,3)
            r0, dr_ = rb[..., :-1, :], rb[..., 1:, :] - rb[..., :-1, :]
            sb = (-(r0 * dr_).sum(-1) / (dr_ * dr_).sum(-1).clamp(min=1e-9)).clamp(0, 1)
            db = torch.linalg.norm(r0 + sb[..., None] * dr_, dim=-1)                                 # (n,S,P,H)
            total += w * (W["bird"] * ((db < self.M_BIRD).float() * ds).sum(-1)).sum(1)
            gd = part["ground"][:, :, None] - dp[:, None, :, :, None]                            # (n,S,P,H,G,3)
            g3 = torch.linalg.norm(gd, dim=-1); gh = torch.linalg.norm(gd[..., :2], dim=-1)
            above = (z[:, None, :, :, None] > 1.7)
            total += w * (W["people"] * ((g3 < self.M_PEOPLE).any(-1).float() * ds).sum(-1)
                          + W["over"] * (((gh < self.M_OVER) & above).any(-1).float() * ds).sum(-1)).sum(1)
        self.risk = total - base                   # violation terms only
        self.prog = prog
        total[:, 0] -= 1e-3                        # ties go to the rule's own action
        return total

    def act(self, o):
        if self.k % self.REPLAN == 0:
            C = self.candidates(o)
            total = self.cost(o, C)
            if self.mode == "shield":
                # Any predicted violation-second (>= 30 x 0.95^20 = 10.7) outweighs
                # the largest possible progress loss (50 m x 0.15 = 7.5): lexicographic.
                lost = (self.prog[:, :1] - self.prog).clamp(min=0.0)
                total = self.risk + 0.15 * lost
                total[:, 0] -= 1e-3
            elif self.mode == "safeplan":
                total = 1e3 * self.risk + self.base
                total[:, 0] -= 1e-3
            best = total.argmin(-1)
            self.cmd = C[torch.arange(C.shape[0], device=C.device), best]
            self.choice, self.C = best, C          # kept so a student can be taught this pick
            cmd = self.cmd
        else:
            self.guidance(o)                       # keep the search state ticking
            # Between decisions a diving bird still gets the rule's checked dodge, as the rule
            # reacts every step. Without it the planner, deciding every 0.5 s, was struck by a
            # magpie on 3.1-4.3% of flights against the rule's 1.9%. No bird seen: a no-op.
            cmd = self.birds(o, self.cmd)
        self.k += 1
        return cmd


def evaluate(world, agent, n=2048, seed=100, **kw):
    env = U.UrbanEnv(n, world, seed=seed, **kw)
    return U.run(env, agent)


def table(rows):
    f = lambda x: 100 * x.float().mean().item()
    print(f"  {'agent':<16}{'clean':>7}{'deliv':>7}{'crash':>7}{'NMAC':>6}{'ceil':>6}{'people':>8}{'vehicle':>8}"
          f"{'over':>6}{'helipad':>8}{'drone':>7}{'med t':>7}")
    for name, s in rows:
        v = s["viol_s"] > 0
        med = s["t_done"][s["clean"]].median().item() if s["clean"].any() else float("nan")
        print(f"  {name:<16}{f(s['clean']):6.1f}%{f(s['delivered']):6.1f}%{f(s['crash']):6.1f}%"
              f"{f(s['nmac']):5.1f}%{f(v[:, 0]):5.1f}%{f(v[:, 1]):7.1f}%{f(v[:, 2]):7.1f}%{f(v[:, 3]):5.1f}%"
              f"{f(v[:, 4]):7.1f}%{f(v[:, 5]):6.1f}%{med:6.0f}s")


if __name__ == "__main__":
    for world in ("corridor", "random"):
        print(f"\n{world}")
        table([(a.name if hasattr(a, "name") else "straight", evaluate(world, a))
               for a in (Rule(), MPC("obs"), MPC("full"), MPC("obs", "safeplan"), MPC("full", "safeplan"))])
