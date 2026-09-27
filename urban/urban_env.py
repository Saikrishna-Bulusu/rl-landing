#!/usr/bin/env python3
"""
urban_env.py -- the last ~1 km of a delivery in Logan, QLD, simulated on the GPU.

Thousands of deliveries advance in lock-step as tensors. Nothing here is a
policy: agents read obs() and hand step() a velocity command, as PX4 offboard
does.

WHERE EVERY NUMBER COMES FROM
  measured   TAU_XY/TAU_Z   PX4 x500 velocity-loop lag (measure_tau.py)
  measured   DET_X0/DET_S   logistic fit to detect_curve_60m.json, flown on PX4
  measured   error model    GeoAnchor fixes replayed from sigma_pool.npz: sigma
                            decile held per episode, magnitude through an AR(1)
                            Gaussian copula (rho from the pool), bearing wandering
                            1.665 rad/fix (error-direction-wanders memory)
  CASA       120 m ceiling; >= 30 m from people and never over a person
             ("everyone counts": recipient and car occupants too); within
             1.4 km of a helicopter landing site, move away from a helicopter
             that is nearby or landing
  ASTM F3442 NMAC 152 m horizontal / 30 m vertical; well clear 610 m / 76 m
  assumed    people near the pad leave at a time uniform in 30-240 s
  assumed    30 m drone-to-drone separation (no CASA number exists); 10 m
             "over a person" radius; 3 s response grace before a condition
             someone ELSE caused counts as a violation; detection ranges
             1000 m (helicopter) and 200 m (drones); people seen when inside
             the camera footprint

EVERY EPISODE IS AN ENCOUNTER SET, NOT A DAY IN LOGAN. A helicopter arrival
happens with probability P_HELI (0.5), far above the real rate at a regional
hospital. That is how detect-and-avoid systems are evaluated: conditional on an
encounter, reported separately. Weight by real encounter frequency afterwards.

THE TWO WORLDS differ only in where intruders come from:
  corridor  the air ambulance always arrives from the north-west (the Brisbane
            direction, +/- 8 deg); other drones fly radials to and from the hub
  random    arrival bearing uniform; other drones fly between random points
If a learned policy can beat a reactive rule, it should be here that it does:
the corridor is information a thousand past flights contain and this flight's
sensors do not.
"""
import math
import os

import numpy as np
import torch

HERE = __file__.rsplit("/", 1)[0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- measured ---------------------------------------------------------------
TAU_XY, TAU_Z = 0.69, 0.23
V_XY, VZ_UP, VZ_DN = 8.0, 3.0, 2.5
F_PX, FOV_TAN = 539.0, math.tan(math.radians(99.7) / 2)
PAD_M = 1.80
DET_X0, DET_S = 23.77, 1.83
BEARING_STEP = math.sqrt(-2.0 * math.log(0.25))
# ---- CASA / ASTM --------------------------------------------------------------
CEIL, PEOPLE_R, HLS_R = 120.0, 30.0, 1400.0
NMAC_H, NMAC_V, WC_H, WC_V = 152.4, 30.5, 609.6, 76.2
# ---- assumed ------------------------------------------------------------------
OVER_R, DRONE_SEP, GRACE = 10.0, 30.0, 3.0
D_HELI, D_DRONE = 1000.0, 200.0
HIT_DRONE, HIT_HELI = 3.0, 15.0
CRUISE = 50.0
DT, T_MAX, FIX_DT = 0.25, 300.0, 0.5
GPS_BIAS, GPS_JITTER = 1.2, 0.3   # m per axis: consumer GNSS, a per-flight offset plus per-fix noise
LOST_S = 3.0      # px4_mission: a sighting is trusted this long after the marker drops out
P_HELI, P_PEOPLE = 0.5, 0.5
# ---- birds: off unless p_swoop > 0 -------------------------------------------------
# A territorial magpie in breeding season (Aug-Nov in SE Queensland). Measured elsewhere:
#   defends ~80-100 m around its nest (D. Jones, Griffith Uni); retreats once a drone is
#   50-100 m away, and flying off at full speed, climbing, avoided contact (Lyons et al.
#   2018, Aust. Field Ornithology 35:51); 11 of 59 Brisbane breeding pairs had an
#   aggressive male (Jones & Thomas 1999, Wildl. Soc. Bull. 27:473); Wing paused Canberra deliveries in 2021 over
#   ravens swooping its drones near a nest.
# ASSUMED: the height it defends, dive speed (~50 km/h level flight, faster diving), the
#   gap between swoops, how close a pass is meant to come, how far the cameras see a bird.
R_TERR, Z_TERR, V_SWOOP, REARM = 100.0, 40.0, (12.0, 18.0), (4.0, 10.0)
P_SWOOP = 0.5      # --birds: an encounter set, like P_HELI, not the real share of pads near a swooper
# The HUNTER magpie (magpie="hunter"; Sai, 24 Sept: real magpies come from behind and keep
# adjusting -- a bird that commits to one straight line tells the drone its whole path at the first
# sighting). While the drone is in its territory it flies to a spot STAGE_BACK behind (up to
# STAGE_ANG either side of the tail) and STAGE_UP above it, drawn anew for every dive (at V_STAGE,
# keeping 8 m clear), launches only from the rear sector (more than BEHIND
# off the drone's heading) and within DIVE_R, steers at where the drone will be -- turning at up to
# A_TURN -- and stops correcting T_COMMIT before it arrives (its reaction time); gives up after
# DIVE_S or when the drone leaves. Same territory, dive speeds, pass offsets and rearm times as the
# straight diver, so a drone that does not dodge is still touched on ~12% of dives. One that dodges
# early gets chased; one that dodges late can beat it. ASSUMED, all of it: to be sensitivity-tested.
BEHIND, STAGE_ANG, STAGE_BACK, STAGE_UP, V_STAGE = math.radians(110.0), math.radians(60.0), (12.0, 25.0), (2.0, 8.0), 9.0
A_TURN, T_COMMIT, DIVE_R, DIVE_S = 20.0, 0.3, (8.0, 35.0), 6.0
# sensitivity runs only: HUNTER_A_TURN (m/s^2) and HUNTER_T_COMMIT (s) override the two guesses that
# decide whether dodging can work at all
A_TURN = float(os.environ.get("HUNTER_A_TURN", A_TURN))
T_COMMIT = float(os.environ.get("HUNTER_T_COMMIT", T_COMMIT))
# ---- fog: off unless p_fog > 0 ---------------------------------------------------------
# MEASURED in Gazebo (24 Sept, urban/gz_weather.py mist + fog skies, 20 flights pooled): the
# share of frames the marker was read while well inside the view, fog / calm, by height --
# the higher the drone, the more fog the line of sight crosses. Patches drift, so a hidden
# pad stays hidden: gaps ran to 7-10 s (90th pct 1.4-2.7 s) against 0.6 s in calm. The same
# patch hides people and cars under it: they are seen by the same downward camera.
FOG_Z, FOG_SEE = (5.0, 15.0, 25.0, 35.0, 46.0), (1.0, 0.92, 0.77, 0.48, 0.36)
FOG_PATCH_S = 2.0
P_FOG = 1.0        # --fog: every flight in fog, a foggy morning (birds: half the pads)
# ---- the unmapped (stage 3): what the 2023 survey does not show -- off unless p_new > 0 ----
# Sheds, shade sails, a caravan, a new tree: 1..N_NEW boxes near the pad, their nearest point
# outside its 2.5 m clear circle. ASSUMED sizes (the survey is 2023, the flights 2026).
N_NEW, P_NEW = 5, 1.0      # slots 0-3 near the pad; slot 4 the tall one below
NEW_SIDE, NEW_H, NEW_R = (2.0, 6.0), (2.5, 10.0), 15.0
# How the drone learns of them. The voxel map is the survey plus whatever these have seen:
#   forward depth camera: OAK-D Lite, as on PX4's x500_depth -- 73 x 58 deg, ideal range to 8 m
#   downward rangefinder: a 15 deg cone under the drone, 40 m
# ASSUMED: fog does not degrade them (stereo depth does degrade in fog; a LiDAR less so).
# ...and on half the flights something TALL and temporary on the way in: a mobile crane lifting a
# pool next door, a boom lift, a tower crane on a build -- 25-60 m, 15-60 m short of the pad,
# within 15 deg of the approach bearing. What a survey can never show, and what a drone at
# 35-50 m can fly into. ASSUMED: the encounter share is an encounter set, like P_HELI.
P_TALL, TALL_H, TALL_R = 0.5, (25.0, 60.0), (15.0, 60.0)
DEPTH_R, DEPTH_HFOV, DEPTH_VFOV = 8.0, math.radians(73.0), math.radians(58.0)
DOWN_TAN, DOWN_R = math.tan(math.radians(15.0)), 40.0
HIT_BIRD, D_BIRD = 0.8, 40.0   # contact: x500 half-span + props ~0.4 m, magpie half-wingspan ~0.4 m
# How close a dive means to pass: uniform in [0, MISS], so 12% of swoops at a target that
# does not dodge make contact -- the injury share of swoops reported to Magpie Alert
# (189 of 1,570 by mid-Sept 2019, reported by Newsweek). People flinch, so this is if
# anything generous to the bird's aim.
MISS = HIT_BIRD / 0.12
N_DRONE, N_PEOPLE, N_CAR, W = 6, 2, 6, 12
GLIDE = math.tan(math.radians(8.0))   # helicopter approach path
CORRIDOR = math.radians(135.0)          # arrivals come FROM the north-west
FAR = 1e6

OUT = {0: "running", 1: "delivered", 2: "crash_obstacle", 3: "crash_drone",
       4: "crash_heli", 5: "landed_off_pad", 6: "timeout", 7: "crash_bird", 8: "crash_unmapped"}
VIOL = ("ceiling", "people", "vehicle", "over", "helipad", "drone_sep")   # people/vehicle split: a car on
# the street and a person under the drone are different findings, even though CASA counts both.


def _interp(x, xp, fp):
    """np.interp on a tensor, clamped at the ends."""
    xp_t, fp_t = torch.tensor(xp, device=x.device), torch.tensor(fp, device=x.device)
    xc = x.clamp(xp[0], xp[-1])
    i = torch.bucketize(xc, xp_t).clamp(1, len(xp) - 1)
    return fp_t[i - 1] + (fp_t[i] - fp_t[i - 1]) * (xc - xp_t[i - 1]) / (xp_t[i] - xp_t[i - 1])


def crashed(outcome):
    """Obstacle (mapped or not), drone, helicopter or bird: every way the airframe is lost."""
    return ((outcome >= 2) & (outcome <= 4)) | (outcome == 7) | (outcome == 8)


def _load_map(path):
    m = np.load(path)
    pads = m["pads"].astype(np.float32)
    pts, off = m["road_pts"], m["road_off"]
    # Per pad: the nearest street, resampled to W points by arc length, so a
    # passing car is a point moving along it.
    lines = [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    dist = np.stack([np.hypot(*(l[:, None] - pads[None]).transpose(2, 0, 1)).min(0) for l in lines])
    cars = np.zeros((len(pads), W, 2), np.float32)
    for k, li in enumerate(dist.argmin(0)):
        best = lines[li]
        seg = np.hypot(*np.diff(best, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        q = np.linspace(0, s[-1], W)
        cars[k] = np.c_[np.interp(q, s, best[:, 0]), np.interp(q, s, best[:, 1])]
    return m, pads, cars


class UrbanEnv:
    def __init__(self, n, world="corridor", seed=0, device=DEV,
                 map_path=f"{HERE}/logan_map.npz", pool=f"{HERE}/../sigma_pool.npz",
                 p_heli=P_HELI, p_people=P_PEOPLE, d_heli=D_HELI, d_drone=D_DRONE,
                 leave=(30.0, 240.0), nav="geoanchor", p_swoop=0.0, p_fog=0.0, p_new=0.0,
                 bird_sense="omni", magpie="straight"):
        assert world in ("corridor", "random")
        # nav: "geoanchor" replays GeoAnchor's measured error (~9-15 m); "gps" is GNSS-grade:
        # a per-flight offset of GPS_BIAS m per axis (|err| ~1.4 m typical) + GPS_JITTER per fix.
        # GPS draws come from their OWN generator, so every other random thing -- intruders,
        # people, marker detection -- matches the GeoAnchor run of the same seed draw for draw.
        assert nav in ("geoanchor", "gps")
        self.nav = nav
        self.g_nav = torch.Generator(device).manual_seed(seed + 7919)
        # Birds draw from their own stream too: with p_swoop=0 every episode is the one
        # it was before birds existed, and with birds on, everything else still pairs.
        self.p_swoop, self.miss, self.p_fog = p_swoop, MISS, p_fog
        self.g_fog = torch.Generator(device).manual_seed(seed + 32452843)   # own stream, as birds
        self.p_new, self.g_new = p_new, torch.Generator(device).manual_seed(seed + 49979687)
        # How the drone knows of the magpie. "omni": within D_BIRD from ANY direction -- an all-round
        # sensor it does not carry (Sai, 24 Sept: magpies attack from behind). "cameras": only what
        # it has -- the forward depth camera's cone about its heading and the landing camera's cone
        # below it, same D_BIRD range, so the comparison is about coverage alone. "rig": a six-camera
        # 360-degree rig (gz_rig.py) seen through its detector and a tracker (rig_sense.py) -- the
        # chance of a detection, the blind spots and the errors all measured in Gazebo, the bird
        # reported only once tracked, its distance from its size in the image (Sai, 24 Sept).
        assert bird_sense in ("omni", "cameras", "rig")
        assert magpie in ("straight", "hunter")
        self.bird_sense, self.magpie = bird_sense, magpie
        if bird_sense == "rig":
            import rig_sense
            self.rig = rig_sense.Rig(n, device, seed)
        self.g_bird = torch.Generator(device).manual_seed(seed + 15485863)
        self.n, self.world, self.dev = n, world, device
        self.p_heli, self.p_people, self.d_heli, self.d_drone = p_heli, p_people, d_heli, d_drone
        self.leave = leave          # people near the pad leave at a time uniform in this range (s)
        self.g = torch.Generator(device).manual_seed(seed)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        m, pads, cars = _load_map(map_path)
        self.H = t(m["H"].astype(np.float32))
        self.half, self.res = float(m["half"]), float(m["res"])
        self.pads, self.car_paths = t(pads), t(cars)
        self.hub = t(m["hub"])
        d = np.load(pool)                  # numeric arrays only; no pickle needed
        sig, err = np.asarray(d["sigma"], float), np.asarray(d["err"], float)
        bins = np.array_split(np.argsort(sig), 10)
        self.qtab = t(np.stack([np.quantile(err[b], np.linspace(0, 1, 256)) for b in bins]))
        self.sig_med = t([float(np.median(sig[b])) for b in bins])
        self.rho = float(d["rho"]) if "rho" in d.files else 0.707

    # ---- random draws on the env's own generator ---------------------------
    def _u(self, *s):
        return torch.rand(*s, generator=self.g, device=self.dev)

    def _nrm(self, *s):
        return torch.randn(*s, generator=self.g, device=self.dev)

    # ---- intruders: analytic in time, so an oracle can ask about the future --
    def sample_intruders(self, n, pad_idx, g=None):
        """
        One draw of every intruder for n episodes. Also used to IMAGINE futures.

        An agent that imagines must pass its own generator. Drawing from the
        env's stream would shift every later draw -- fix noise, marker decodes
        -- so each agent would face a different episode and the pairing that
        every comparison here rests on would silently break.
        """
        g = self.g if g is None else g
        u = lambda *s: torch.rand(*s, generator=g, device=self.dev)
        nrm = lambda *s: torch.randn(*s, generator=g, device=self.dev)
        heli = {}
        heli["on"] = u(n) < self.p_heli
        th = (CORRIDOR + math.radians(8.0) * nrm(n)) if self.world == "corridor" \
            else 2 * math.pi * u(n)
        heli["dir"] = torch.stack([torch.cos(th), torch.sin(th)], -1)
        heli["v"] = 15.0 + 10.0 * u(n)                      # Benali & Hamoud's 15-25 m/s
        heli["t_arr"] = 20.0 + 260.0 * u(n)

        k = torch.randint(0, N_DRONE + 1, (n,), generator=g, device=self.dev)
        used = torch.arange(N_DRONE, device=self.dev)[None] < k[:, None]
        if self.world == "corridor":            # radials to / from the hub
            far = self.pads[torch.randint(0, len(self.pads), (n, N_DRONE), generator=g,
                                          device=self.dev)]
            hub = self.hub.expand(n, N_DRONE, 2)
            out = u(n, N_DRONE) < 0.5
            A = torch.where(out[..., None], hub, far); B = torch.where(out[..., None], far, hub)
        else:
            A = (u(n, N_DRONE, 2) - 0.5) * 3600.0; B = (u(n, N_DRONE, 2) - 0.5) * 3600.0
        dr = {"used": used, "A": A, "B": B, "alt": 45.0 + 10.0 * u(n, N_DRONE),
              "v": 8.0 + 2.0 * u(n, N_DRONE), "t0": -120.0 + 360.0 * u(n, N_DRONE)}

        # Ground agents: people in the yard, cars on the nearest street.
        pad = self.pads[pad_idx]
        G = N_PEOPLE + N_CAR
        wp = torch.full((n, G, W, 2), FAR, device=self.dev); tw = torch.zeros(n, G, W, device=self.dev)
        npl = torch.where(u(n) < self.p_people,
                          1 + torch.randint(0, N_PEOPLE, (n,), generator=g, device=self.dev),
                          torch.zeros(n, dtype=torch.long, device=self.dev))
        for j in range(N_PEOPLE):               # wander near the pad, then go inside
            r = 3.0 + 27.0 * torch.sqrt(u(n, W)); a = 2 * math.pi * u(n, W)
            p = pad[:, None] + torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)
            p[:, 1::2] = p[:, 0::2]                                    # dwell = stay put
            a_out = 2 * math.pi * u(n)
            p[:, -1] = pad + 80.0 * torch.stack([torch.cos(a_out), torch.sin(a_out)], -1)
            step = torch.hypot(*(p[:, 1:] - p[:, :-1]).unbind(-1)) / 1.2
            # ASSUMED: each person leaves at a time uniform in 30-240 s. The first
            # version had them wander for a median 332 s -- longer than the whole
            # mission -- so no agent, oracle included, could ever land while they
            # were there. That created no headroom, it only capped every agent's
            # success. This is an assumption to sensitivity-test, not a tuned value.
            t_leave = self.leave[0] + (self.leave[1] - self.leave[0]) * u(n)
            per = ((t_leave - step[:, :10].sum(-1)) / 6.0).clamp(min=1.0)
            dwell = torch.where(torch.arange(W - 1, device=self.dev) % 2 == 0,
                                per[:, None].expand(n, W - 1), torch.zeros(n, W - 1, device=self.dev))
            tt = torch.cat([torch.full((n, 1), -1e3, device=self.dev),
                            torch.cumsum(step + dwell, -1)], -1)
            on = (npl > j)[:, None]
            wp[:, j] = torch.where(on[..., None], p, wp[:, j]); tw[:, j] = torch.where(on, tt, tw[:, j])
        path = self.car_paths[pad_idx]
        length = torch.hypot(*(path[:, 1:] - path[:, :-1]).unbind(-1)).sum(-1)
        for j in range(N_CAR):                  # ~1 car a minute, 40 km/h, either way
            rev = u(n) < 0.5
            pp = torch.where(rev[:, None, None], path.flip(1), path)
            t0 = -30.0 + 330.0 * u(n)
            tt = t0[:, None] + torch.linspace(0, 1, W, device=self.dev)[None] * (length / 11.0)[:, None]
            on = (u(n) < 330.0 / 60.0 / N_CAR)[:, None]
            wp[:, N_PEOPLE + j] = torch.where(on[..., None], pp, wp[:, N_PEOPLE + j])
            tw[:, N_PEOPLE + j] = torch.where(on, tt, tw[:, N_PEOPLE + j])
        return heli, dr, {"wp": wp, "tw": tw}

    @staticmethod
    def heli_at(heli, t):
        """Position (..., 3), velocity (..., 3), airborne (...) at times t (broadcast)."""
        r = heli["v"][..., None] * (heli["t_arr"][..., None] - t)
        air = heli["on"][..., None] & (r > 0)
        rr = r.clamp(min=0.0)
        xy = heli["dir"][..., None, :] * rr[..., None]
        z = (rr * GLIDE).clamp(max=300.0)
        vz = torch.where((rr * GLIDE < 300.0) & air, -heli["v"][..., None] * GLIDE, torch.zeros_like(z))
        vxy = -heli["dir"][..., None, :] * heli["v"][..., None, None] * air[..., None]
        pos = torch.cat([xy, z[..., None]], -1)
        pos = torch.where(heli["on"][..., None, None], pos, torch.full_like(pos, FAR))
        return pos, torch.cat([vxy, vz[..., None]], -1), air

    @staticmethod
    def drones_at(dr, t):
        """(..., K, N_DRONE, 3) positions and velocities, active mask, at times t (..., K)."""
        L = torch.hypot(*(dr["B"] - dr["A"]).unbind(-1)).clamp(min=1.0)
        T = L / dr["v"]
        s = ((t[..., None] - dr["t0"][..., None, :]) / T[..., None, :])
        act = dr["used"][..., None, :] & (s >= 0) & (s <= 1)
        xy = dr["A"][..., None, :, :] + s.clamp(0, 1)[..., None] * (dr["B"] - dr["A"])[..., None, :, :]
        pos = torch.cat([xy, dr["alt"][..., None, :, None].expand(*xy.shape[:-1], 1)], -1)
        pos = torch.where(act[..., None], pos, torch.full_like(pos, FAR))
        vel = ((dr["B"] - dr["A"]) / T[..., None])[..., None, :, :].expand_as(xy) * act[..., None]
        return pos, torch.cat([vel, torch.zeros_like(vel[..., :1])], -1), act

    @staticmethod
    def ground_at(gr, t):
        """(..., K, G, 3) positions of people/cars at times t (..., K); FAR when absent."""
        tw, wp = gr["tw"], gr["wp"]                               # (n,G,W), (n,G,W,2)
        tq = t[..., None].expand(*t.shape, tw.shape[-2])          # (n,K,G)
        twk = tw[:, None].expand(tw.shape[0], t.shape[-1], *tw.shape[1:]).contiguous()
        i = torch.searchsorted(twk, tq[..., None].contiguous()).squeeze(-1).clamp(1, W - 1)
        t0 = twk.gather(-1, (i - 1)[..., None]).squeeze(-1); t1 = twk.gather(-1, i[..., None]).squeeze(-1)
        wk = wp[:, None].expand(wp.shape[0], t.shape[-1], *wp.shape[1:])
        p0 = wk.gather(-2, (i - 1)[..., None, None].expand(*i.shape, 1, 2)).squeeze(-2)
        p1 = wk.gather(-2, i[..., None, None].expand(*i.shape, 1, 2)).squeeze(-2)
        f = ((tq - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0, 1)[..., None]
        act = (tq >= tw[:, None, :, 0]) & (tq <= tw[:, None, :, -1]) & (wp[:, None, :, 0, 0] < FAR / 2)
        xy = p0 + f * (p1 - p0)
        pos = torch.cat([xy, torch.full_like(xy[..., :1], 1.7)], -1)
        return torch.where(act[..., None], pos, torch.full_like(pos, FAR)), act

    # ---- episode --------------------------------------------------------------
    def reset(self):
        n, u = self.n, self._u
        self.pad_idx = torch.randint(0, len(self.pads), (n,), generator=self.g, device=self.dev)
        self.pad = self.pads[self.pad_idx]
        to_hub = self.hub - self.pad
        dist = torch.hypot(*to_hub.unbind(-1)).clamp(min=1.0)
        ang = torch.atan2(to_hub[:, 1], to_hub[:, 0]) + math.radians(20.0) * (2 * u(n) - 1)
        run = dist.clamp(max=1000.0)
        start = self.pad + run[:, None] * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        start = start.clamp(-self.half + 50, self.half - 50)
        self.pos = torch.cat([start, torch.full((n, 1), CRUISE, device=self.dev)], -1)
        self.vel = torch.zeros(n, 3, device=self.dev)
        self.heli, self.dr, self.gr = self.sample_intruders(n, self.pad_idx)
        # the magpie: nest in a tree 20-90 m from the pad, so the pad is always in its territory
        gb = lambda *s: torch.rand(*s, generator=self.g_bird, device=self.dev)
        a, r = 2 * math.pi * gb(n), 20.0 + 70.0 * gb(n)
        nest = torch.cat([self.pad + r[:, None] * torch.stack([torch.cos(a), torch.sin(a)], -1),
                          (10.0 + 5.0 * gb(n))[:, None]], -1)
        self.bird = {"on": gb(n) < self.p_swoop, "nest": nest, "pos": nest.clone(),
                     "vel": torch.zeros(n, 3, device=self.dev), "aim": nest.clone(),
                     "dive": torch.zeros(n, dtype=torch.bool, device=self.dev),
                     "wait": 2.0 + 4.0 * gb(n), "v": V_SWOOP[0] + (V_SWOOP[1] - V_SWOOP[0]) * gb(n),
                     "swoops": torch.zeros(n, device=self.dev),
                     "off": torch.zeros(n, 3, device=self.dev), "t_dive": torch.zeros(n, device=self.dev),
                     "st": self._stage_draw(n) if self.magpie == "hunter" else torch.zeros(n, 3, device=self.dev)}
        self.bird_min = torch.full((n,), FAR, device=self.dev)
        if self.bird_sense == "rig":
            self.rig.reset()
        self.fog = torch.rand(n, generator=self.g_fog, device=self.dev) < self.p_fog
        self.fogged = torch.zeros(n, dtype=torch.bool, device=self.dev)      # a patch hides the ground now
        gn = lambda *s: torch.rand(*s, generator=self.g_new, device=self.dev)
        k = 1 + (gn(n) * (N_NEW - 1)).long().clamp(max=N_NEW - 2)          # 1-4 near the pad
        half = 0.5 * (NEW_SIDE[0] + (NEW_SIDE[1] - NEW_SIDE[0]) * gn(n, N_NEW, 2))
        r = 2.5 + half.norm(dim=-1) + NEW_R * gn(n, N_NEW)       # nearest point stays >= 2.5 m from the pad
        a = 2 * math.pi * gn(n, N_NEW)
        on = gn(n) < self.p_new
        used = on[:, None] & (torch.arange(N_NEW, device=self.dev) < k[:, None])
        c = self.pad[:, None] + r[..., None] * torch.stack([torch.cos(a), torch.sin(a)], -1)
        h = NEW_H[0] + (NEW_H[1] - NEW_H[0]) * gn(n, N_NEW)
        # the tall one, in slot N_NEW-1: on the approach bearing, a 2 x 2 m column
        brg = torch.atan2(*(self.pos[:, :2] - self.pad).unbind(-1)[::-1]) + math.radians(15.0) * (2 * gn(n) - 1)
        rt = TALL_R[0] + (TALL_R[1] - TALL_R[0]) * gn(n)
        c[:, -1] = self.pad + rt[:, None] * torch.stack([torch.cos(brg), torch.sin(brg)], -1)
        half[:, -1] = 1.0
        h[:, -1] = TALL_H[0] + (TALL_H[1] - TALL_H[0]) * gn(n)
        used[:, -1] = on & (gn(n) < P_TALL)
        self.new = {"used": used, "c": c, "half": half, "h": h,
                    "known": torch.zeros(n, N_NEW, dtype=torch.bool, device=self.dev)}
        self.heading = torch.nn.functional.normalize(self.pad - self.pos[:, :2], dim=-1)   # the camera faces the way it flies
        # GeoAnchor error, replayed
        self.bin = torch.randint(0, 10, (n,), generator=self.g, device=self.dev)
        self.sigma = self.sig_med[self.bin] if self.nav == "geoanchor" else torch.full((n,), GPS_BIAS * 1.18, device=self.dev)
        self.gps_bias = GPS_BIAS * torch.randn(n, 2, generator=self.g_nav, device=self.dev)
        self.z = self._nrm(n); self.bearing = 2 * math.pi * u(n)
        self.err = torch.zeros(n, 2, device=self.dev); self._fix()
        self.err_sum, self.n_fix = self.err.clone(), torch.ones(n, device=self.dev)
        self.err_seen = self.err.clone()      # navigation error when the marker was last read
        self.t, self.k = 0.0, 0
        z = lambda dt=torch.float32: torch.zeros(n, dtype=dt, device=self.dev)
        self.done, self.outcome, self.t_done = z(torch.bool), z(torch.long), torch.full((n,), T_MAX, device=self.dev)
        self.seen, self.ever_seen, self.aware = z(torch.bool), z(torch.bool), z(torch.bool)
        self.last_seen = torch.full((n,), -FAR, device=self.dev)
        self.aware_t = torch.full((n,), FAR, device=self.dev)
        self.viol = torch.zeros(n, len(VIOL), device=self.dev)
        self.run = torch.zeros(n, len(VIOL), device=self.dev)      # consecutive seconds
        self.nmac, self.wc_s = z(torch.bool), z()
        self.obliged_ever, self.drone_near_ever = z(torch.bool), z(torch.bool)
        self.people_at_arrival = torch.full((n,), -1.0, device=self.dev)
        return self.obs()

    def _fix(self):
        rho = self.rho
        self.z = rho * self.z + math.sqrt(1 - rho * rho) * self._nrm(self.n)
        uq = (0.5 * (1 + torch.erf(self.z / math.sqrt(2))) * 255).round().long().clamp(0, 255)
        mag = self.qtab[self.bin, uq]
        self.bearing = self.bearing + BEARING_STEP * self._nrm(self.n)
        self.err = mag[:, None] * torch.stack([torch.cos(self.bearing), torch.sin(self.bearing)], -1)
        if self.nav == "gps":          # the GeoAnchor draws above still happen, keeping self.g in step
            self.err = self.gps_bias + GPS_JITTER * torch.randn(self.n, 2, generator=self.g_nav, device=self.dev)

    def _bird(self, live, me0, v0):
        """One step of the magpie: it waits, then dives at where the drone WILL be (lead
        pursuit, then committed to that line), overshoots 15 m, waits, dives again -- but
        only while the drone is in its territory. Returns its closest approach during the
        step: sampling the ends alone would let a 15 m/s bird pass through a 0.8 m radius.
        It decides from where the drone was at the START of the step (me0, v0), when its own
        flight this step begins; aiming from the end made it arrive a step early, crossing
        ~1.2 m in front of a drone flying across its line (15% of dead-on dives missed)."""
        if self.magpie == "hunter":
            return self._hunter(live, me0, v0)
        b, n, me = self.bird, self.n, me0
        gb = lambda *s: torch.rand(*s, generator=self.g_bird, device=self.dev)
        inside = b["on"] & live & (me[:, 2] < Z_TERR) & \
            (torch.hypot(*(me[:, :2] - b["nest"][:, :2]).unbind(-1)) < R_TERR)
        b["wait"] = torch.where(inside & ~b["dive"], b["wait"] - DT, b["wait"])
        go = inside & ~b["dive"] & (b["wait"] <= 0)
        t_hit = torch.linalg.norm(me - b["pos"], dim=-1) / b["v"]
        for _ in range(3):
            t_hit = torch.linalg.norm(me + v0 * t_hit[:, None] - b["pos"], dim=-1) / b["v"]
        aim = me + v0 * t_hit[:, None]
        line = aim - b["pos"]
        side = torch.randn(n, 3, generator=self.g_bird, device=self.dev)
        side = side - (side * line).sum(-1, keepdim=True) / (line * line).sum(-1, keepdim=True).clamp(min=1e-9) * line
        aim = aim + self.miss * gb(n)[:, None] * side / torch.linalg.norm(side, dim=-1, keepdim=True).clamp(min=1e-9)
        b["aim"] = torch.where(go[:, None], aim, b["aim"])
        head = (aim - b["pos"]) / torch.linalg.norm(aim - b["pos"], dim=-1, keepdim=True).clamp(min=1e-9)
        b["vel"] = torch.where(go[:, None], b["v"][:, None] * head, b["vel"])
        b["dive"], b["swoops"] = b["dive"] | go, b["swoops"] + go.float()
        p0 = b["pos"]
        b["pos"] = torch.where(b["dive"][:, None], b["pos"] + b["vel"] * DT, b["pos"])
        b["pos"][:, 2] = b["pos"][:, 2].clamp(min=1.0)                # pulls up before the ground
        past = b["dive"] & (((b["pos"] - b["aim"]) * b["vel"]).sum(-1) / b["v"] > 15.0)
        b["dive"] = b["dive"] & ~past
        b["wait"] = torch.where(past, REARM[0] + (REARM[1] - REARM[0]) * gb(n), b["wait"])
        b["vel"] = torch.where(past[:, None], torch.zeros_like(b["vel"]), b["vel"])
        r0, r1 = p0 - me0, b["pos"] - self.pos                        # both moved in straight lines
        d = r1 - r0
        s = (-(r0 * d).sum(-1) / (d * d).sum(-1).clamp(min=1e-9)).clamp(0, 1)
        return torch.where(b["on"], torch.linalg.norm(r0 + s[:, None] * d, dim=-1), torch.full_like(s, FAR))

    def _stage_draw(self, n):
        """Where the hunter waits for its next dive: angle off the tail, metres back, metres up."""
        u = torch.rand(n, 3, generator=self.g_bird, device=self.dev)
        return torch.stack([STAGE_ANG * (2 * u[:, 0] - 1), STAGE_BACK[0] + (STAGE_BACK[1] - STAGE_BACK[0]) * u[:, 1],
                            STAGE_UP[0] + (STAGE_UP[1] - STAGE_UP[0]) * u[:, 2]], -1)

    def _hunter(self, live, me0, v0):
        """One step of the hunter magpie (see BEHIND..DIVE_S). Returns its closest approach during
        the step while diving -- repositioning is not an attack, and keeps 8 m clear."""
        b, n, me = self.bird, self.n, me0
        gb = lambda *s: torch.rand(*s, generator=self.g_bird, device=self.dev)
        unit = lambda x: x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-9)
        inside = b["on"] & live & (me[:, 2] < Z_TERR) & \
            (torch.hypot(*(me[:, :2] - b["nest"][:, :2]).unbind(-1)) < R_TERR)
        rel = b["pos"] - me
        dist, flat = torch.linalg.norm(rel, dim=-1), rel[:, :2].norm(dim=-1)
        rear = (rel[:, :2] * self.heading).sum(-1) < math.cos(BEHIND) * flat
        b["wait"] = torch.where(inside & ~b["dive"], b["wait"] - DT, b["wait"])
        go = inside & ~b["dive"] & (b["wait"] <= 0) & rear & (dist >= DIVE_R[0]) & (dist <= DIVE_R[1])
        t_go = dist / b["v"]                                            # lead pursuit
        for _ in range(3):
            t_go = torch.linalg.norm(me + v0 * t_go[:, None] - b["pos"], dim=-1) / b["v"]
        line = me + v0 * t_go[:, None] - b["pos"]
        side = torch.randn(n, 3, generator=self.g_bird, device=self.dev)
        side = side - (side * line).sum(-1, keepdim=True) / (line * line).sum(-1, keepdim=True).clamp(min=1e-9) * line
        b["off"] = torch.where(go[:, None], self.miss * gb(n)[:, None] * unit(side), b["off"])
        aim = me + v0 * t_go[:, None] + b["off"]                         # the pass it means to make
        c, d = unit(b["vel"]), unit(aim - b["pos"])
        th = torch.acos((c * d).sum(-1).clamp(-1.0, 1.0))
        f = (A_TURN * DT / b["v"] / th.clamp(min=1e-6)).clamp(max=1.0)   # share of the turn it can make
        sn = torch.sin(th).clamp(min=1e-6)
        turned = torch.where((th < 1e-4)[:, None], d, (torch.sin((1 - f) * th) / sn)[:, None] * c
                             + (torch.sin(f * th) / sn)[:, None] * d)
        steer = b["dive"] & (t_go > T_COMMIT)                           # too close to correct: committed
        dive_v = torch.where(go[:, None], d, torch.where(steer[:, None], unit(turned), c)) * b["v"][:, None]
        b["aim"] = torch.where((go | steer)[:, None], aim, b["aim"])
        b["dive"], b["swoops"] = b["dive"] | go, b["swoops"] + go.float()
        b["t_dive"] = torch.where(b["dive"], b["t_dive"] + DT, torch.zeros_like(b["t_dive"]))
        # not diving: behind and above the drone while it is in the territory, else home to the nest
        ca, sa = torch.cos(b["st"][:, 0]), torch.sin(b["st"][:, 0])
        tail = -torch.stack([ca * self.heading[:, 0] - sa * self.heading[:, 1], sa * self.heading[:, 0] + ca * self.heading[:, 1]], -1)
        stage = me + torch.cat([b["st"][:, 1:2] * tail, b["st"][:, 2:3]], -1)
        stage[:, 2] = stage[:, 2].clamp(max=Z_TERR - 1.0)
        hop = torch.where(inside[:, None], stage, b["nest"]) - b["pos"]
        hop = hop * (V_STAGE * DT / torch.linalg.norm(hop, dim=-1, keepdim=True).clamp(min=V_STAGE * DT))
        b["vel"] = torch.where(b["dive"][:, None], dive_v, hop / DT) * b["on"][:, None]
        p0, diving = b["pos"], b["dive"].clone()
        b["pos"] = b["pos"] + b["vel"] * DT
        away = b["pos"] - self.pos                                      # repositioning keeps 8 m clear
        near = ~b["dive"] & b["on"] & (torch.linalg.norm(away, dim=-1) < 8.0)
        b["pos"] = torch.where(near[:, None], self.pos + 8.0 * unit(away), b["pos"])
        b["pos"][:, 2] = b["pos"][:, 2].clamp(min=1.0)
        past = ((b["pos"] - self.pos) * b["vel"]).sum(-1) / b["v"] > 15.0
        stop = b["dive"] & (past | (b["t_dive"] > DIVE_S) | ~inside)
        b["dive"] = b["dive"] & ~stop
        b["wait"] = torch.where(stop, REARM[0] + (REARM[1] - REARM[0]) * gb(n), b["wait"])
        b["st"] = torch.where(stop[:, None], self._stage_draw(n), b["st"])
        r0, r1 = p0 - me0, b["pos"] - self.pos
        dd = r1 - r0
        s = (-(r0 * dd).sum(-1) / (dd * dd).sum(-1).clamp(min=1e-9)).clamp(0, 1)
        return torch.where(b["on"] & diving, torch.linalg.norm(r0 + s[:, None] * dd, dim=-1), torch.full_like(s, FAR))

    def survey_at(self, xy):
        ij = ((xy + self.half) / self.res).long().clamp(0, self.H.shape[0] - 1)
        return self.H[ij[..., 1], ij[..., 0]]

    def _box_height(self, xy, mask):
        """Tallest of this episode's unmapped boxes at xy (n, ..., 2), counting those in mask."""
        b = self.new
        q = xy.reshape(self.n, -1, 1, 2)
        inside = ((q - b["c"][:, None]).abs() <= b["half"][:, None]).all(-1) & mask[:, None]
        return torch.where(inside, b["h"][:, None], torch.zeros_like(b["h"][:, None])).amax(-1).reshape(xy.shape[:-1])

    def height_at(self, xy):
        """The TRUE world: survey plus everything it missed. Physics -- crashes, sight lines -- uses this."""
        return self.survey_at(xy) if not self.p_new else torch.maximum(self.survey_at(xy), self._box_height(xy, self.new["used"]))

    def known_rel(self, d):
        """What the sensors have seen of the unmapped, at displacement d (n, ..., 2) from the drone.
        A depth camera measures relative to the drone, so this geometry is exact whatever the
        navigation error; only the survey, an absolute map, inherits that error. (Stored in the
        true frame and looked up at the believed position, a 2 m column sat ~1.4 m off; stored
        at the believed position and looked up while tracking the marker, likewise -- drones flew
        into columns they had already seen, both ways.)"""
        if not self.p_new:
            return torch.zeros(d.shape[:-1], device=self.dev)
        p = self.pos[:, :2].reshape(self.n, *([1] * (d.dim() - 2)), 2) + d
        return self._box_height(p, self.new["known"])

    def _line_of_sight(self):
        s = torch.linspace(0.05, 0.95, 12, device=self.dev)
        xy = self.pos[:, None, :2] + s[:, None] * (self.pad - self.pos[:, :2])[:, None]
        zl = self.pos[:, None, 2] * (1 - s)
        return (self.height_at(xy) <= zl + 0.2).all(-1)

    # ---- what the drone knows ----------------------------------------------------
    def obs(self):
        """Everything an agent may use. Map features are offset by the localization
        error; intruders are relative measurements and are not."""
        t = torch.full((self.n, 1), self.t, device=self.dev)
        hp, hv, hair = self.heli_at(self.heli, t)
        dp, dv, _ = self.drones_at(self.dr, t)
        gp, _ = self.ground_at(self.gr, t)
        gp2, _ = self.ground_at(self.gr, t + DT)              # velocity by differencing the track
        gv = torch.where((gp2 < FAR / 2) & (gp < FAR / 2), (gp2 - gp) / DT, torch.zeros_like(gp))
        hp, hv, hair, dp, dv, gp, gv = hp[:, 0], hv[:, 0], hair[:, 0], dp[:, 0], dv[:, 0], gp[:, 0], gv[:, 0]
        rel_h = hp - self.pos
        see_h = torch.linalg.norm(rel_h, dim=-1) <= self.d_heli
        rel_d = dp - self.pos[:, None]
        see_d = torch.linalg.norm(rel_d, dim=-1) <= self.d_drone
        rel_g = gp - self.pos[:, None]
        see_g = (torch.hypot(*rel_g[..., :2].unbind(-1)) <= self.pos[:, 2:3].clamp(min=1.0) * FOV_TAN) & ~self.fogged[:, None]
        mean_err = self.err_sum / self.n_fix[:, None]
        rel_b = self.bird["pos"] - self.pos
        see_b = self.bird["on"] & (torch.linalg.norm(rel_b, dim=-1) <= D_BIRD)
        if self.bird_sense == "cameras":
            horiz = rel_b[:, :2].norm(dim=-1).clamp(min=1e-6)
            ahead = (rel_b[:, :2] * self.heading).sum(-1) / horiz
            elev = torch.atan2(rel_b[:, 2], horiz)
            fwd = (ahead >= math.cos(DEPTH_HFOV / 2)) & (elev.abs() <= DEPTH_VFOV / 2)
            down = (rel_b[:, 2] < 0) & (horiz <= -rel_b[:, 2] * FOV_TAN)
            see_b = see_b & (fwd | down)
        if self.bird_sense == "rig":                 # what the tracker believes, once it has a track
            tr = self.rig.track
            rel_b, see_b = tr.x[:, :3] - self.pos, tr.reported & self.bird["on"]
            bird_vel = tr.x[:, 3:]
        else:
            bird_vel = self.bird["vel"]
        return {
            "t": self.t, "pos_z": self.pos[:, 2], "vel": self.vel,
            "bel_xy": self.pos[:, :2] + self.err,              # where it thinks it is (GPS-denied fix)
            "pad_xy": self.pad,                                 # the delivery address: surveyed, known
            "pad_rel_bel": (self.pad - mean_err) - self.pos[:, :2],   # averaged-fix aim
            "pad_rel_vis": self.pad - self.pos[:, :2],         # valid only when tracked
            # Once locked: where the last sighting put the pad, carried on the drone's own
            # navigation (exact but for how much its error moved since: GPS ~0.4 m).
            "pad_rel_fix": torch.where(self.ever_seen[:, None], (self.pad - self.pos[:, :2]) + (self.err_seen - self.err),
                                       (self.pad - mean_err) - self.pos[:, :2]),
            "fog": self.fog,                                   # the weather is known; the patches are not
            "seen": self.seen, "ever_seen": self.ever_seen,
            # A per-frame decode flickers; the PX4 mission keeps the last sighting
            # (dead-reckoned) for LOST_S. Without this a lander bounces between the
            # visual fix and a GPS aim that is ~9 m off every time a frame misses.
            "track": (self.t - self.last_seen) < LOST_S,
            "hls_rel_bel": -(self.pos[:, :2] + self.err),       # helipad at the origin
            "heli_rel": torch.where(see_h[:, None], rel_h, torch.full_like(rel_h, FAR)),
            "heli_vel": hv * see_h[:, None], "heli_seen": see_h,
            "aware": self.aware, "obliged": self.aware & (self.t < self.heli["t_arr"] + 30.0) & self.heli["on"],
            "drone_rel": torch.where(see_d[..., None], rel_d, torch.full_like(rel_d, FAR)),
            "drone_vel": dv * see_d[..., None], "drone_seen": see_d,
            "ground_rel": torch.where(see_g[..., None], rel_g, torch.full_like(rel_g, FAR)),
            "ground_seen": see_g, "ground_vel": gv * see_g[..., None], "done": self.done,
            "bird_rel": torch.where(see_b[:, None], rel_b, torch.full_like(rel_b, FAR)),
            "bird_vel": bird_vel * see_b[:, None], "bird_seen": see_b,
        }

    # ---- one step ------------------------------------------------------------------
    def step(self, cmd):
        live = ~self.done
        me0, v0 = self.pos.clone(), self.vel.clone()
        cmd = cmd.clone()
        sp = torch.hypot(cmd[:, 0], cmd[:, 1]).clamp(min=1e-6)
        cmd[:, :2] *= (V_XY / sp).clamp(max=1.0)[:, None]
        cmd[:, 2] = cmd[:, 2].clamp(-VZ_DN, VZ_UP)
        a = torch.tensor([1 - math.exp(-DT / TAU_XY)] * 2 + [1 - math.exp(-DT / TAU_Z)], device=self.dev)
        v_new = self.vel + a * (cmd - self.vel)
        self.pos = torch.where(live[:, None], self.pos + 0.5 * (self.vel + v_new) * DT, self.pos)
        self.vel = torch.where(live[:, None], v_new, self.vel)
        self.pos[:, 2] = self.pos[:, 2].clamp(min=0.0)
        self.t += DT; self.k += 1
        if self.k % round(FIX_DT / DT) == 0:
            self._fix()
            self.err_sum = torch.where(live[:, None], self.err_sum + self.err, self.err_sum)
            self.n_fix = self.n_fix + live.float()

        # marker: inside the footprint, in line of sight, and decoded this frame
        off = torch.hypot(*(self.pad - self.pos[:, :2]).unbind(-1))
        h = self.pos[:, 2].clamp(min=0.3)
        px = PAD_M * F_PX / h
        p_dec = torch.sigmoid((px - DET_X0) / DET_S)
        # fog: a two-state patch process whose hidden share at this height is 1 - FOG_SEE
        clear = _interp(self.pos[:, 2], FOG_Z, FOG_SEE)
        p_lift = DT / FOG_PATCH_S
        u_f = torch.rand(self.n, generator=self.g_fog, device=self.dev)
        self.fogged = self.fog & torch.where(self.fogged, u_f >= p_lift, u_f < p_lift * (1 - clear) / clear.clamp(min=1e-3))
        self.seen = (off <= h * FOV_TAN) & self._line_of_sight() & (self._u(self.n) < p_dec) & live & ~self.fogged
        self.err_seen = torch.where(self.seen[:, None], self.err, self.err_seen)
        self.ever_seen |= self.seen
        self.last_seen = torch.where(self.seen, torch.full_like(self.last_seen, self.t), self.last_seen)

        if self.p_new or self.bird_sense != "omni" or self.magpie == "hunter":   # the way it flies
            v2 = self.vel[:, :2]; sp = v2.norm(dim=-1, keepdim=True)
            self.heading = torch.where(sp > 0.5, v2 / sp.clamp(min=1e-6), self.heading)
        if self.p_new:   # sensing the unmapped: the voxel map fills in
            b, z = self.new, self.pos[:, 2:3]
            rel = b["c"] - self.pos[:, None, :2]
            gap = (rel.abs() - b["half"]).clamp(min=0).norm(dim=-1)                  # to its nearest point
            ahead = (rel * self.heading[:, None]).sum(-1) / rel.norm(dim=-1).clamp(min=1e-6)
            fwd = (gap <= DEPTH_R) & ((ahead >= math.cos(DEPTH_HFOV / 2)) | (gap < 1.0)) \
                & (b["h"] >= z - gap * math.tan(DEPTH_VFOV / 2))
            down = (gap <= z * DOWN_TAN) & (z <= DOWN_R)
            b["known"] = b["known"] | (b["used"] & (fwd | down) & live[:, None])
        bp0 = self.bird["pos"].clone() if self.bird_sense == "rig" else None
        db = self._bird(live, me0, v0)
        self.bird_min = torch.where(live, torch.minimum(self.bird_min, db), self.bird_min)
        if self.bird_sense == "rig":
            self.rig.step(self, self.t, DT, me0, bp0)

        # intruders now
        t = torch.full((self.n, 1), self.t, device=self.dev)
        hp, _, hair = self.heli_at(self.heli, t); hp, hair = hp[:, 0], hair[:, 0]
        dp, _, _ = self.drones_at(self.dr, t); dp = dp[:, 0]
        gp, _ = self.ground_at(self.gr, t); gp = gp[:, 0]
        dh = hp - self.pos
        hh, hv = torch.hypot(dh[:, 0], dh[:, 1]), dh[:, 2].abs()
        self.aware |= (torch.linalg.norm(dh, dim=-1) <= self.d_heli) & hair & live
        self.aware_t = torch.where(self.aware & (self.aware_t >= FAR), torch.full_like(self.aware_t, self.t), self.aware_t)
        dd = torch.linalg.norm(dp - self.pos[:, None], dim=-1).min(-1).values
        dg3 = torch.linalg.norm(gp - self.pos[:, None], dim=-1)
        dgh = torch.hypot(*(gp - self.pos[:, None])[..., :2].unbind(-1))

        # CASA / ASTM compliance, judged on TRUE position
        r_hls = torch.hypot(self.pos[:, 0], self.pos[:, 1])
        away = (self.pos[:, 0] * self.vel[:, 0] + self.pos[:, 1] * self.vel[:, 1]) / r_hls.clamp(min=1.0)
        obliged = self.aware & (self.t < self.heli["t_arr"] + 30.0) & self.heli["on"] & (r_hls <= HLS_R)
        self.obliged_ever |= obliged & live
        cond = torch.stack([
            self.pos[:, 2] > CEIL,
            (dg3[:, :N_PEOPLE] < PEOPLE_R).any(-1),
            (dg3[:, N_PEOPLE:] < PEOPLE_R).any(-1),
            ((dgh < OVER_R) & (gp[..., 2] < self.pos[:, None, 2])).any(-1),
            obliged & (away < 1.0),
            dd < DRONE_SEP], -1) & live[:, None]
        self.run = torch.where(cond, self.run + DT, torch.zeros_like(self.run))
        grace = torch.tensor([0.0] + [GRACE] * 5, device=self.dev)
        self.viol += ((self.run > grace) & cond).float() * DT
        self.nmac |= (hh < NMAC_H) & (hv < NMAC_V) & hair & live
        self.wc_s += ((hh < WC_H) & (hv < WC_V) & hair & live).float() * DT
        self.drone_near_ever |= (dd < self.d_drone) & live
        near_pad = (off < 60.0) & (self.people_at_arrival < 0) & live
        self.people_at_arrival = torch.where(near_pad, (torch.hypot(*(gp[..., :2] - self.pad[:, None]).unbind(-1)) < PEOPLE_R).any(-1).float(),
                                             self.people_at_arrival)

        # termination
        hit_obst = self.height_at(self.pos[:, :2]) > self.pos[:, 2]
        if self.p_new:   # a 2 m column fits between 0.25 s samples at 8 m/s: test the step's midpoint too
            u, mid = self.new["used"], 0.5 * (me0 + self.pos)
            hit_new = (self._box_height(self.pos[:, :2], u) > self.pos[:, 2]) | (self._box_height(mid[:, :2], u) > mid[:, 2])
            hit_obst = hit_obst | hit_new
        hit_drone = dd < HIT_DRONE
        hit_heli = torch.linalg.norm(dh, dim=-1) < HIT_HELI
        landed = self.pos[:, 2] <= 0.3
        on_pad = self.ever_seen & (off <= 1.0)
        out = torch.zeros_like(self.outcome)
        out = torch.where(landed & ~on_pad, torch.full_like(out, 5), out)
        out = torch.where(landed & on_pad, torch.full_like(out, 1), out)
        out = torch.where(hit_heli, torch.full_like(out, 4), out)
        out = torch.where(db < HIT_BIRD, torch.full_like(out, 7), out)
        out = torch.where(hit_drone, torch.full_like(out, 3), out)
        out = torch.where(hit_obst & ~landed, torch.full_like(out, 2), out)
        if self.p_new:   # into something the survey did not show
            unmapped = hit_new & ~landed & (self.survey_at(self.pos[:, :2]) <= self.pos[:, 2])
            out = torch.where(unmapped, torch.full_like(out, 8), out)
        if self.t >= T_MAX - 1e-6:
            out = torch.where(out == 0, torch.full_like(out, 6), out)
        new = live & (out > 0)
        self.outcome = torch.where(new, out, self.outcome)
        self.t_done = torch.where(new, torch.full_like(self.t_done, self.t), self.t_done)
        self.done |= new
        return self.obs()

    # ---- the scorecard ----------------------------------------------------------------
    def summary(self):
        crit = self.nmac | (self.viol > 0).any(-1)
        return {"delivered": self.outcome == 1,
                "clean": (self.outcome == 1) & ~crit,
                "crash": crashed(self.outcome),
                "outcome": self.outcome, "t_done": self.t_done,
                "viol_s": self.viol, "nmac": self.nmac, "wellclear_s": self.wc_s,
                "obliged_ever": self.obliged_ever, "drone_near_ever": self.drone_near_ever,
                "people_at_arrival": self.people_at_arrival, "heli_on": self.heli["on"],
                "sigma": self.sigma, "bird_on": self.bird["on"], "swoops": self.bird["swoops"],
                "bird_min": self.bird_min, "fog": self.fog, "new_on": self.new["used"].any(-1),
                "new_known": self.new["known"].sum(-1)}


def run(env, agent, seed=None):
    """Roll every episode to the end. Agents see obs(), never the env's internals
    -- except an oracle, which is handed the env on purpose and says so."""
    o = env.reset()
    agent.reset(env)
    while not bool(env.done.all()) and env.t < T_MAX - 1e-6:
        o = env.step(agent.act(o))
    return env.summary()


class StraightLander:
    """The dumbest competent agent: fly at the averaged-fix aim, descend once the
    marker is seen. No intruder handling at all -- the control for demo()."""

    def reset(self, env):
        pass

    def act(self, o):
        tgt = torch.where(o["track"][:, None], o["pad_rel_vis"], o["pad_rel_bel"])
        d = torch.hypot(*tgt.unbind(-1)).clamp(min=1e-6)
        v = (torch.sqrt(2 * 1.2 * d).clamp(max=V_XY) / d)[:, None] * tgt
        over = o["track"] & (d < 2.0)
        vz = torch.where(over, -torch.where(o["pos_z"] > 8, 2.5, 0.8), 0.6 * (CRUISE - o["pos_z"]))
        vz = torch.where(~over & (d < 40), 0.6 * (35.0 - o["pos_z"]), vz)
        return torch.cat([v, vz[:, None]], -1)


def demo_hunter(device=DEV):
    """9. The hunter magpie: attacks only from the rear sector, from varied angles; a drone that
    does not dodge is touched on ~12% of dives (the straight diver's calibration); dodging EARLY no
    longer helps -- the bird re-aims. (The straight diver misses a drone that sidesteps 2 s out on
    ~99% of dives: what made the old bird results too easy.)"""
    def hunt(t_dodge, magpie="hunter"):
        e = UrbanEnv(2048, "random", seed=9, p_heli=0.0, p_people=0.0, p_swoop=1.0, magpie=magpie, device=device)
        e.reset(); e.dr["used"][:] = False; e.H.zero_()
        e.pos[:, :2] = e.bird["nest"][:, :2] - torch.tensor([60.0, 0.0], device=e.dev); e.pos[:, 2] = 25.0
        e.heading = torch.tensor([[1.0, 0.0]], device=e.dev).expand(e.n, 2).clone()
        bearing = []
        for _ in range(int(60.0 / DT)):
            b = e.bird
            cmd = torch.tensor([[5.0, 0.0, 0.0]], device=e.dev).expand(e.n, 3).clone()
            if t_dodge:
                perp = torch.nn.functional.normalize(torch.stack([-b["vel"][:, 1], b["vel"][:, 0]], -1), dim=-1)
                dodge = b["dive"] & ((b["pos"] - e.pos).norm(dim=-1) / b["v"] < t_dodge)
                cmd[:, :2] += 6.0 * perp * dodge[:, None]
            p0, me0, sw0, h0 = b["pos"].clone(), e.pos.clone(), b["swoops"].clone(), e.heading.clone()
            e.step(cmd)
            new = e.bird["swoops"] > sw0
            r = (p0 - me0)[new][:, :2]
            bearing.append(torch.rad2deg(torch.acos(((r * h0[new]).sum(-1) / r.norm(dim=-1).clamp(min=1e-6)).clamp(-1, 1))))
        return (e.outcome == 7).sum().item() / max(e.bird["swoops"].sum().item(), 1), torch.cat(bearing)
    still, brg = hunt(None)
    early, _ = hunt(2.0)
    early_straight, _ = hunt(2.0, "straight")
    assert 0.09 < still < 0.15, f"hunter touched {still:.1%} of dives at a drone that did not dodge, want ~12%"
    assert early > 0.8 * still, f"dodging 2 s early cut the hunter's touches {still:.1%} -> {early:.1%}: it should re-aim"
    assert brg.min() > 110.0 - 1e-3 and brg.std() > 10.0, f"launches off the heading: min {brg.min():.0f}, spread {brg.std():.0f} deg"
    print(f"9. hunter magpie: touched {still:.1%} of dives when the drone does not dodge, {early:.1%} when it dodges "
          f"2 s early (straight diver: {early_straight:.1%}); launched {brg.min():.0f}-{brg.max():.0f} deg off the heading")


def demo(device=DEV):
    """The checks that have to pass before any number from this env means anything."""
    torch.manual_seed(0)
    env_q = UrbanEnv(2048, "corridor", seed=1, p_heli=0.0, p_people=0.0)
    o = env_q.reset(); env_q.dr["used"][:] = False
    agent = StraightLander()
    while not bool(env_q.done.all()) and env_q.t < T_MAX - 1e-6:
        o = env_q.step(agent.act(o))
    q = env_q.summary()
    dl = q["delivered"].float().mean().item()
    print(f"1. empty sky, straight at the averaged fix: {100*dl:.1f}% delivered, "
          f"{100*q['crash'].float().mean().item():.1f}% crashed, median {q['t_done'][q['delivered']].median().item():.0f} s")
    assert dl > 0.95, "an agent with no intruders to handle cannot deliver -- the env is broken"

    # 2. a building is solid
    e = UrbanEnv(64, "random", seed=2); e.reset()
    tall = (e.H > 8).nonzero()[0]
    e.pos[:, 0] = tall[1].float() - e.half + 0.5; e.pos[:, 1] = tall[0].float() - e.half + 0.5
    e.pos[:, 2] = 2.0; e.step(torch.zeros(64, 3, device=e.dev))
    assert (e.outcome == 2).all(), "flying into a building did not crash"
    print("2. a drone placed inside a building crashes")

    # 3. the CASA helipad rule fires: aware, inside 1.4 km, flying at the helipad
    e = UrbanEnv(64, "random", seed=3); e.reset()
    e.heli["on"][:] = True; e.heli["t_arr"][:] = 30.0; e.heli["dir"][:] = torch.tensor([1.0, 0.0])
    e.heli["v"][:] = 20.0
    e.pos[:] = torch.tensor([800.0, 0.0, 50.0], device=e.dev); e.dr["used"][:] = False
    for _ in range(40):
        e.step(torch.tensor([[-8.0, 0.0, 0.0]], device=e.dev).expand(64, 3))
    assert e.aware.all() and (e.viol[:, 4] > 0).all(), "approaching the helipad while aware was not flagged"
    print(f"3. flying at the helipad while a helicopter is inbound: {e.viol[0, 4].item():.1f} s of violation logged")

    # 4. everyone counts: a person beside the pad blocks a descent onto it
    e = UrbanEnv(64, "random", seed=4, p_heli=0.0); e.reset(); e.dr["used"][:] = False
    e.gr["wp"][:, 0] = e.pad[:, None] + torch.tensor([5.0, 0.0], device=e.dev)
    e.gr["tw"][:, 0] = torch.linspace(-1e3, 1e3, W, device=e.dev)
    e.pos[:, :2] = e.pad; e.pos[:, 2] = 20.0
    for _ in range(40):
        e.step(torch.tensor([[0.0, 0.0, -0.4]], device=e.dev).expand(64, 3))
    assert (e.viol[:, 1] > 0).all(), "descending beside a person was not flagged"
    print(f"4. descending 5 m from a person: {e.viol[0, 1].item():.1f} s of CASA 30 m violation logged")

    # 5. do the encounters actually happen? (report, never tune to taste)
    for world in ("corridor", "random"):
        e = UrbanEnv(4096, world, seed=5)
        s = run(e, StraightLander())
        f = lambda x: 100 * x.float().mean().item()
        print(f"5. {world:<8} straight lander: delivered {f(s['delivered']):.0f}%  clean {f(s['clean']):.0f}%  | "
              f"helipad duty {f(s['obliged_ever']):.0f}%  NMAC {f(s['nmac']):.1f}%  drone within 200 m "
              f"{f(s['drone_near_ever']):.0f}%  people at pad {f(s['people_at_arrival'] > 0):.0f}%")

    # 6. --nav gps: GNSS-sized error, and every other draw -- intruders, people, pads -- unchanged
    a, b = UrbanEnv(4096, "random", seed=6), UrbanEnv(4096, "random", seed=6, nav="gps")
    a.reset(); b.reset()
    assert all(torch.equal(a.heli[k], b.heli[k]) for k in ("on", "t_arr", "dir", "v")), "GPS mode moved the helicopters"
    assert torch.equal(a.pad_idx, b.pad_idx) and torch.equal(a.gr["wp"], b.gr["wp"]), "GPS mode moved pads or people"
    med = lambda e: torch.hypot(*e.err.unbind(-1)).median().item()
    assert 1.2 < med(b) < 1.8, f"GPS error median {med(b):.2f} m, expected ~1.45 m"
    print(f"6. GPS mode: median position error {med(b):.2f} m (GeoAnchor {med(a):.1f} m); same intruders, people and pads")

    # 7. the magpie. With no deliberate miss, a dive must hit a drone flying a straight line
    # (lead pursuit and the in-step contact test both work); above its territory, no swoops.
    def magpie(z, v, miss, secs):
        e = UrbanEnv(1024, "random", seed=7, p_heli=0.0, p_people=0.0, p_swoop=1.0); e.reset()
        e.dr["used"][:] = False; e.miss = miss
        e.pos[:, :2] = e.bird["nest"][:, :2] - torch.tensor([40.0, 0.0], device=e.dev); e.pos[:, 2] = z
        e.H.zero_()                                           # open sky: this tests the bird, not the map
        for _ in range(int(secs / DT)):
            e.step(torch.tensor([[v, 0.0, 0.0]], device=e.dev).expand(e.n, 3))
        return e
    e = magpie(30.0, 5.0, 0.0, 20.0)
    hit = (e.outcome == 7).float().mean().item()
    assert hit > 0.95, f"a dead-on dive hit only {hit:.0%} of straight-flying drones"
    e = magpie(45.0, 0.0, MISS, 60.0)
    assert (e.bird["swoops"] == 0).all(), "a drone above the territory was swooped"
    e = magpie(30.0, 0.0, MISS, 60.0)
    print(f"7. magpie: a dead-on dive hits {hit:.0%} of drones flying straight; above {Z_TERR:.0f} m, no swoops; "
          f"hovering 60 s in its territory without dodging: struck {(e.outcome == 7).float().mean().item():.0%}, "
          f"{e.bird['swoops'].mean().item():.1f} swoops on average")

    # 8. fog: hovering over its pad at each height, the drone sees through as often as it did in
    # Gazebo's fog, in patches that last about FOG_PATCH_S; with p_fog = 0 nothing is hidden.
    for z, want in zip(FOG_Z[1:], FOG_SEE[1:]):
        e = UrbanEnv(2048, "random", seed=8, p_heli=0.0, p_people=0.0, p_fog=1.0, device=device); e.reset()
        e.dr["used"][:] = False; e.pos[:, :2] = e.pad; e.pos[:, 2] = z; e.H.zero_()
        hid, flips = [], 0
        for _ in range(600):
            was = e.fogged.clone(); e.step(torch.zeros(e.n, 3, device=e.dev)); e.pos[:, 2] = z
            hid.append(e.fogged.float().mean().item()); flips += (was & ~e.fogged).sum().item()
        clear, patch = 1 - sum(hid[100:]) / 500, sum(hid) * e.n * DT / max(flips, 1)
        assert abs(clear - want) < 0.03 and abs(patch - FOG_PATCH_S) < 0.4, (z, clear, want, patch)
        print(f"8. fog at {z:.0f} m: sees through {clear:.2f} of the time (Gazebo {want:.2f}), patches {patch:.1f} s")
    e = UrbanEnv(256, "random", seed=8, device=device); e.reset()
    for _ in range(40): e.step(torch.zeros(e.n, 3, device=e.dev))
    assert not e.fogged.any(), "fog without p_fog"
    demo_hunter(device)


if __name__ == "__main__":
    demo()
