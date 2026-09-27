#!/usr/bin/env python3
"""
cage.py -- the last metres of a delivery at Tello scale, in Sai's cage (3 m x 2 m floor, 2 m net ceiling).

A DJI Tello takes off from the hub pad and must land on its customer's ArUco pad -- not a neighbour's --
among tiny model houses and trees, one obstacle possibly missing from the map. Thousands of flights advance
in lock-step as tensors, like urban/urban_env.py; a decider reads obs() and hands step() a velocity command,
as the Tello SDK's `rc` command does, or asks to land (the SDK's `land`: once started, it finishes).

WHAT THE TELLO KNOWS
  position   dead reckoning from its downward optical flow: it drifts, and misjudges speed by a few percent
             for the whole flight. Reading any floor marker resets it (the pads are surveyed markers).
  markers    Sai's 45-degree mirror turns the bottom of the forward picture straight down: a strip 22 deg
             along the flight axis by 70 deg across. A 15 cm marker spans ~100/z px there, so its ID reads
             at any height under the net; the narrow strip, not resolution, is what limits the search.
  obstacles  the map (the houses and trees as placed) plus what the upper picture shows ahead: 70 deg wide,
             1 m deep. The Tello keeps facing +x (east), so it sees nothing west of it.
Every number marked [knob] is a guess until measured on the real Tello in the cage.

    python cage.py             self-checks, then the rule on 2,048 flights
"""
import math

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# ---- the cage and the drone ------------------------------------------------------------------------
L, B, H = 3.0, 2.0, 2.0                  # cage floor 3 m (x, east) by 2 m (y, north); net at 2 m (Sai, 20 Sept)
DT, HOLD, T_MAX = 0.1, 5, 90.0           # physics step; a decision every 0.5 s; flight time limit
V_XY, V_Z, V_LAND = 0.30, 0.25, 0.20     # [knob] rc speeds flown indoors (m/s)
TAU_XY, TAU_Z, LAG = 0.45, 0.30, 2       # [knob] velocity response lag (s); command latency (0.1 s steps, Wi-Fi)
R_DRONE = 0.10                           # Tello with guards 98 x 93 mm: half-diagonal plus 3 cm
Z_TAKEOFF, Z_CRUISE, Z_FLARE = 0.8, 1.4, 0.8    # `takeoff` height; transit above the mapped obstacles; `land` from
# (below ~0.4 m the strip is narrower than the marker along the flight axis: 0.19 z < 7.5 cm -- no lock left to
# centre on. From 0.8 m the marker stays whole within +/-8 cm of centre, and `land` descends on its own)
DRIFT, SCALE_SD, FIX_SD = 0.03, 0.04, 0.01      # [knob] VPS walk m/sqrt(s); speed scale error; fix error + 1 cm/m
TAN_ALONG, TAN_ACROSS = math.tan(math.radians(11.0)), math.tan(math.radians(35.1))   # mirror strip half-angles
MARKER, ID_PX = 0.15, 40                 # [knob] marker side (m); pixels across it needed to read its ID
PX_AT_1M = 960 / (2 * TAN_ACROSS)        # 683 px per metre of floor from 1 m up
FWD_R, FWD_TAN = 1.0, math.tan(math.radians(35.0))   # the forward picture: depth, half-width
PAD_OK, LAND_SD = 0.10, 0.04             # delivered if within 10 cm of the pad centre; [knob] `land` scatter from 0.8 m
N_PADS, N_HOUSES, N_TREES, P_NEW = 3, 4, 4, 0.5      # pad 0 is the customer's; the rest are neighbours'
HUB = (0.35, 1.0)
OUTCOMES = ["flying", "delivered", "wrong pad", "off pad", "crash", "net", "timeout"]


def _place(g, n, count, lo, hi, avoid, gap, tries=40, dev=DEV):
    """count points per flight, uniform in the box lo..hi, each >= gap (per point already placed) from the
    points in avoid (n, m, 2) and from each other. Rejection sampling, first valid of `tries`."""
    pts = avoid
    out = []
    for _ in range(count):
        c = torch.tensor(lo, device=dev) + torch.rand(n, tries, 2, generator=g, device=dev) * torch.tensor(
            [hi[0] - lo[0], hi[1] - lo[1]], device=dev)
        ok = (torch.cdist(c, pts[..., :2]) >= pts[..., 2][:, None]).all(-1) if pts.shape[1] else \
            torch.ones(n, tries, dtype=torch.bool, device=dev)
        k = torch.where(ok.any(-1), ok.float().argmax(-1), torch.full((n,), tries - 1, device=dev))
        p = c[torch.arange(n, device=dev), k]
        out.append(p)
        pts = torch.cat([pts, torch.cat([p, torch.full((n, 1), gap, device=dev)], -1)[:, None]], 1)
    return torch.stack(out, 1)


class Cage:
    def __init__(self, n, seed=0, p_new=P_NEW, drift=DRIFT, scale_sd=SCALE_SD, obstacles=True, dev=DEV):
        self.n, self.dev, self.seed, self.p_new, self.drift, self.scale_sd = n, dev, seed, p_new, drift, scale_sd
        self.obstacles = obstacles

    def reset(self):
        n, dev = self.n, self.dev
        g = self.g = torch.Generator(device=dev); g.manual_seed(self.seed)
        U = lambda *s: torch.rand(*s, generator=g, device=dev)
        hub = torch.tensor([[HUB[0], HUB[1], 0.45]], device=dev).expand(n, 1, 3)       # [x, y, keep-clear radius]
        self.pads = _place(g, n, N_PADS, (1.3, 0.3), (2.7, 1.7), hub, 0.45, dev=dev)      # (n, P, 2)
        keep = torch.cat([hub, torch.cat([self.pads, torch.full((n, N_PADS, 1), 0.30, device=dev)], -1)], 1)
        hc = _place(g, n, N_HOUSES, (0.8, 0.2), (2.8, 1.8), keep, 0.35, dev=dev)
        self.houses = torch.cat([hc, 0.08 + 0.10 * U(n, N_HOUSES, 2), 0.15 + 0.35 * U(n, N_HOUSES, 1)], -1)   # cx cy hx hy h
        keep = torch.cat([keep, torch.cat([hc, torch.full((n, N_HOUSES, 1), 0.30, device=dev)], -1)], 1)
        tc = _place(g, n, N_TREES, (0.8, 0.15), (2.85, 1.85), keep, 0.25, dev=dev)
        r = 0.05 + 0.07 * U(n, N_TREES, 1)
        self.trees = torch.cat([tc, r, 0.35 + 0.85 * U(n, N_TREES, 1)], -1)             # cx cy r h
        if not self.obstacles:                                                           # (self-checks)
            self.houses[..., 4] = -1.0; self.trees[..., 3] = -1.0
        # one pole the map lacks (a lamp post, a crane), somewhere on the way to the customer
        a, side = 0.3 + 0.4 * U(n, 1), 0.08 * torch.randn(n, 1, generator=g, device=dev)
        hub2 = torch.tensor(HUB, device=dev)
        line = self.pads[:, 0] - hub2
        perp = torch.stack([-line[:, 1], line[:, 0]], -1) / line.norm(dim=-1, keepdim=True)
        pc = hub2 + a * line + side * perp
        self.pole = torch.cat([pc, torch.full((n, 1), 0.04, device=dev), torch.full((n, 1), 1.75, device=dev)], -1)
        self.pole_on = U(n) < self.p_new
        near_pad = torch.cdist(pc[:, None], self.pads).squeeze(1).min(-1).values < 0.3  # never on top of a pad
        self.pole_on &= ~near_pad
        self.p = torch.tensor([[HUB[0], HUB[1], Z_TAKEOFF]], device=dev).repeat(n, 1)
        self.v = torch.zeros(n, 3, device=dev)
        self.e = torch.zeros(n, 3, device=dev)                                            # believed - true position
        self.k = self.scale_sd * torch.randn(n, 2, generator=g, device=dev)
        self.q = torch.zeros(LAG, n, 3, device=dev)                                       # commands in flight
        self.landing = torch.zeros(n, dtype=torch.bool, device=dev)
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.outcome = torch.zeros(n, dtype=torch.long, device=dev)
        self.t_fix = torch.zeros(n, device=dev); self.d_fix = torch.zeros(n, device=dev)
        self.pole_seen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t, self.t_done = 0.0, torch.full((n,), T_MAX, device=dev)
        return self.obs()

    # ---- what the drone can know -------------------------------------------------------------------
    def view(self, p):
        """Markers in the mirror strip, their IDs readable: (n, P) bool, and their offset from the drone."""
        rel = self.pads - p[:, None, :2]
        z = p[:, 2:3].clamp(min=0.05)
        inside = (rel[..., 0].abs() < z * TAN_ALONG - MARKER / 2) & (rel[..., 1].abs() < z * TAN_ACROSS - MARKER / 2)
        return inside & (MARKER * PX_AT_1M / z >= ID_PX), rel

    def sigma(self):
        """The drone's own estimate of its position error (1 sd): walk + speed-scale error since the last fix."""
        return torch.sqrt((FIX_SD + 0.01 * self.p[:, 2]) ** 2 + self.drift ** 2 * self.t_fix + (self.scale_sd * self.d_fix) ** 2)

    def obs(self):
        seen, rel = self.view(self.p)
        noise = 0.005 * torch.randn(rel.shape, generator=self.g, device=self.dev)       # the flight's own stream
        return dict(b=self.p + self.e, v=self.v, sig=self.sigma(), seen=seen, rel=rel + noise,
                    pole=torch.where((self.pole_seen & self.pole_on)[:, None], self.pole, torch.full_like(self.pole, 1e3)))

    # ---- physics ------------------------------------------------------------------------------------
    def step(self, cmd, land=None):
        """cmd (n, 3) velocity; land (n,) bool starts the SDK's `land` (from then on it only descends)."""
        live = ~self.done
        if land is not None:
            self.landing |= land & live
        cmd = torch.where(self.landing[:, None], torch.tensor([0.0, 0.0, -V_LAND], device=self.dev), cmd)
        self.q = torch.cat([self.q[1:], cmd[None]])
        c = self.q[0]
        self.v[:, :2] += (c[:, :2] - self.v[:, :2]) * (DT / TAU_XY)
        self.v[:, 2] += (c[:, 2] - self.v[:, 2]) * (DT / TAU_Z)
        self.v = torch.where(self.landing[:, None], torch.tensor([0.0, 0.0, -V_LAND], device=self.dev), self.v)   # `land`: its own descent
        self.v *= live[:, None]
        self.p = self.p + self.v * DT
        # dead reckoning: the believed position drifts, and speed is misjudged by k
        step = torch.cat([self.k * self.v[:, :2] * DT + self.drift * math.sqrt(DT) * torch.randn(self.n, 2, generator=self.g, device=self.dev),
                          torch.zeros(self.n, 1, device=self.dev)], -1)
        self.e = self.e + step * live[:, None]
        self.t_fix += DT * live; self.d_fix += self.v[:, :2].norm(dim=-1) * DT
        seen, _ = self.view(self.p)
        fix = seen.any(-1) & live                                        # a surveyed marker read: position reset
        noise = torch.cat([(FIX_SD + 0.01 * self.p[:, 2:3]) * torch.randn(self.n, 2, generator=self.g, device=self.dev),
                           torch.zeros(self.n, 1, device=self.dev)], -1)
        self.e = torch.where(fix[:, None], noise, self.e)
        self.t_fix = torch.where(fix, 0.0, self.t_fix); self.d_fix = torch.where(fix, 0.0, self.d_fix)
        d = self.pole[:, :2] - self.p[:, :2]
        self.pole_seen |= self.pole_on & (d[:, 0] > 0) & (d[:, 0] < FWD_R) & (d[:, 1].abs() < d[:, 0] * FWD_TAN)
        self.t += DT
        # how it ends
        hit = self.hits(self.p, R_DRONE) | ((self.p[:, 2] < 0.03) & ~self.landing)     # or the floor, not landing
        net = (self.p[:, 0] < R_DRONE) | (self.p[:, 0] > L - R_DRONE) | (self.p[:, 1] < R_DRONE) | \
              (self.p[:, 1] > B - R_DRONE) | (self.p[:, 2] > H - 0.05)
        down = self.landing & (self.p[:, 2] <= 0.02)
        xy = self.p[:, :2] + LAND_SD * torch.randn(self.n, 2, generator=self.g, device=self.dev)
        dist = (self.pads - xy[:, None]).norm(dim=-1)
        oc = torch.where(dist[:, 0] < PAD_OK, 1, torch.where((dist[:, 1:] < PAD_OK).any(-1), 2, 3))
        new = torch.zeros_like(self.outcome)
        new = torch.where(down, oc, new)
        new = torch.where(net & live, 5, new)
        new = torch.where(hit & live, 4, new)
        new = torch.where(live & (new == 0) & (self.t >= T_MAX - 1e-6), 6, new)
        ended = live & (new > 0)
        self.outcome = torch.where(ended, new, self.outcome)
        self.t_done = torch.where(ended, torch.full_like(self.t_done, self.t), self.t_done)
        self.done |= ended
        return self.obs()

    def hits(self, p, r, pole=None, grow=0.0):
        """Would a drone of radius r at p (..., 3) touch a house, a tree or the pole? p may carry extra
        leading dims after n (rollouts); grow widens every obstacle (a planner's margin)."""
        ex = (slice(None),) + (None,) * (p.dim() - 2)
        h = self.houses[ex]; t = self.trees[ex]
        pq, g = p[..., None, :], (grow[ex + (None,)] if torch.is_tensor(grow) else grow)
        dx = (pq[..., 0] - h[..., 0]).abs() - h[..., 2] - r - g
        dy = (pq[..., 1] - h[..., 1]).abs() - h[..., 3] - r - g
        house = ((dx < 0) & (dy < 0) & (pq[..., 2] < h[..., 4] + 0.03)).any(-1)
        tree = (((pq[..., :2] - t[..., :2]).norm(dim=-1) < t[..., 2] + r + g) & (pq[..., 2] < t[..., 3])).any(-1)
        pl = self.pole if pole is None else pole
        pole_hit = ((p[..., :2] - pl[ex][..., :2]).norm(dim=-1) < pl[ex][..., 2] + r + (g[..., 0] if torch.is_tensor(g) else g)) \
            & (p[..., 2] < pl[ex][..., 3])
        on = self.pole_on[ex] if pole is None else torch.ones_like(pole_hit)
        return house | tree | (pole_hit & on)

    def summary(self):
        return dict(outcome=self.outcome, delivered=self.outcome == 1, t_done=self.t_done)


# ---- deciders ------------------------------------------------------------------------------------------
class Rule:
    """Climb to the transit height, fly straight at the customer's pad on the map, and once its marker is
    read in the mirror, centre on it, descend and land. Not found where the map says: sweep along the narrow
    axis of the mirror strip. Blind to anything the map lacks."""
    SWEEP = torch.tensor([0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9])

    def reset(self, env):
        self.env, self.k = env, 0
        self.leg = torch.zeros(env.n, dtype=torch.long, device=env.dev)       # search leg index

    def target(self, o):
        """Where the rule is heading (believed frame), whether it is locked on the customer's marker."""
        env = self.env
        lock = o["seen"][:, 0]
        goal = env.pads[:, 0] + torch.stack([self.SWEEP.to(env.dev)[self.leg % 7], torch.zeros_like(self.leg, dtype=torch.float)], -1)
        at = (goal - o["b"][:, :2]).norm(dim=-1) < 0.08
        self.leg = torch.where(at & ~lock, self.leg + 1, self.leg)
        return goal, lock

    def act(self, o):
        """-> (cmd (n, 3), land (n,) bool)"""
        self.k += 1
        goal, lock = self.target(o)
        b, rel = o["b"], o["rel"][:, 0]
        self.goal, self.aim = goal, torch.where(lock[:, None], b[:, :2] + rel, goal)   # the point it steers to
        to = goal - b[:, :2]
        d = to.norm(dim=-1, keepdim=True)
        cmd_xy = to / d.clamp(min=1e-6) * torch.minimum(d / 0.8, torch.full_like(d, V_XY))
        cmd_xy = torch.where(lock[:, None], rel / 0.8, cmd_xy).clamp(-V_XY, V_XY)   # locked: centre on the marker
        low = (b[:, 2] < Z_CRUISE - 0.1) & ~lock & (d[:, 0] > 0.15)
        cmd_xy = torch.where(low[:, None], torch.zeros_like(cmd_xy), cmd_xy)     # climb before transit
        centred = lock & (rel.norm(dim=-1) < 0.05)
        vz = torch.where(lock, torch.where(centred, -V_Z * torch.ones_like(d[:, 0]), torch.zeros_like(d[:, 0])),
                         ((Z_CRUISE - b[:, 2]) / 0.5).clamp(-V_Z, V_Z))
        land = centred & (b[:, 2] <= Z_FLARE)
        return torch.cat([cmd_xy, vz[:, None]], -1), land


class Planner(Rule):
    """Shield: fly the rule's command unless a 2.5 s rollout of it touches a known obstacle -- the map, plus
    the pole once the forward picture has shown it -- widened by the drone's own position uncertainty; then
    the safe candidate that gains most on the rule's goal. info="full": the true position and the pole known
    from takeoff (the all-knowing planner). Candidates: the rule's command, eight compass headings, hover,
    climb, descend, the rule's heading while climbing, land -- 14, the same menu the learner chooses from."""
    N, T = 14, 25

    def __init__(self, info="obs"):
        self.info = info

    def candidates(self, o, rule_cmd):
        n, dev = rule_cmd.shape[0], rule_cmd.device
        ang = torch.arange(8, device=dev) * (math.pi / 4)
        comp = torch.stack([V_XY * torch.cos(ang), V_XY * torch.sin(ang), torch.zeros(8, device=dev)], -1)
        up, dn = torch.tensor([0.0, 0.0, V_Z], device=dev), torch.tensor([0.0, 0.0, -V_Z], device=dev)
        climb_on = rule_cmd.clone(); climb_on[:, 2] = V_Z
        C = torch.cat([rule_cmd[:, None], comp.expand(n, 8, 3), torch.zeros(n, 1, 3, device=dev), up.expand(n, 1, 3),
                       dn.expand(n, 1, 3), climb_on[:, None], torch.tensor([0.0, 0.0, -V_LAND], device=dev).expand(n, 1, 3)], 1)
        return C

    def rollout(self, b, v, C):
        """Positions (n, 14, T, 3) under each candidate held for T steps, the drone's own velocity lag. The
        rule's two candidates keep steering to its aim point, slowing on arrival as the rule does (held
        constant they would fly on through the pad and into the net, and the shield would never approach)."""
        vv, pp, out = v[:, None].expand_as(C).clone(), b[:, None].expand_as(C).clone(), []
        tau = torch.tensor([TAU_XY, TAU_XY, TAU_Z], device=C.device)
        C = C.clone()
        for _ in range(self.T):
            for j in (0, 12):
                to = self.aim - pp[:, j, :2]
                d = to.norm(dim=-1, keepdim=True)
                C[:, j, :2] = to / d.clamp(min=1e-6) * torch.minimum(d / 0.8, torch.full_like(d, V_XY)) * (C[:, j, :2].norm(dim=-1, keepdim=True) > 1e-6)
            vv = vv + (C - vv) * (DT / tau)
            vv[:, 13] = C[:, 13]                                    # `land` descends at once, straight down
            pp = pp + vv * DT
            out.append(pp)
        return torch.stack(out, 2)

    def act(self, o):
        env = self.env
        if self.info == "full":
            o = dict(o, b=env.p, sig=torch.zeros_like(o["sig"]),
                     pole=torch.where(env.pole_on[:, None], env.pole, torch.full_like(env.pole, 1e3)))
        cmd, land = Rule.act(self, o)
        C = self.candidates(o, cmd)
        P = self.rollout(o["b"], o["v"], C)
        margin = (2 * o["sig"]).clamp(max=0.10)
        hit = env.hits(P, R_DRONE, pole=o["pole"], grow=margin)                        # (n, 14, T)
        net = (P[..., 0] < R_DRONE) | (P[..., 0] > L - R_DRONE) | (P[..., 1] < R_DRONE) | (P[..., 1] > B - R_DRONE) \
            | (P[..., 2] > H - 0.10) | ((P[..., 2] < 0.05) & (torch.arange(self.N, device=P.device) != 13)[None, :, None])
        bad = hit | net
        first = torch.where(bad.any(-1), bad.float().argmax(-1), torch.full(bad.shape[:2], self.T, device=P.device))
        safe = first == self.T
        safe[:, 13] &= land                                          # land only where the rule would land
        gain = (self.goal - o["b"][:, :2]).norm(dim=-1)[:, None] - (self.goal[:, None] - P[:, :, -1, :2]).norm(dim=-1)
        score = torch.where(safe, gain, -1e3 + first.float())       # none safe: the one that hits latest
        pick = torch.where(safe[:, 0], torch.zeros_like(first[:, 0]), score.argmax(-1))
        pick = torch.where((pick == 0) & land, 13, pick)            # the rule landing IS the land candidate
        self.choice, self.C, self.first, self.gain, self.safe, self.land_ok = pick, C, first, gain, safe, land
        ar = torch.arange(C.shape[0], device=C.device)
        return C[ar, pick], pick == 13


def run(env, agent, record=False):
    o = env.reset(); agent.reset(env)
    cmd, land = torch.zeros(env.n, 3, device=env.dev), torch.zeros(env.n, dtype=torch.bool, device=env.dev)
    k, path = 0, []
    while not bool(env.done.all()) and env.t < T_MAX - 1e-6:
        if k % HOLD == 0:
            cmd, land = agent.act(o)
        o = env.step(cmd, land if k % HOLD == 0 else None)
        if record:
            path.append(env.p.clone())
        k += 1
    s = env.summary()
    return (s, torch.stack(path, 1)) if record else s


def table(s):
    oc = s["outcome"]
    return "  ".join(f"{name} {100 * (oc == i).float().mean().item():5.1f}%" for i, name in enumerate(OUTCOMES) if i)


def demo():
    torch.manual_seed(0)
    # 1. perfect knowledge, empty cage: the rule delivers every time
    s = run(Cage(512, seed=1, p_new=0.0, drift=0.0, scale_sd=0.0, obstacles=False), Rule())
    oc = s["outcome"]                                  # every flight lands; only `land`'s own scatter misses the pad
    assert ((oc == 1) | (oc == 3)).all() and (oc == 1).float().mean() > 0.9, table(s)
    # 2. the pole on the straight line, and the rule is blind to it: it must hit it sometimes
    s = run(Cage(512, seed=2, p_new=1.0), Rule())
    assert (s["outcome"] == 4).float().mean() > 0.2, table(s)
    # 3. a marker directly below is read; one outside the strip is not
    e = Cage(4, seed=3); e.reset()
    p = torch.cat([e.pads[:, 0], torch.ones(4, 1, device=e.dev)], -1)
    assert e.view(p)[0][:, 0].all()
    p[:, 0] += 0.3                                                             # 0.3 m along the narrow axis at 1 m
    assert not e.view(p)[0][:, 0].any()
    print("self-checks passed")


if __name__ == "__main__":
    demo()
    for name, agent in (("rule", Rule()), ("planner", Planner()), ("all-knowing", Planner("full"))):
        s = run(Cage(2048, seed=7), agent)
        print(f"{name:<12}", table(s), f" median time {s['t_done'][s['outcome'] == 1].median().item():.0f} s")
