#!/usr/bin/env python3
"""
approach_env_v5.py -- trees. The point where the rule stops being derivable.

WHY THIS ENVIRONMENT EXISTS

Everything up to v4 was a problem with a closed-form answer. The pad's position
is unknown but its error is zero-mean and weakly correlated, so the optimal
estimate is the running mean of the fixes -- textbook, and a one-line rule using
it beat every trained policy we produced. Asking a network to rediscover an
average by trial and error is the wrong use of the tool, and it lost by 4 points.

Trees break that. Three ways, and none of them has a formula:

  1. A tree between the camera and the pad HIDES the marker even when the pad
     is geometrically inside the field of view. Coverage stops equalling
     visibility, so a search pattern computed from camera geometry is wrong.
  2. Where the marker is visible from depends on which SIDE you approach from,
     because looking down past a trunk is a line-of-sight problem.
  3. The layout is discovered while flying, not known in advance, so the plan
     has to be revised against information that arrives mid-flight.

You cannot write "search a box of radius k*sigma" for that. You can write a
reactive rule -- and RuleAvoid in rule_agent.py is a good one -- but it is a
heuristic with no optimality claim, which is exactly the regime where a learned
policy has something to win.

SENSING, NOT A MAP

The drone gets range readings in eight sectors out to 30 m, the way a depth
camera or a small lidar would report. It is NOT given obstacle positions. That
matters for the argument: if the layout were known up front, a planner could
solve it analytically and we would be back where we started.
"""

import math

import numpy as np
from gymnasium import spaces

from approach_env_v4 import ApproachEnvV4


class ApproachEnvV5(ApproachEnvV4):

    N_OBST = (3, 6)              # trees per garden
    OBST_R = (0.8, 1.8)          # TRUNK radius, m; canopy is wider
    OBST_H = (6.0, 14.0)         # height, m
    OBST_SPAN = 22.0             # scattered within this of the pad
    PAD_CLEAR = 4.0              # no tree closer than this to the pad itself
    SENSE_R = 30.0               # how far the depth sensor sees
    N_SECTORS = 8
    R_CRASH = -150.0             # worse than any bad landing: the airframe is gone

    # How the obstacles are reported to the policy.
    #   "sectors"  8 range readings, what v5 has always given: bearing-binned
    #              nearest distance, exactly what a cheap ranger or a coarse
    #              depth map provides.
    #   "rich"     the K nearest obstacles as (dx, dy, radius, height) relative
    #              vectors -- what a stereo depth camera plus a little
    #              segmentation would actually recover at close range.
    # The whole point of comparing them is that the policy ALREADY had
    # "sectors" and still lost to a hand-written rule by 3.58 pp. Before
    # building a new simulator to supply richer sensing, measure whether
    # richer sensing carries anything the sectors did not.
    SENSE_MODE = "sectors"
    N_RICH = 4

    # CANOPIES, NOT BOLLARDS.
    #
    # The first obstacle model was a cylinder from the ground up, placed clear
    # of the pad. Measured against it, a policy that ignored obstacles entirely
    # crashed 0.3% of the time over 300 episodes -- it passed within 0.6 m of a
    # trunk at the 5th percentile and still never hit one, because a vertical
    # descent onto a point that is not inside a trunk never intersects one.
    # There was nothing for any sensor to recover, which is why giving the
    # policy range readings changed nothing.
    #
    # A real tree is not a bollard. It is a thin trunk under a wide canopy, and
    # the canopy OVERHANGS -- which is what actually obstructs a descent. With
    # canopies the naive vertical approach can be blocked outright, so the
    # aircraft has to find a gap, come down through it, and move in underneath;
    # and sometimes no gap exists and the correct answer is to abort. That is a
    # decision that depends on what it can see.
    CANOPY = True
    TRUNK_R = (0.8, 1.8)       # the part that reaches the ground
    CANOPY_R = (3.5, 7.0)      # the part that overhangs
    CANOPY_FRAC = 0.55         # canopy occupies the top 45% of the tree

    def __init__(self, *a, obstacles=True, sense_mode=None, **kw):
        self.obstacles = bool(obstacles)
        if sense_mode:
            self.SENSE_MODE = sense_mode
        # The parent's reset() builds an observation before this class has had
        # a chance to place anything, so the list has to exist from the start.
        self.obst = []
        super().__init__(*a, **kw)
        if self.obstacles:
            extra = (self.N_SECTORS if self.SENSE_MODE == "sectors"
                     else 4 * self.N_RICH)
            n = self.observation_space.shape[0] + extra
            self.observation_space = spaces.Box(-np.inf, np.inf, (n,), np.float32)

    # ---- layout -----------------------------------------------------------
    def _place_obstacles(self):
        r = self.rng
        n = int(r.integers(self.N_OBST[0], self.N_OBST[1] + 1))
        out = []
        for _ in range(n):
            for _try in range(20):
                th = r.uniform(0, 2 * np.pi)
                d = r.uniform(self.PAD_CLEAR, self.OBST_SPAN)
                p = self.pad + d * np.array([np.cos(th), np.sin(th)])
                rad = r.uniform(*self.OBST_R)
                # trunks stay off the pad; canopies are allowed to reach over it
                if np.linalg.norm(p - self.pad) > self.PAD_CLEAR + rad:
                    out.append((p, rad, r.uniform(*self.OBST_H)))
                    break
        self.obst = out

    # ---- physics ----------------------------------------------------------
    def _blocked(self):
        """True if a tree stands between the camera and the pad."""
        if not self.obstacles or not getattr(self, 'obst', []):
            return False
        a = np.array([self.xy[0], self.xy[1], self.h])
        b = np.array([self.pad[0], self.pad[1], 0.0])
        # Sample the sight line. Cheap and exact enough: a canopy is metres
        # across and the line is tens of metres long.
        for t in np.linspace(0.0, 1.0, 12):
            p = a + t * (b - a)
            for c, rad, hh in self.obst:
                if p[2] < hh and np.linalg.norm(p[:2] - c) < rad:
                    return True
        return False

    def _canopy_r(self, rad):
        """Canopy radius for a trunk of this radius."""
        lo, hi = self.CANOPY_R
        t = (rad - self.TRUNK_R[0]) / max(self.TRUNK_R[1] - self.TRUNK_R[0], 1e-6)
        return lo + t * (hi - lo)

    def _crashed(self):
        if not self.obstacles:
            return False
        for c, rad, hh in getattr(self, 'obst', []):
            d = float(np.linalg.norm(self.xy - c))
            if self.h < hh and d < rad:
                return True                      # into the trunk
            if self.CANOPY and self.CANOPY_FRAC * hh < self.h < hh \
                    and d < self._canopy_r(rad):
                return True                      # into the canopy
        return False

    def descent_blocked(self, xy=None):
        """Is a vertical descent at this point obstructed by a canopy?"""
        p = self.xy if xy is None else np.asarray(xy, float)
        if not (self.obstacles and self.CANOPY):
            return False
        for c, rad, hh in getattr(self, 'obst', []):
            if float(np.linalg.norm(p - c)) < self._canopy_r(rad):
                return True
        return False

    def _acquired(self):
        # The marker is only decoded if it is BOTH geometrically in view and
        # not hidden behind something. This is the whole point of the env.
        return bool(super()._acquired()) and not self._blocked()

    # ---- sensing ----------------------------------------------------------
    def _body_r(self, rad, hh):
        """
        Radius of the part of the tree at OUR CURRENT ALTITUDE.

        This was reporting the trunk radius at every height, which is the part
        the aircraft does not hit. Above CANOPY_FRAC * hh the body that kills
        you is the canopy, 2-3 m wider. A depth sensor pointed at a tree from
        inside the canopy layer returns the canopy, so reporting the trunk
        understated every range by the overhang and handicapped the cheap
        sensor for a reason that is bookkeeping, not physics.
        """
        if self.CANOPY and self.CANOPY_FRAC * hh < self.h < hh:
            return self._canopy_r(rad)
        return rad

    def _ranges(self):
        """Distance to the nearest obstacle per sector, like a depth camera."""
        out = np.full(self.N_SECTORS, self.SENSE_R, float)
        if not self.obstacles:
            return out / self.SENSE_R
        for c, rad, hh in getattr(self, 'obst', []):
            d = c - self.xy
            rng = float(np.linalg.norm(d)) - self._body_r(rad, hh)
            if rng > self.SENSE_R or self.h > hh + 2.0:
                continue                     # above it: not an obstacle to us
            s = int(((math.atan2(d[1], d[0]) + 2 * np.pi) % (2 * np.pi))
                    / (2 * np.pi) * self.N_SECTORS) % self.N_SECTORS
            out[s] = min(out[s], max(rng, 0.0))
        return out / self.SENSE_R

    def _rich(self):
        """The K nearest obstacles as relative vectors, stereo-depth style."""
        out = np.zeros((self.N_RICH, 4), float)
        obs = getattr(self, "obst", [])
        if obs:
            d = sorted(((float(np.linalg.norm(c - self.xy)), c, r, h)
                        for c, r, h in obs), key=lambda t: t[0])[:self.N_RICH]
            for i, (dist, c, r, h) in enumerate(d):
                if dist - r > self.SENSE_R:
                    continue
                v = (c - self.xy) / self.SENSE_R
                out[i] = [np.clip(v[0], -2, 2), np.clip(v[1], -2, 2),
                          self._body_r(r, h) / 10.0, h / 20.0]
        return out.ravel()

    def _obs(self):
        o = super()._obs()
        if not self.obstacles:
            return o
        extra = (self._ranges() if self.SENSE_MODE == "sectors" else self._rich())
        return np.concatenate([o, extra.astype(np.float32)]).astype(np.float32)

    # ---- episode ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options)
        self._place_obstacles()
        return self._obs(), i

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        info["blocked"] = bool(self._blocked())

        # LANDING UNSEEN IS NOT A DELIVERY.
        #
        # Without this the trees are decorative. The pad is guaranteed clear of
        # trunks, so "fly to the averaged fix and descend" never crashes and
        # never needs to see anything -- a tree-blind rule scored 98.8% and
        # occlusion cost nothing at all. The real aircraft is not allowed to do
        # that: px4_mission.py aborts rather than land on a marker it has never
        # decoded, because putting a parcel down blind is how it ends up in a
        # stranger's garden. Enforcing the same rule here is what finally makes
        # a tree between the camera and the pad expensive, and makes WHERE YOU
        # LOOK FROM a decision instead of a detail.
        if term and info.get("success") and not self.was_acquired:
            info["success"] = False
            info["outcome"] = "landed_unseen"
        info["n_obst"] = len(getattr(self, "obst", []))
        if self.obstacles and self._crashed():
            r += self.R_CRASH
            info["outcome"] = "crash"
            info["success"] = False
            return obs, float(r), True, False, info
        return obs, r, term, trunc, info


def demo(pool="sigma_pool.npz"):
    """Checks the trees actually do something, before anything trains on them."""
    kw = dict(pool=pool, seed=0, t_max=60, rho=0.707, sigma_scale=1.75)
    e = ApproachEnvV5(arm="sigma", **kw)
    o, _ = e.reset(seed=0)
    assert o.shape == (17,), f"expected 9 + 8 sectors, got {o.shape}"
    print(f"1. observation is {o.shape[0]} channels (9 + {e.N_SECTORS} range readings)")

    # 2. occlusion must hide the marker in states where it WOULD otherwise be
    #    visible. Measuring "blocked" over states where the pad is nowhere near
    #    the field of view says nothing -- the quantity that matters is
    #    P(hidden | geometrically in view).
    inview = hidden = 0
    for s in range(400):
        e.reset(seed=s)
        e.h = float(e.rng.uniform(6.0, 18.0))
        e.xy = e.pad + e.rng.normal(0, 6.0, 2)
        if super(ApproachEnvV5, e)._acquired():
            inview += 1
            hidden += bool(e._blocked())
    print(f"2. of {inview} states with the pad geometrically in view, "
          f"{100*hidden/max(inview,1):.0f}% are hidden behind a tree")
    assert inview > 50, "test never got the pad in view"
    assert hidden > 5, "trees never hide the marker -- occlusion is inert"

    # 3. hitting a tree must be possible. Two checks: the geometry itself, and
    #    whether a policy that traverses the garden low actually does it.
    e.reset(seed=1)
    c, rad, hh = e.obst[0]
    e.xy, e.h = c.copy(), hh * 0.5
    assert e._crashed(), "sitting inside a trunk is not registering as a crash"
    e.h = hh + 5.0
    assert not e._crashed(), "flying ABOVE a tree counts as a crash"
    print("3. crash geometry: inside a trunk crashes, above the canopy does not")

    crashes = trials = 0
    for s in range(200):
        e.reset(seed=s)
        e.h, e.xy = 5.0, e.pad + np.array([20.0, 0.0])
        trials += 1
        for _ in range(150):
            vz = 0.0 if e.h > 4.0 else 0.3          # hold low, traverse
            _, _, t, tr, info = e.step(np.array([vz, 1.0, 0.0]))
            if t or tr:
                crashes += info.get("outcome") == "crash"
                break
    print(f"4. traversing the garden at 5 m hits a tree in "
          f"{100*crashes/trials:.0f}% of episodes")
    assert crashes > 5, "cannot hit a tree while flying through one"

    print("\nobstacles verified")


if __name__ == "__main__":
    demo()
