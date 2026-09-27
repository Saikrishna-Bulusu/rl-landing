#!/usr/bin/env python3
"""
local_planner.py -- a local 3D planner on what the drone knows: A* through a window around it, on the
survey plus the voxel map (both as the tops of 1 m columns: tops(x, y) -> metres). It turns the rule's
straight line into a path that keeps clear of everything on the map; static safety then rests on search,
not on anything learned.

  cells      1 m square, DZ m tall, from the ground to Z_TOP, WIN m either side of the drone
  blocked    below a column's top + Z_PAD, and within R_PAD of such a column sideways (the x500 is 0.9 m
             across; the fast simulator scores a point)
  cost       length, climbing and descending 2x (going round beats going over), and a clearance
             penalty inside C_CLEAR of anything blocked (distance field, scipy)
  goal       the rule's target at the rule's altitude; outside the window, where the line to it leaves
             the window (the nearest free cell if that one is blocked)
  follow     a carrot LOOK m along the path; speed eased near obstacles

    python local_planner.py        self-checks: a mast on the line is passed at >= R_PAD, a free line is straight
"""
import heapq
import math

import numpy as np
from scipy import ndimage

WIN, DZ, Z_TOP = 30, 2.0, 90.0
Z_PAD, R_PAD, C_CLEAR, W_CLEAR = 2.0, 2.0, 5.0, 3.0
LOOK, V_NEAR = 6.0, 3.0
# Speed the depth camera can see for: at 8 m/s a drone needs ~10-14 m to react and stop, more than an 8-12 m
# OAK-D gives it -- Gazebo seeds 3-4 hit a crane 42-46 m out while still at cruise (25 Sept). 6 m/s stops in ~9 m.
V_SENSE = 6.0
W_H = 1.5                  # weighted A*, paths within 1.5x the best: a plan from 75 m went 2 s -> 17 ms (27 Sept)


class LocalPlanner:
    def __init__(self, win=WIN, dz=DZ, z_top=Z_TOP):
        self.win, self.dz = win, dz
        self.nz = int(z_top / dz)
        self.path = None                                     # last path, map frame (for the video's ribbon)
        self.clear_ahead = np.inf                            # least clearance along the next 10 m of it
        # 26 neighbours and their lengths (x, y in 1 m cells, z in dz)
        d = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
        self.nbr = d
        self.step = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + (2.0 * dz * d[:, 2]) ** 2)   # vertical moves cost double

    def grid(self, me_xy, tops):
        """Blocked cells and the clearance cost, for the window centred on the 1 m cell holding me_xy."""
        n = 2 * self.win + 1
        c0 = np.floor(me_xy).astype(int) - self.win
        xs, ys = c0[0] + np.arange(n) + 0.5, c0[1] + np.arange(n) + 0.5
        top = tops(np.stack(np.meshgrid(xs, ys, indexing="ij"), -1))          # (n, n) metres
        z = (np.arange(self.nz) + 0.5) * self.dz
        solid = z[None, None, :] < top[..., None] + Z_PAD
        r = int(math.ceil(R_PAD))
        k = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= R_PAD + 1e-6
        blocked = ndimage.binary_dilation(solid, structure=k[..., None])
        blocked[..., 0] = True                                                 # never plan along the ground
        clear = ndimage.distance_transform_edt(~blocked, sampling=(1.0, 1.0, self.dz))
        self.dist = clear
        pen = 1.0 + W_CLEAR * np.clip((C_CLEAR - clear) / C_CLEAR, 0.0, 1.0)
        return blocked, pen, c0

    def plan(self, me, goal, tops):
        """me, goal: (x, y, z) in the map frame. -> [(x, y, z)] from me towards goal, or None."""
        me, goal = np.asarray(me, float), np.asarray(goal, float)
        blocked, pen, c0 = self.grid(me[:2], tops)
        n = blocked.shape[0]
        cell = lambda p: (int(np.clip(math.floor(p[0]) - c0[0], 0, n - 1)), int(np.clip(math.floor(p[1]) - c0[1], 0, n - 1)),
                          int(np.clip(p[2] // self.dz, 1, self.nz - 1)))
        d = goal[:2] - me[:2]
        lim = self.win - 2
        if np.abs(d).max() > lim:                                              # leave the window towards it
            goal = np.r_[me[:2] + d * lim / np.abs(d).max(), goal[2]]
        s, g = cell(me), cell(goal)
        if blocked[s]:                                                         # inside a padded cell already: start
            blocked = blocked.copy(); blocked[s] = False                       # there, the search leads it out
        if blocked[g]:
            free = np.argwhere(~blocked)
            if not len(free):
                return None
            g = tuple(free[np.argmin(((free - np.array(g)) * [1, 1, self.dz]) ** 2 @ np.ones(3))])
        # the step's own metric (a metre of height costs two): still admissible, and a 40 m descent no longer
        # floods the window -- with height counted once, the geo flight's plans from 75 m took 2 s each and the
        # answer to a crane seen 2.5 s out came too late (27 Sept, geo_obst4)
        h = lambda c: math.sqrt((c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2 + ((c[2] - g[2]) * 2.0 * self.dz) ** 2)
        best, prev, openq = {s: 0.0}, {s: None}, [(W_H * h(s), s)]        # W_H: weighted A*, paths within
        shape = blocked.shape
        while openq:
            f, c = heapq.heappop(openq)                                    # W_H of the best, a tenth of the work
            if c == g:
                break
            gc = best[c]
            if f - W_H * h(c) > gc + 1e-9:
                continue
            for (di, dj, dk), st in zip(self.nbr, self.step):
                q = (c[0] + di, c[1] + dj, c[2] + dk)
                if not (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1] and 0 < q[2] < shape[2]) or blocked[q]:
                    continue
                gq = gc + st * pen[q]
                if gq < best.get(q, np.inf):
                    best[q], prev[q] = gq, c
                    heapq.heappush(openq, (gq + W_H * h(q), q))
        if g not in prev:
            return None
        cells, c = [], g
        while c is not None:
            cells.append(c); c = prev[c]
        pts = [np.array([c0[0] + i + 0.5, c0[1] + j + 0.5, (k + 0.5) * self.dz]) for i, j, k in reversed(cells)]
        pts[0] = me.copy()
        new = self.smooth(pts, blocked, c0)
        # COMMIT TO A SIDE: the voxel map fills in a few columns at a time as the camera takes a mast in, and
        # each replan could pick the other side -- the first Gazebo flight dithered 6 s beside the crane. Keep
        # the old path (from where the drone is now) while it is still free and not much longer.
        # The kept path must end where the new one does -- the goal moves on with the drone; carrying a stale end
        # made the first fixed flight shuttle about the first plan's goal (25 Sept) -- so it is extended from its
        # end to the new goal, and kept only if that is free too.
        old = self.rest(me) if self.path is not None else None
        if old is not None:
            old = old if np.linalg.norm(old[-1] - new[-1]) < 1.0 else old + [new[-1]]
            if self.clear(old, blocked, c0) and self.length(old) <= 1.25 * self.length(new) + 2.0:
                new = old
        self.path = new
        near = [p for p in self.dense(new) if np.linalg.norm(p[:2] - me[:2]) < 10.0]
        n_ = np.array(near)
        ij = (np.floor(n_[:, 0]) - c0[0]).astype(int).clip(0, n - 1), (np.floor(n_[:, 1]) - c0[1]).astype(int).clip(0, n - 1), \
            (n_[:, 2] // self.dz).astype(int).clip(0, self.nz - 1)
        self.clear_ahead = float(self.dist[ij].min()) if len(near) else np.inf
        return self.path

    @staticmethod
    def dense(path, step=0.5):
        return np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / step) + 1)) for a, b in zip(path[:-1], path[1:])]) \
            if len(path) > 1 else np.array(path)

    @staticmethod
    def length(path):
        return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))

    def rest(self, me):
        """The last path from the point on it nearest the drone, starting at the drone."""
        d = self.dense(self.path)
        # nearest by HORIZONTAL distance: descending, the drone lags below the path's height profile, and a 3D
        # nearest point was one behind it -- the path turned back and the drone shuttled (25 Sept)
        k = int(np.argmin(np.linalg.norm(d[:, :2] - me[:2], axis=1)))
        off = me[:2] - d[k, :2]
        if np.linalg.norm(off) > 3.0 or k >= len(d) - 1:
            return None
        # THE PATH MOVES WITH THE DRONE (27 Sept). On GeoAnchor, not GPS, PX4's estimate is corrected in steps of a
        # metre or two, and the drone found itself 2.3 m beside its own committed path: rejoining it planned a detour
        # back, the carrot swung behind, and the command turned up to 180 deg within a second, over and over. Shifted
        # by that offset, the path keeps the side it committed to and starts ahead of the drone (on GPS the offset
        # is the tracking error, centimetres, and nothing changes).
        d = d.copy()
        d[:, :2] += off
        return [me.copy(), *d[k + 1:][::max(1, (len(d) - k) // 12)], d[-1]]

    def clear(self, path, blocked, c0):
        n = blocked.shape[0]
        for p in self.dense(path)[1:]:
            i, j, k = int(math.floor(p[0]) - c0[0]), int(math.floor(p[1]) - c0[1]), int(p[2] // self.dz)
            if not (0 <= i < n and 0 <= j < n and 0 <= k < self.nz) or blocked[i, j, k]:
                return False
        return True

    def smooth(self, pts, blocked, c0):
        """Drop waypoints while the straight line between the kept ones stays in free cells."""
        free = lambda a, b: all(not blocked[int(math.floor(p[0]) - c0[0]), int(math.floor(p[1]) - c0[1]),
                                            min(max(int(p[2] // self.dz), 0), self.nz - 1)]
                                for p in np.linspace(a, b, max(2, int(np.linalg.norm(b - a) * 2))))
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j]); i = j
        return out

    def carrot(self, me, path):
        """The point LOOK m along the path (the last point if it is shorter)."""
        left, a = LOOK, np.asarray(me, float)
        for b in path[1:]:
            seg = np.linalg.norm(b - a)
            if seg >= left:
                return a + (b - a) * left / max(seg, 1e-9)
            left -= seg; a = b
        return path[-1]


def demo():
    lp = LocalPlanner()
    # a 2 m square mast, 58 m tall, 12 m ahead on the line, the drone at 50 m flying +x
    mast = lambda xy: np.where((np.abs(xy[..., 0] - 12.0) <= 1.0) & (np.abs(xy[..., 1] - 0.3) <= 1.0), 58.0, 0.0)
    path = lp.plan((0.0, 0.0, 50.0), (60.0, 0.0, 50.0), mast)
    assert path is not None
    dense = np.concatenate([np.linspace(a, b, 50) for a, b in zip(path[:-1], path[1:])])
    clear = np.min(np.hypot(np.maximum(np.abs(dense[:, 0] - 12.0) - 1.0, 0), np.maximum(np.abs(dense[:, 1] - 0.3) - 1.0, 0)))
    assert clear >= R_PAD - 0.5, f"passed the mast at {clear:.2f} m"
    # nothing in the way: the straight line
    free = lp.plan((0.0, 0.0, 50.0), (20.0, 0.0, 50.0), lambda xy: np.zeros(xy.shape[:-1]))
    assert len(free) == 2, free
    # fly it: a drone following the carrot; the mast only enters the map within 8 m (the depth camera)
    lp, p, v, seen = LocalPlanner(), np.array([0.0, 0.0, 50.0]), np.array([8.0, 0.0, 0.0]), False
    xs = []
    for _ in range(60):
        seen |= abs(p[0] - 12.0) < 9.0
        tops = mast if seen else (lambda xy: np.zeros(xy.shape[:-1]))
        path = lp.plan(p, np.array([200.0, 0.0, 50.0]), tops)
        c = lp.carrot(p, path) - p
        v = v + (c / max(np.linalg.norm(c), 1e-6) * 5.0 - v) * 0.3
        p = p + v * 0.25
        xs.append(p.copy())
    xs = np.array(xs)
    assert np.all(np.diff(xs[:, 0]) > -0.05), "the drone went backwards"
    m = np.min(np.hypot(np.maximum(np.abs(xs[:, 0] - 12.0) - 1.0, 0), np.maximum(np.abs(xs[:, 1] - 0.3) - 1.0, 0)))
    assert m > 0.9, f"flew within {m:.2f} m of the mast"
    print(f"self-checks passed: mast passed at {clear:.1f} m via {len(path)} waypoints; free line straight; "
          f"flown: never backwards, mast seen late passed at {m:.1f} m, {xs[-1, 0]:.0f} m covered in 15 s")


if __name__ == "__main__":
    demo()
