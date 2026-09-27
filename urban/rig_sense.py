"""
rig_sense.py -- what the six-camera rig (gz_rig.py) tells the drone about the magpie, instead of
the exact position and speed urban_env's "omni" sense hands over.

Two parts, both batched over flights so the fast simulator and the Gazebo test run the same code:

  BirdTrack   a Kalman filter on the bird's position and velocity (constant velocity, the bird's
              steering as process noise), fed one measurement per step: a direction from the
              camera -- sharp -- and a distance from the bird's size in the image -- poor, because
              a magpie head-on and side-on differ ~2x in size. A track is reported only after two
              detections, and dropped after DROP_S without one.
  Rig         the fast simulator's stand-in for the cameras + detector, from MEASURED numbers
              (rig_calib.json, written from the Gazebo bench and the detector's held-out score):
              the chance of detection by the bird's size in pixels, the blind directions the
              drone's own body makes, line of sight past trees and roofs, the detector's
              direction and size errors. In Gazebo the real detector replaces it; the tracker stays.
"""
import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.environ.get("RIG_CALIB", f"{HERE}/rig_calib.json")    # RIG_CALIB: e.g. a pessimistic variant
SIGMA_A = 15.0      # bird's steering, m/s^2 of white-noise acceleration (the hunter turns at up to 20)
CONFIRM, DROP_S = 2, 1.0
HEAD_SD = math.radians(4.0)   # per-flight heading error of the drone's own estimate (one measurement: 5.8 deg)
FA_SIZE = (6.0, 40.0)   # px, clutter false alarms' apparent size (held-out set: roof corners, trunks, edges)
V_MIN = 3.0         # m/s: slower than this is not an attacking bird (static clutter, see BirdTrack.reported)
GATE_ANG, V_REL = math.radians(3.0), 25.0   # direction gate: 3 deg + a bird at 25 m/s relative moving across
GATE_MAX, TENT_S = math.radians(15.0), 0.15  # ...never wider; an unconfirmed sighting must be matched within 3 frames


class BirdTrack:
    def __init__(self, n, device):
        self.n, self.dev = n, device
        self.x = torch.zeros(n, 6, device=device)
        self.P = torch.eye(6, device=device).repeat(n, 1, 1)
        self.hits = torch.zeros(n, dtype=torch.long, device=device)
        self.t_last = torch.full((n,), -1e9, device=device)

    def reset(self, mask=None):
        m = torch.ones(self.n, dtype=torch.bool, device=self.dev) if mask is None else mask
        self.hits = torch.where(m, torch.zeros_like(self.hits), self.hits)
        self.t_last = torch.where(m, torch.full_like(self.t_last, -1e9), self.t_last)

    def predict(self, dt):
        F = torch.eye(6, device=self.dev); F[:3, 3:] = dt * torch.eye(3, device=self.dev)
        q = SIGMA_A ** 2
        Q = torch.zeros(6, 6, device=self.dev)
        Q[:3, :3] = q * dt ** 4 / 4 * torch.eye(3, device=self.dev); Q[:3, 3:] = Q[3:, :3] = q * dt ** 3 / 2 * torch.eye(3, device=self.dev)
        Q[3:, 3:] = q * dt ** 2 * torch.eye(3, device=self.dev)
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + Q

    def angle(self, z, me):
        """Angle between where a detection says the bird is and where the track says it should be,
        both seen from the drone at me -- direction is what the rig measures sharply (distance comes
        from size, ~37% off, so a position gate would pass almost any pair of clutter false alarms:
        both 'consistent' with a bird near the drone)."""
        a, b = z - me, self.x[:, :3] - me
        cos = (a * b).sum(-1) / (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)).clamp(min=1e-6)
        return torch.acos(cos.clamp(-1.0, 1.0))

    def allowed(self, t, me):
        """The gate: GATE_ANG, plus how far across the view a bird at V_REL can have moved since the
        track last took a detection, at the track's distance."""
        r = torch.linalg.norm(self.x[:, :3] - me, dim=-1).clamp(min=1.0)
        return (GATE_ANG + torch.atan(V_REL * (t - self.t_last).clamp(max=DROP_S) / r)).clamp(max=GATE_MAX)

    def update(self, t, det, z, R, me):
        """det (n,) bool: a detection this frame; z (n,3) measured bird position; R (n,3,3) its
        covariance; me (n,3) the drone. A confirmed track takes only detections in the direction it
        predicts (the gate) and ignores the rest -- the detector's clutter false alarms are scattered
        about the picture; an unconfirmed one is replaced by the newest sighting, so one false alarm
        cannot hold the real bird off for DROP_S."""
        fit = self.angle(z, me) < self.allowed(t, me)
        ok = det & (self.hits > 0) & fit
        new = det & ((self.hits == 0) | ((self.hits < CONFIRM) & ~fit))
        P0 = torch.zeros(self.n, 6, 6, device=self.dev); P0[:, :3, :3] = R; P0[:, 3:, 3:] = 15.0 ** 2 * torch.eye(3, device=self.dev)
        x0 = torch.cat([z, torch.zeros_like(z)], -1)          # fresh: velocity unknown (a magpie's speed)
        H = torch.zeros(3, 6, device=self.dev); H[:, :3] = torch.eye(3, device=self.dev)
        K = self.P[:, :, :3] @ torch.linalg.inv(self.P[:, :3, :3] + R)
        x1 = self.x + (K @ (z - self.x[:, :3])[..., None])[..., 0]
        P1 = (torch.eye(6, device=self.dev) - K @ H) @ self.P
        self.x = torch.where(new[:, None], x0, torch.where(ok[:, None], x1, self.x))
        self.P = torch.where(new[:, None, None], P0, torch.where(ok[:, None, None], P1, self.P))
        self.hits = torch.where(new, torch.ones_like(self.hits), torch.where(ok, self.hits + 1, self.hits))
        self.t_last = torch.where(ok | new, torch.full_like(self.t_last, t), self.t_last)
        self.reset((t - self.t_last > DROP_S) | ((self.hits < CONFIRM) & (t - self.t_last > TENT_S)))

    @property
    def confirmed(self):
        return self.hits >= CONFIRM

    @property
    def reported(self):
        """What the decider is told: a confirmed track that MOVES like a bird (over V_MIN). Landing
        pads' black-and-white markers, roof corners and tree shadows stand still, yet came through
        the detector often enough to be tracked as birds in Gazebo (25 Sept); a magpie that attacks
        is flying (12-18 m/s diving, up to 9 m/s repositioning). The price: a new dive is reported a
        few frames late, once its speed is estimated."""
        return self.confirmed & (torch.linalg.norm(self.x[:, 3:], dim=-1) > V_MIN)


def measure_cov(rel_dir, dist, sig_ang, sig_rel):
    """Covariance of a bird position from a direction (sharp: sig_ang rad) and a distance
    (poor: sig_rel of it) -- a long thin cigar along the line of sight."""
    b = rel_dir[..., :, None]
    along = b @ b.transpose(-1, -2)
    eye = torch.eye(3, device=rel_dir.device).expand_as(along)
    return (sig_rel * dist)[..., None, None] ** 2 * along + (sig_ang * dist)[..., None, None] ** 2 * (eye - along)


class Rig:
    """The fast simulator's cameras + detector, calibrated; owns the tracker and its own random stream."""

    def __init__(self, n, device, seed, calib=None):
        c = json.load(open(calib or CALIB))
        self.c, self.n, self.dev = c, n, device
        self.g = torch.Generator(device).manual_seed(seed + 67867967)
        self.size_px = torch.tensor(c["p_detect"]["size_px"], device=device)       # bin centres
        pd = c["p_detect"]                      # sky behind the bird (it is above the drone) or ground
        self.p_sky = torch.tensor(pd.get("p_sky", pd["p"]), device=device)
        self.p_ground = torch.tensor(pd.get("p_ground", pd["p"]), device=device)
        self.blind = torch.tensor(c["blind"]["p_visible"], device=device)            # (n_el, n_az) in the body frame
        # the focal length (detector pixels) of the camera that sees each direction best -- rig v2's OAK-D ahead shows a
        # bird 2.1x bigger than its wide cameras; the six-camera rig had one focal length everywhere
        self.focal = torch.tensor(c.get("focal_map", [[c["focal_px"]] * c["blind"]["n_az"]] * c["blind"]["n_el"]), device=device)
        self.track = BirdTrack(n, device)
        self.head = torch.zeros(n, device=device)
        self.p_fa = 1 - (1 - c["source"]["false_alarms_per_image"]) ** c.get("n_cameras", 6)   # pictures per frame

    def reset(self):
        self.track.reset()
        # the drone's own heading estimate is off by a few degrees for a whole flight (PX4 in Gazebo:
        # 5.8 deg on the pad, rig_diag.py); every direction the rig reports turns with it
        self.head = HEAD_SD * torch.randn(self.n, generator=self.g, device=self.dev)

    def _p_detect(self, size, above):
        """Linear in log size between the measured bins, flat beyond them; sky or ground behind the bird."""
        ls, lb = torch.log(size.clamp(min=0.1)), torch.log(self.size_px)
        i = torch.searchsorted(lb, ls.contiguous()).clamp(1, len(lb) - 1)
        w = ((ls - lb[i - 1]) / (lb[i] - lb[i - 1])).clamp(0, 1)
        pc = torch.where(above[:, None], self.p_sky, self.p_ground)
        lo, hi = pc.gather(1, (i - 1)[:, None])[:, 0], pc.gather(1, i[:, None])[:, 0]
        return lo + w * (hi - lo)

    def _frame(self, env, me, bp):
        """One camera frame: does the rig see the bird at bp from me, and what does the detector report."""
        c, b = self.c, env.bird
        u = lambda *s: torch.rand(*s, generator=self.g, device=self.dev)
        rel = bp - me
        dist = torch.linalg.norm(rel, dim=-1).clamp(min=0.5)
        d = rel / dist[:, None]
        # size in the image: head-on vs side-on (angle between the bird's flight and the sight line)
        fly = b["vel"] / torch.linalg.norm(b["vel"], dim=-1, keepdim=True).clamp(min=1e-6)
        side = torch.sqrt((1 - (fly * d).sum(-1) ** 2).clamp(min=0.0))
        # the drone's own blind spots, in its body frame (heading = the way it flies; no tilt here)
        h = env.heading
        az = torch.atan2(d[:, 1] * h[:, 0] - d[:, 0] * h[:, 1], d[:, 0] * h[:, 0] + d[:, 1] * h[:, 1])
        el = torch.asin(d[:, 2].clamp(-1, 1))
        n_el, n_az = self.blind.shape
        ia = ((az + math.pi) / (2 * math.pi) * n_az).long().clamp(0, n_az - 1)
        ie = ((el + math.pi / 2) / math.pi * n_el).long().clamp(0, n_el - 1)
        foc = self.focal[ie, ia]                              # the camera that sees this direction best
        size = foc * (c["span_head_m"] + (c["span_side_m"] - c["span_head_m"]) * side) / dist
        # trees and roofs in the way
        s = torch.linspace(0.05, 0.95, 12, device=self.dev)
        pts = me[:, None] + s[None, :, None] * rel[:, None]
        clear = (env.height_at(pts[..., :2]) <= pts[..., 2]).all(-1)
        p = self._p_detect(size, rel[:, 2] > 0) * self.blind[ie, ia] * clear * b["on"]
        det = u(self.n) < p
        # what the detector reports: a direction a little off, a size a little off -> distance. The
        # whole measured spread of distance-from-size (detector + the bird's banking and flapping), on
        # top of head-on vs side-on -- which double-counts that small part: errs pessimistic
        ang = c["sig_ang_rad"] * torch.randn(self.n, 3, generator=self.g, device=self.dev)
        d_m = d + ang - (ang * d).sum(-1, keepdim=True) * d
        d_m = d_m / torch.linalg.norm(d_m, dim=-1, keepdim=True)
        ch, sh = torch.cos(self.head), torch.sin(self.head)
        d_m = torch.stack([ch * d_m[:, 0] - sh * d_m[:, 1], sh * d_m[:, 0] + ch * d_m[:, 1], d_m[:, 2]], -1)
        size_m = size * torch.exp(c["sig_rel_dist"] * torch.randn(self.n, generator=self.g, device=self.dev))
        dist_m = foc * c["span_assumed_m"] / size_m
        z, R = me + d_m * dist_m[:, None], measure_cov(d_m, dist_m, c["sig_ang_rad"], c["sig_rel_dist"])
        # the detector's clutter false alarms (measured per picture, six pictures a frame): anywhere,
        # at a size log-uniform in FA_SIZE. As live: a confirmed track takes whichever detection fits
        # it better; otherwise the true bird's detection wins (it scores higher, as clutter mostly
        # scores 0.5-0.65 in the held-out set)
        fa = u(self.n) < self.p_fa
        dfa = torch.randn(self.n, 3, generator=self.g, device=self.dev)
        dfa = dfa / torch.linalg.norm(dfa, dim=-1, keepdim=True)
        sfa = FA_SIZE[0] * (FA_SIZE[1] / FA_SIZE[0]) ** u(self.n)
        rfa = c["focal_px"] * c["span_assumed_m"] / sfa
        zfa, Rfa = me + dfa * rfa[:, None], measure_cov(dfa, rfa, c["sig_ang_rad"], c["sig_rel_dist"])
        conf = self.track.confirmed
        use_fa = fa & (~det | (conf & (self.track.angle(zfa, me) < self.track.angle(z, me))))
        return det | fa, torch.where(use_fa[:, None], zfa, z), torch.where(use_fa[:, None, None], Rfa, R)

    def step(self, env, t, dt, me0, bp0, frames=5):
        """One simulator step = `frames` camera frames (20 Hz for DT = 0.25 s, as on the Gazebo rig),
        the drone and the bird moving in straight lines between the step's ends; the tracker takes
        each frame. Frames are drawn independently -- if real misses cluster, this is optimistic
        (the Gazebo flights, which run the real detector, are the check)."""
        det_any = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        for k in range(1, frames + 1):
            f = k / frames
            det, z, R = self._frame(env, me0 + (env.pos - me0) * f, bp0 + (env.bird["pos"] - bp0) * f)
            self.track.predict(dt / frames)
            self.track.update(t - dt + f * dt, det, z, R, me0 + (env.pos - me0) * f)
            det_any |= det
        return det_any
