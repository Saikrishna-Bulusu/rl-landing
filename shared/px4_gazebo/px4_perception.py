#!/usr/bin/env python3
"""
px4_perception.py -- the two things that turn PX4+Gazebo into the actual task:
ArUco detection from the downward camera, and GeoAnchor sigma injected into the
believed pad position.

    python3 px4_perception.py --selftest       prove both halves work

ARUCO DETECTION

The x500_mono_cam_down publishes 1280x960 images on a gz-transport topic. They
are read directly with the gz.transport13 Python bindings -- no ROS bridge in
the loop, which keeps the latency honest and one fewer moving part.

The tag in aruco.sdf is DICT_4X4_50, id 0, 0.5 x 0.5 m (identified by trying
every predefined dictionary against the model's own PNG rather than guessing).

THE IDENTITY CHECK IS A HARD SAFETY LAYER, NOT AN OBSERVATION

`expect_id` is compared against the decoded marker id, and a mismatch is
reported as NOTHING SEEN. That is deliberate: in the cage experiment a
neighbour's marker fooled an unprotected drone 138 times out of 150, and the
fix that worked was making a wrong marker indistinguishable from no marker, so
the abort behaviour the policy already had could handle it. Keeping the check
outside the policy means no retraining and nothing new to learn.

SIGMA INJECTION

GeoAnchor estimates the DRONE's position, not the pad's. The pad is a delivery
coordinate, known exactly. So if the aircraft believes it sits at (true + err),
it computes the pad as (pad - err). Getting that sign backwards made the
guidance target run away in the very first version of this project, so it is
written once, here, and asserted in the selftest.

Errors are replayed from the real pool via the same AR(1) Gaussian copula used
everywhere else: a hidden smoothly-drifting number selects WHICH measured error
to replay, so the marginal distribution stays exactly what was measured while
gaining realistic time structure.
"""

import argparse
import math
import os
import time

import numpy as np

POOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fast_sim", "sigma_pool.npz")

BEARING_STEP = math.sqrt(-2.0 * math.log(0.25))   # 1.665 rad
ARUCO_DICT = "DICT_4X4_50"
ARUCO_ID = 0
TAG_SIZE = 0.5                     # metres, from arucotag/model.sdf
CAM_TOPIC = ("/world/aruco/model/x500_mono_cam_down_0/link/camera_link"
             "/sensor/imager/image")


def _make_detector(cv2, dict_name):
    """Return detect(gray) -> (corners, ids), working on OpenCV 4.6 and 5.x."""
    dic_id = getattr(cv2.aruco, dict_name)
    if hasattr(cv2.aruco, "ArucoDetector"):                 # 4.7+
        dic = cv2.aruco.getPredefinedDictionary(dic_id)
        det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())

        def detect(g):
            c, i, _ = det.detectMarkers(g)
            return c, i
        return detect
    # 4.6 and earlier
    dic = (cv2.aruco.Dictionary_get(dic_id)
           if hasattr(cv2.aruco, "Dictionary_get")
           else cv2.aruco.getPredefinedDictionary(dic_id))
    par = (cv2.aruco.DetectorParameters_create()
           if hasattr(cv2.aruco, "DetectorParameters_create")
           else cv2.aruco.DetectorParameters())

    def detect(g):
        c, i, _ = cv2.aruco.detectMarkers(g, dic, parameters=par)
        return c, i
    return detect


# ---------------------------------------------------------------------------
class ArucoEye:
    """Reads the downward camera and reports whether OUR marker is visible."""

    def __init__(self, topic=CAM_TOPIC, expect_id=ARUCO_ID, hfov_deg=99.7,
                 expect_ids=None):
        import cv2
        from gz.transport13 import Node
        self.cv2 = cv2
        # Ordered COARSE to FINE. The pad carries a 1.2 m marker for acquiring
        # from cruise and a 0.15 m one for the last metre, because a marker big
        # enough to see from 18 m stops fitting in frame below 0.8 m. look()
        # returns the finest one currently visible, so the handover is implicit.
        self.expect_ids = list(expect_ids) if expect_ids else [expect_id]
        self.expect_id = self.expect_ids[0]
        self.hfov = math.radians(hfov_deg)
        # OpenCV changed the aruco API at 4.7: the old free functions were
        # replaced by an ArucoDetector class. The conda env here has 5.0 and the
        # system python (which owns the gz bindings) has 4.6, so BOTH have to
        # work or the module is unusable in the process that can read the camera.
        self._detect = _make_detector(cv2, ARUCO_DICT)
        self._last = None
        self.last_corners = None      # for the video overlay only
        self.others = []              # markers decoded but NOT ours (neighbours' pads), overlay only
        self._node = Node()
        from gz.msgs10.image_pb2 import Image as GzImage
        self._GzImage = GzImage
        ok = self._node.subscribe(GzImage, topic, self._on_image)
        if not ok:
            raise RuntimeError(f"could not subscribe to {topic}")

    def _on_image(self, msg):
        try:
            h, w = msg.height, msg.width
            buf = np.frombuffer(msg.data, np.uint8)
            if buf.size == h * w * 3:
                img = buf.reshape(h, w, 3)
                gray = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2GRAY)
            elif buf.size == h * w:
                gray = buf.reshape(h, w)
            else:
                return
            self._last = gray
            self._last_frame = (gray, msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9)   # with its sim-time stamp
        except Exception:
            pass

    def look(self):
        """
        Returns (seen, offset_px, decoded_ids, which_id).

        `seen` is True only when one of OUR marker ids is decoded. A wrong id
        returns False -- indistinguishable from nothing in view, which is the
        point of the identity check. When both pad markers are visible the
        FINEST is returned, so the coarse-to-fine handover needs no state.
        """
        fr = getattr(self, "_last_frame", None)
        g, self.look_stamp = (fr if fr is not None else (self._last, None))   # look_stamp: when this picture was taken
        self.others, self.look_gray, self.last_id, self.found = [], g, None, {}   # the picture checked, for the film
        if g is None:
            return False, None, [], None
        corners, ids = self._detect(g)
        if ids is None or not len(ids):
            return False, None, [], None
        ids = ids.flatten().tolist()
        found = {int(i): c for c, i in zip(corners, ids)}
        self.found = found                                  # every marker decoded in this picture (the film boxes them)
        self.others = [(i, c.reshape(4, 2)) for i, c in found.items() if i not in self.expect_ids]
        for want in reversed(self.expect_ids):        # finest first
            if want in found:
                c = found[want]
                ctr = c.reshape(4, 2).mean(axis=0)
                h, w = g.shape[:2]
                self.last_corners, self.last_id = c.reshape(4, 2), want
                return (True, (float(ctr[0] - w / 2), float(ctr[1] - h / 2)),
                        ids, want)
        self.last_corners = None
        return False, None, ids, None    # saw A marker, but not OURS

    def bearing(self, offset_px, width=1280):
        """Convert a pixel offset into an angular offset, radians."""
        if offset_px is None:
            return None
        f = (width / 2) / math.tan(self.hfov / 2)
        return (math.atan2(offset_px[0], f), math.atan2(offset_px[1], f))


# ---------------------------------------------------------------------------
class SigmaSource:
    """Replays GeoAnchor's measured (sigma, error) pairs with AR(1) structure."""

    def __init__(self, pool=POOL, rho=None, scale=1.0, seed=0):
        # allow_pickle is needed only for the string scene-name array. The pool
        # is generated locally by calibrate.py from GeoAnchor's own CSVs and is
        # never fetched from anywhere, so it is not untrusted input.
        d = np.load(pool, allow_pickle=True)
        self.sigma = np.asarray(d["sigma"], float)
        self.err = np.asarray(d["err"], float)
        scene = np.asarray(d["scene"]).astype(str)
        self.rho = float(rho) if rho is not None else (
            float(d["rho"]) if "rho" in d.files else 0.0)
        self.scale = float(scale)
        self.scenes = []
        for s in sorted(set(scene)):
            m = scene == s
            if m.sum() < 20:
                continue
            o = np.argsort(self.sigma[m])
            self.scenes.append({"name": str(s), "sigma": self.sigma[m][o],
                                "err": self.err[m][o]})
        self.rng = np.random.default_rng(seed)
        self.new_episode()

    def new_episode(self):
        self._sc = self.scenes[int(self.rng.integers(len(self.scenes)))]
        self._z = float(self.rng.normal())
        self._bearing = float(self.rng.uniform(0, 2 * np.pi))
        self.t_fix = 0.0
        self.fix()

    def fix(self):
        """One replayed GeoAnchor fix: a real (sigma, error) pair."""
        sc = self._sc
        self._z = (self.rho * self._z +
                   math.sqrt(max(1.0 - self.rho ** 2, 0.0)) * self.rng.normal())
        u = 0.5 * (1.0 + math.erf(self._z / math.sqrt(2.0)))
        i = int(np.clip(round(u * (len(sc["sigma"]) - 1)), 0, len(sc["sigma"]) - 1))
        self.sigma_m = float(sc["sigma"][i]) * self.scale
        mag = float(sc["err"][i]) * self.scale
        # The error DIRECTION wanders. Measured from GeoAnchor's own per-frame
        # error vectors across 2993 clean frames: the mean cosine between
        # consecutive error directions is +0.25, and |mean error vector| / sd
        # is 0.08-0.33, so errors substantially cancel when averaged. A 0.02 rad
        # step made the direction effectively fixed for a whole flight, under
        # which averaging fixes is useless -- which is not what the data says.
        # For a Gaussian bearing step E[cos] = exp(-s^2/2), so s = sqrt(-2 ln 0.25).
        self._bearing += self.rng.normal(0.0, BEARING_STEP)
        self.err_xy = mag * np.array([math.cos(self._bearing),
                                      math.sin(self._bearing)])
        self.t_fix = 0.0
        return self.sigma_m, self.err_xy

    def believed_pad(self, true_pad_xy):
        """
        Where the aircraft THINKS the pad is, in true coordinates.

        GeoAnchor estimates the DRONE's position. The pad is known exactly. So
        believing itself at (true + err), the aircraft computes the pad as
        (pad - err). NOT (pad + err) -- that sign error produced 1-5% success
        across the board the first time it was made.
        """
        return np.asarray(true_pad_xy, float) - self.err_xy

    @property
    def scene(self):
        return self._sc["name"]


OBSERVER_TOPIC = "/observer/image"


class Recorder:
    """
    Writes a side-by-side mission video: overhead observer on the left, the
    aircraft's own downward camera with a detection overlay on the right.

    This is documentation, not instrumentation -- nothing here feeds back into
    the flight. It exists because a table of miss distances does not show a
    person that the drone flew to the wrong place and then corrected.

    The overhead camera is a static sensor added to the world (observer_cam in
    aruco.sdf). Gazebo runs headless, so there is no GUI recorder to use; the
    frames are pulled off gz-transport exactly like the onboard camera's.
    """

    W = H = 480                          # per panel
    # observer_cam sits at ENU (10, 0, 50) looking straight down, which is NED
    # north 0, east 10. Its image axes are NOT north-up: rotating the camera
    # 90 deg about +Y puts its optical axis down, its +Y along ENU north and
    # its +Z along ENU east, and image u/v run opposite to those. So u
    # decreases with north and v decreases with east. Checked against a frame:
    # the tag at NED (0, 20) lands at panel (240, 192), which is what this
    # predicts, and where the previous guess put it 100 px off.
    CENTRE_NE = (0.0, 120.0)             # observer_cam ground position, NED
    SPAN_M = 120.0                       # 60 m up at 90 deg FOV

    def __init__(self, path, fps=20, topic=OBSERVER_TOPIC):
        import cv2
        from gz.transport13 import Node
        from gz.msgs10.image_pb2 import Image as GzImage
        self.cv2 = cv2
        self._obs = None
        self._node = Node()              # keep the reference or it is collected
        self._node.subscribe(GzImage, topic, self._on_obs)
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (self.W * 2, self.H))
        if not self.vw.isOpened():
            raise RuntimeError(f"could not open {path} for writing")
        self.path, self.n, self.fps = path, 0, fps

    def _on_obs(self, msg):
        try:
            buf = np.frombuffer(msg.data, np.uint8)
            if buf.size == msg.height * msg.width * 3:
                self._obs = buf.reshape(msg.height, msg.width, 3)[:, :, ::-1]
        except Exception:
            pass

    def tick(self, eye, phase, p, believed, seen, ids, sigma, pad=(0.0, 0.0), t_sim=None):
        # One frame per 1/fps of SIMULATED time when t_sim is given, so the video plays at
        # true speed even when Gazebo stalls (0.04-0.4x real time while RL trains on the
        # same GPU); per wall-clock step, stalls looked like sudden slow motion.
        # ...and the same frame repeated when the ~17 Hz control loop falls behind 20 fps,
        # so video time equals simulated time both ways.
        reps = 1
        if t_sim is not None:
            if getattr(self, "_t_last", None) is None:
                self._t_last = t_sim
            else:
                reps = min(int((t_sim - self._t_last) * self.fps), 10)
                if reps < 1:
                    return
                self._t_last += reps / self.fps
        cv2 = self.cv2
        blank = np.zeros((self.H, self.W, 3), np.uint8)

        left = blank if self._obs is None else cv2.resize(
            np.ascontiguousarray(self._obs), (self.W, self.H))

        # The overhead camera is fixed and pointing straight down, so world
        # metres map linearly to pixels: mark the real pad and the believed one.
        def to_px(n, e):                 # NED north/east -> image u, v
            cn, ce = self.CENTRE_NE
            k = self.W / self.SPAN_M
            return (int(self.W / 2 - (n - cn) * k),
                    int(self.H / 2 - (e - ce) * k))

        cv2.drawMarker(left, to_px(*pad), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.drawMarker(left, to_px(*believed), (0, 128, 255),
                       cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.circle(left, to_px(p[0], p[1]), 9, (255, 80, 80), 2)   # the aircraft
        cv2.line(left, to_px(*believed), to_px(p[0], p[1]), (0, 128, 255), 1)
        cv2.putText(left, "blue = drone   green = real pad   orange = guess",
                    (8, self.H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1)

        g = eye._last
        right = blank if g is None else cv2.cvtColor(
            cv2.resize(g, (self.W, self.H)), cv2.COLOR_GRAY2BGR)
        if g is not None:                # a neighbour's marker: decoded, refused
            sc = np.array([self.W / g.shape[1], self.H / g.shape[0]])
            for mid, cn in eye.others:
                pts = (cn * sc).astype(np.int32)
                cv2.polylines(right, [pts], True, (40, 40, 230), 2)
                cv2.putText(right, f"id {mid}: not ours", tuple(int(v) for v in pts[0] + [0, -6]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 230), 1)
        if seen and eye.last_corners is not None and g is not None:
            sc = np.array([self.W / g.shape[1], self.H / g.shape[0]])
            pts = (eye.last_corners * sc).astype(np.int32)
            cv2.polylines(right, [pts], True, (0, 255, 0), 2)
        cv2.putText(right, "onboard camera (down)", (8, self.H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        f = np.hstack([left, right])
        bar = (f"{phase:<8} alt {p[2]:5.2f} m   sigma {sigma:4.1f} m   "
               f"marker {'LOCKED' if seen else ('seen ' + str(ids) if ids else 'none')}")
        cv2.rectangle(f, (0, 0), (self.W * 2, 26), (0, 0, 0), -1)
        cv2.putText(f, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if seen else (255, 255, 255), 1)
        for _ in range(reps):
            self.vw.write(f)
        self.n += reps

    def close(self):
        self.vw.release()
        return self.path, self.n


# ---------------------------------------------------------------------------
def selftest(pool=POOL):
    import cv2
    print("1. ArUco dictionary identification")
    from PIL import Image
    im = np.array(Image.open("/home/sai/thesis_2.0/PX4-Autopilot/Tools/"
                             "simulation/gz/models/arucotag/arucotag.png"
                             ).convert("L"))
    pad = np.full((im.shape[0] + 120, im.shape[1] + 120), 255, np.uint8)
    pad[60:60 + im.shape[0], 60:60 + im.shape[1]] = im
    detect = _make_detector(cv2, ARUCO_DICT)
    c, ids = detect(pad)
    got = sorted(int(i) for i in ids.flatten()) if ids is not None else []
    # The pad carries THREE markers, so check all three -- a texture where only
    # some decode is exactly the failure that makes the coarse-to-fine handover
    # silently impossible, and it looks like a flaky detector in flight.
    assert got == [0, 1, 2], f"{ARUCO_DICT} decoded {got}, expected [0, 1, 2]"
    print(f"   ok  {ARUCO_DICT} decodes all three pad markers {got}")

    print("\n2. THE SAFETY LAYER: a wrong id must read as NOTHING SEEN")

    class _Fake(ArucoEye):
        def __init__(self, expect):        # no gz subscription needed
            self.cv2 = cv2
            self.expect_id = expect
            self.expect_ids = [expect]
            self.last_corners = None
            self.hfov = math.radians(99.7)
            self._detect = detect
            self._last = pad
    ours = _Fake(ARUCO_ID).look()
    theirs = _Fake(ARUCO_ID + 7).look()     # id 7: a neighbour's, not on our pad
    print(f"   our marker    -> seen={ours[0]}, decoded {ours[2]}")
    print(f"   wrong id      -> seen={theirs[0]}, decoded {theirs[2]}")
    assert ours[0] is True, "our own marker was not accepted"
    assert theirs[0] is False, "SAFETY LAYER LEAKS: a wrong id was accepted"
    print("   ok  wrong marker is indistinguishable from no marker")

    print("\n3. sigma replay")
    s = SigmaSource(pool=pool, seed=0)
    sg, er = [], []
    for _ in range(400):
        a, b = s.fix()
        sg.append(a); er.append(np.linalg.norm(b))
    sg, er = np.array(sg), np.array(er)
    rank = lambda x: np.argsort(np.argsort(x)).astype(float)
    c_rep = np.corrcoef(rank(sg), rank(er))[0, 1]
    # Compare against THIS SCENE's own correlation, not the pooled one. An
    # episode replays a single scene, and the pooled figure mixes four scenes
    # whose within-scene correlations range from +0.03 to +0.62. Comparing a
    # single-scene replay against the pooled number is the same reference error
    # that caused a working env design to be thrown away earlier in this project.
    c_scene = np.corrcoef(rank(s._sc["sigma"]), rank(s._sc["err"]))[0, 1]
    c_pool = np.corrcoef(rank(s.sigma), rank(s.err))[0, 1]
    print(f"   scene {s.scene}, rho {s.rho:.3f}")
    print(f"   sigma p50 {np.median(sg):.1f} m, error p50 {np.median(er):.1f} m")
    print(f"   rank-corr(sigma,err): replay {c_rep:+.3f} vs this scene "
          f"{c_scene:+.3f} (pooled across all scenes {c_pool:+.3f})")
    assert abs(c_rep - c_scene) < 0.15, (
        f"sigma-error relationship distorted: replay {c_rep:+.3f} vs "
        f"scene {c_scene:+.3f}")

    print("\n4. THE SIGN: believed pad must be (pad - err), not (pad + err)")
    pad_xy = np.array([0.0, 0.0])
    bel = s.believed_pad(pad_xy)
    assert np.allclose(bel, pad_xy - s.err_xy), "believed pad has the wrong sign"
    print(f"   err {s.err_xy.round(2)} -> believed pad {bel.round(2)}  ok")

    print("\n5. live camera (requires SITL running)")
    try:
        eye = ArucoEye()
        t0 = time.time()
        while time.time() - t0 < 8 and eye._last is None:
            time.sleep(0.2)
        if eye._last is None:
            print("   no frames yet -- is SITL up and the drone airborne?")
        else:
            seen, off, ids, _ = eye.look()
            print(f"   frame {eye._last.shape}, marker seen={seen}, ids={ids}")
    except Exception as e:
        print(f"   camera unavailable: {e}")

    print("\nselftest complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pool", default=POOL)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.pool)
