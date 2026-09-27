#!/usr/bin/env python3
"""
px4_mission.py -- the whole pipeline, end to end, on real PX4 firmware.

    python3 px4_mission.py                 one mission
    python3 px4_mission.py --runs 5        repeat
    python3 px4_mission.py --wrong-id 7    expect a marker we will never see

THE MISSION

  1. take off to cruise altitude
  2. fly toward where GeoAnchor BELIEVES the pad is -- which is wrong by a
     replayed real error, so the aircraft is aiming at the wrong place
  3. below the detection ceiling, look for the ArUco marker
  4. once OUR marker is decoded, switch from the believed position to what the
     camera actually sees, centre on it, and descend
  5. if the marker is never found, ABORT: climb and return, never land blind

Every layer here is the real thing: PX4's firmware, its EKF, its
mc_pos_control, the Gazebo x500 with a downward camera, and a real ArUco tag.
The only injected fiction is the localisation error, and that is REPLAYED from
GeoAnchor's own measurements rather than invented.

WHY THE ABORT MATTERS MORE THAN THE LANDING

Across 1200 simulated episodes the trained policy never once landed without
seeing the marker -- it always aborted instead. A delivery drone that puts a
package down while blind has delivered it into a stranger's garden. The same
rule is enforced here explicitly, because it is the behaviour that makes the
system safe rather than merely capable.
"""

import argparse
import math
import json
import os
import threading
import time

import numpy as np

from px4_perception import ArucoEye, SigmaSource, Recorder, ARUCO_ID

# 8 m, not 14 m. The theoretical ceiling for a 0.5 m tag is 18.4 m, but at 14.8 m
# the tag spans only ~18 px against a 20 px decode minimum and detection was
# intermittent -- seen in roughly one frame in ten. At 8 m it spans ~34 px and
# detection is reliable. The theoretical limit is where detection BEGINS to
# fail, not where it works.
# A 0.5 m TAG IS NOT FINDABLE AT GEOANCHOR'S ACCURACY. MEASURED, NOT ASSUMED.
#
# The camera sees a square 2*h*tan(hfov/2) = 2.37*h metres wide, and the tag
# must also span >=34 px to decode reliably. Those two pull opposite ways: fly
# high enough to have the pad in frame and the tag is too few pixels; fly low
# enough to read it and the pad is outside the frame.
#
# With a 0.5 m tag the window is empty. Four missions flown to the believed pad
# with real GeoAnchor errors of 11.9-34.0 m found the marker ZERO times out of
# four. The window opens only when the pad is physically bigger:
#
#     error to absorb   min altitude   tag size needed
#          10 m            12.1 m          0.76 m
#          14 m            16.9 m          1.06 m
#          20 m            24.1 m          1.52 m
#
# So the pad is 1.5 m (arucotag/model.sdf) and cruise is 20 m, where a 1.5 m tag
# spans 40 px and the frame covers a 24 m radius. This is a real design
# constraint on the product, not a simulation detail: a 0.5 m delivery target
# cannot be acquired from a 14 m position fix.
# THE PAD CARRIES THREE MARKERS, NOT ONE.
#
# A single marker cannot cover the descent. It must be large to be acquired
# from cruise and small to still fit in frame on short final, and those are
# opposite requirements: the camera sees only 2.37*h metres, so a 1.2 m marker
# stops fitting below 0.8 m while a 0.15 m one is unreadable above 3 m.
#
# Nesting them -- the obvious layout -- does not work: any centre hole destroys
# the outer marker's data bits, and testing all 50 ids of DICT_4X4_50 found not
# one that survives even a 7% hole. So they sit side by side on a 6.6 m mat,
# with the SMALLEST at the model origin so it is the touchdown point, and
# look() returns whichever is finest and visible:
#
#     1.80 m  id 2   usable 28.6 m -> 0.76 m
#     1.20 m  id 0   usable 19.1 m -> 0.51 m
#     0.15 m  id 1   usable  2.4 m -> 0.06 m
#
# The bands overlap, so there is no altitude with nothing to see.
#
# THE 1.80 m RUNG IS NOT COSMETIC. Detection altitude is linear in marker size,
# and at 1.20 m it is 19.1 m -- BELOW a mature tree. That forces the aircraft to
# hunt for the pad from inside the canopy layer, the one place flying is
# dangerous. On identical obstacle layouts (bigger_marker.py, with detection
# SAMPLED FROM THE MEASURED CURVE rather than thresholded) going from 1.20 m to
# 1.80 m is worth +17.2 pp delivered and -30.5 pp crash with the same one-line
# rule. A larger square of paint beat every sensing and learning change
# measured in this project.
#
# 1.80 AND NOT 2.40: both reach 100% delivered once the time budget is
# realistic, so the extra 0.60 m buys nothing, and under a tight budget 2.40 m
# is worse -- seeing from 46 m makes the rule commit to a descent while its pad
# estimate is still ~12 m out.
# The pad model is generated and checked by make_pad.py.
TAG_SIZES = {2: 1.80, 0: 1.20, 1: 0.15}
# Where each marker sits on the mat, from the file that paints it: side by side along the
# mat's x axis (ENU east with the pad at yaw 0 in the world), id 1 at the touchdown point.
# Aiming at a marker's own centre put the aim point 2.15 m apart for ids 2 and 0, so
# while id 0 flickered at the edge of its range (30-36 m) the target jumped back and
# forth and the command direction flipped step to step (trace_fixed.csv).
# (Valid for the 3-marker pad make_pad.py paints now. The forest A/B's old 1.20 m
#  two-marker pad, ab_pad.sh old, had its own layout: re-derive before reusing it.)
from make_pad import MARKERS as _MARKERS
PAD_OFF_NED = {mid: np.array([0.0, x]) for mid, _size, x in _MARKERS}   # (north, east) m
MARK_ALPHA = 0.25          # per frame (~17 Hz): the pad-position estimate's time constant ~0.3 s
PAD_IDS = [2, 0, 1]               # coarse to fine
HFOV = math.radians(99.7)
# Cruise high, search low. The trained policy's episodes begin at 45-65 m and
# its central decision is HOW FAR TO DESCEND -- the marker cannot be decoded
# until about 19 m, so descending is what buys sight, and descending too far
# wastes the battery it will need to go home. A mission that cruises straight
# to 18 m has already made that decision by hand and leaves the policy nothing
# to decide. So the aircraft transits at delivery-cruise altitude and only
# comes down once it is over GeoAnchor's believed pad.
CRUISE_Z = 50.0
# LEFT AT 18 m ON PURPOSE. Raising it to 30 m with the 2.40 m rung looked
# obviously right -- a 71 m frame instead of 43 m -- and bought nothing: +8.9 s,
# paired p = 0.119, n = 5 (missions_pad24.json vs missions_pad24_alt18.json).
# The reason is that the 2.40 m marker is acquired during TRANSIT at cruise, so
# the search phase never runs and its altitude cannot matter. The marker alone
# is worth -52.2 s, paired p = 0.009.
# LEFT AT 18 m ON PURPOSE. Raising it to 30 m alongside a bigger coarse marker
# looked obviously right -- a 71 m frame instead of 43 m -- and bought nothing:
# +8.9 s, paired p = 0.119, n = 5 (missions_pad24.json vs
# missions_pad24_alt18.json). The reason is that the coarse marker is acquired
# during TRANSIT at cruise, so the search phase never runs and its altitude
# cannot matter. The marker alone was worth -52.2 s, paired p = 0.009.
SEARCH_ALT = 35.0                 # 1.8 m marker spans ~28 px here
# RAISED TO 35 m FOR REAL SUBURBS (--search-alt; the forest study flew 18 m, reproduce
# it with --search-alt 18). On the Logan LiDAR map the tallest obstacle within 45 m of
# a pad has a median of 22.9 m (90th pct 29.9 m): an 18 m search with 5 m clearance
# is safe at 16% of pads, 30 m at 61%, 35 m at 87% -- and 35 m is what the fast
# training env searches at, so both simulators now fly the same altitudes.
# Delivery-realistic cruise. At 2.5 m/s a 120 m leg ate 48 s of a 300 s budget
# before the aircraft could even look, which spends the battery on transit
# rather than on the decision being studied. PX4's own MPC_XY_VEL_MAX is 12.
V_LAT = 8.0
V_SEARCH = 6.0
V_DESC = 2.0
V_FLARE = 0.4
CENTRE_PX = 40             # pixels: close enough to call it centred
# 120 m from the depot, and that distance is the whole point. At 20 m altitude
# the camera covers a 47 m circle, so with the pad only 20 m away the aircraft
# saw the tag during the CLIMB and went straight to centring -- transit and
# search never ran in any of five missions, and GeoAnchor's error could not
# possibly matter. The pad has to be beyond sight of the launch point before
# sigma decides anything.
PAD_XY = (0.0, 120.0)      # NED; the SDF pose "120 0 0" is ENU = 120 m EAST


def load_trees(path="trees.json"):
    """
    The obstacles in the world, in the frame the mission flies in.

    make_trees.py writes both frames; this asserts the mapping instead of
    trusting it, because ENU/NED has already cost this project a day once.
    """
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    e = d["pad_enu"]
    assert abs(e[1] - PAD_XY[0]) < 1e-6 and abs(e[0] - PAD_XY[1]) < 1e-6, (
        f"trees.json pad ENU {e} does not map to PAD_XY {PAD_XY} under "
        f"NED=(ENU_y, ENU_x) -- the forest would sit somewhere else entirely")
    return d["trees"]


TREES = load_trees(os.environ.get("TREES_JSON", "trees.json"))   # gz_logan/trees_logan.json for the Logan world

# GEO_MISSION (the dashboard's mission planner, geo/fly_mission.sh): land on the mission's pad -- any of the sixteen in
# gz_logan/pads_logan.json. Same mat, other ids: position j of its ids is position j of ours (2, 1, 0).
if os.environ.get("GEO_MISSION"):
    _pad = json.load(open(os.environ["GEO_MISSION"]))["pad"]
    _ours = dict(zip(_pad["ids"], (2, 1, 0)))                   # its id -> our equivalent
    _its = {v: k for k, v in _ours.items()}
    PAD_XY = (float(_pad["enu"][1]), float(_pad["enu"][0]))    # NED
    PAD_IDS = [_its[i] for i in PAD_IDS]
    PAD_OFF_NED = {k: PAD_OFF_NED[v] for k, v in _ours.items()}
    TAG_SIZES = {k: TAG_SIZES[v] for k, v in _ours.items()}


def replant_forest(seed):
    """
    A DIFFERENT forest for this mission, spawned without restarting Gazebo.

    Twenty seeds against one fixed layout measures that layout precisely and
    says nothing about any other. Varying the trees per mission is what makes
    the result about forests rather than about forest number 11. Paired across
    arms by trial index, so arm A and arm B fly the SAME tree layouts.
    """
    global TREES
    import forest
    TREES = forest.replant(seed)
    worst = min(float(np.linalg.norm(np.array(t["ned"]) - np.array(PAD_XY)))
                - t["canopy_r"] for t in TREES)
    assert worst > 0, f"forest seed {seed} puts a canopy over the pad"
    print(f"  forest seed {seed}: {len(TREES)} trees, "
          f"clear column {worst:.2f} m")
    return worst
# A sigma-sized search takes time, and time is the cost the whole study is
# about. 120 s was enough only because the old missions never really searched.
T_MAX = 300.0
LOST_S = 3.0               # marker lost this long at altitude -> search again (unless committed)
# COMMIT ON LOCK. After LOCK_FRAMES reads of THIS pad's own marker (the id check has
# already refused the neighbours'), where the pad is -- local NED, held by the GPS/EKF --
# is known. Losing the marker for longer than LOST_S then means keep going down onto that
# spot at V_BLIND, not search: in the 24 Sept mist and fog skies, searching from 18 m
# found only neighbours' pads and aborted 0.7-80 m from our own (1 of 10 mist, 2 of 10 fog
# flights had a lock). Sai: once locked on the right pad, commit to landing.
LOCK_FRAMES, V_BLIND = 5, 0.5
# Losing the tag on short final does NOT mean it is gone. Close in, the tag is
# large in frame and a metre of lateral error pushes it over the edge of the
# image. Flying a 100 m sigma-sized spiral from 2 m up, which is what the
# altitude-blind recovery did, is the wrong answer to the right observation.
# Above this height a loss means search; below it, climb a little and look again.
REACQUIRE_ALT = 6.0

# Pixel offset -> world NED direction.
#
#     dNED = (h / f) * R(yaw) * B * (ex, ey)
#
# The previous version was a single 2x2 matrix fitted by flying to six offsets.
# It silently baked in the yaw it was fitted at, so it was only correct on that
# heading -- on a mission flown at a different one it drove the aircraft 40 m
# the wrong way. It was not even a valid rotation: its singular values were
# 0.81 and 0.62 instead of 1.0, so it skewed and shrank every offset too.
#
# B is the camera-to-body mounting rotation, solved once by calib_cam.py and
# constrained orthonormal; R(yaw) is read from the vehicle every frame. The
# calibration's singular values came out 0.964 and 0.947 -- the model's own
# consistency check, which the old fit failed.
F_PX = 539.94                 # from the camera_info topic, not assumed
CAM_B = np.load("cam_B.npy") if __import__("os").path.exists("cam_B.npy") else None
# RIG=v2 (gz_unified.py, Sai 25 Sept): the landing camera is PX4's CGO3 gimbal camera, which PX4's gimbal module
# holds straight down (MNT_DO_STAB: the airframe's roll and pitch taken out); cam_ray projects with the camera's own
# measured attitude (GimbalAtt), checked on flights by gz_unified.py --check. 1280x720, 2.0 rad: f = 640/tan(1.0).
# B (only before the camera's first IMU reading, taken as level): image right = body right, image down = body back.
GIMBAL = os.environ.get("RIG", "") in ("v2", "v2t", "v2u", "v2l")   # gz_unified.VARIANTS with the gimbal (v2f: fixed)
GIMBAL_TOPIC = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image"
if GIMBAL:
    F_PX = 640.0 / math.tan(1.0)
    CAM_B = np.array([[0.0, -1.0], [1.0, 0.0]])


class GimbalAtt:
    """Rig v2: the gimbal camera's attitude, from the IMU on its camera link (its x axis is the optical axis: both are
    posed yaw pi on the link), as a real gimbal reports it (GIMBAL_DEVICE_ATTITUDE_STATUS). Needed because Gazebo's PX4
    gimbal bridge moves the joints only 5 times a second (GZGimbal: ScheduleOnInterval 200 ms): in hard manoeuvres the
    camera trailed the airframe by up to 30 deg (25 Sept) -- assuming it level would misplace the marker by metres."""

    def __init__(self, model="x500_mono_cam_down_0"):
        from gz.transport13 import Node
        from gz.msgs10.imu_pb2 import IMU
        import collections
        self.q, self.hist, self.node = None, collections.deque(maxlen=250), Node()
        self.node.subscribe(IMU, f"/world/aruco/model/{model}/link/camera_link/sensor/camera_imu/imu", self._on)

    def _on(self, m):
        o, st = m.orientation, m.header.stamp
        self.q = (o.w, o.x, o.y, o.z)
        self.hist.append((st.sec + st.nsec * 1e-9, self.q))

    def q_at(self, t):
        """The attitude when a picture stamped t was taken (the 250 Hz IMU's nearest reading): the camera swings
        between the bridge's joint updates, and the newest reading was 12-18 px off the picture (25 Sept)."""
        h = list(self.hist)
        if t is None or not h:
            return self.q
        ts, q = min(h, key=lambda e: abs(e[0] - t))
        return q if abs(ts - t) < 0.05 else self.q

    def R(self, q=None):
        """World ENU <- camera (x ahead along the optical axis, y image-left, z image-up)."""
        w, x, y, z = q or self.q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def tilt_deg(self, q=None):
        """The optical axis's angle from straight down (nan before the first reading)."""
        q = q or self.q
        return float("nan") if q is None else math.degrees(math.acos(float(np.clip(-self.R(q)[2, 0], -1, 1))))


def cam_ray(off, roll, pitch, yaw, gimbal=None, t=None):
    """NED direction to a pixel offset (ex, ey) from the landing camera's centre (only its direction is used). A
    body-fixed camera turns with the airframe's full attitude; the gimbal camera with its own measured attitude
    (GimbalAtt), or -- before its first reading -- is taken as level, turned by the yaw alone."""
    if GIMBAL and gimbal is not None and gimbal.q is not None:
        d = gimbal.R(gimbal.q_at(t)) @ np.array([F_PX, -off[0], -off[1]])   # ENU; image right = -y, image down = -z
        return np.array([d[1], d[0], -d[2]])
    fr = CAM_B @ np.asarray(off, float)                        # body forward / right, px
    return (rot_nb(0.0, 0.0, yaw) if GIMBAL else rot_nb(roll, pitch, yaw)) @ np.array([fr[0], fr[1], F_PX])


def rot_nb(roll, pitch, yaw):
    """Body (forward, right, down) -> local NED, aerospace 3-2-1 Euler angles."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])

# GUIDANCE, FROM THE MEASURED PLANT RATHER THAN FROM TUNING BY HAND.
#
# measure_tau.py commanded PX4 a 2 m/s step and logged what happened:
#
#     lateral  tau = 0.69 s, settles at 2.02 m/s (commanded 2.00)
#     vertical tau = 0.23 s, settles at 1.52 m/s (commanded 1.50)
#     hover attitude: roll +/-0.05 deg, pitch +/-0.04 deg
#
# That hover figure matters: PX4's inner loops are not the problem, so tuning
# MC_* PID gains would have been tuning the wrong loop entirely. The visible
# wobble came from the OUTER law here, and specifically from SATURATION. The
# old law was proportional-derivative clipped to a speed cap, and whenever the
# marker was more than a few metres away the clip threw away the derivative
# term along with everything else -- so the aircraft ran at full speed with no
# damping at all and only began braking once the error was small, which is
# precisely an overshoot-and-oscillate cycle.
#
# The replacement never saturates by accident: it commands the fastest speed it
# can still stop from, sqrt(2*A_MAX*d), and it aims at where the aircraft will
# BE one lag-time from now rather than where it is, using the measured tau.
V_CENTRE = 2.5
K_LIN = 0.8               # 1/s: proportional zone of the approach law, near the target
TAU_LAT = 0.69            # measured, px4_response.json
A_MAX = 1.5               # comfortable deceleration, m/s^2

# ON SHORT FINAL THE MARKER MUST BE LOST, AND THAT IS NOT A FAILURE.
#
# The camera sees a patch 2*h*tan(hfov/2) = 2.37*h metres wide. At 0.35 m that
# is 0.83 m and the 0.5 m tag fills 60% of the frame; at 0.18 m it is 0.43 m
# and the tag is LARGER THAN THE FRAME, so no corner is visible and decoding is
# impossible. Waiting for a detection below this height waits forever -- which
# is exactly what one mission did, hovering at 0.18 m with a 0.07 m miss until
# the 120 s timeout, because the touchdown test lived inside the `if seen`
# branch and therefore never ran once the marker went out of frame.
#
# Real autoland behaves the same way: below a commit height you stop looking
# and put it down. This is not blind landing -- the rule is never to land
# having never seen the marker, and by here it has been tracked all the way in.
# Derived from the tag size rather than hardcoded, because hardcoding 0.8 m is
# what broke when the tag grew: a 1.5 m tag stops fitting in frame at 0.63 m,
# not 0.21 m, so the commit height has to move with it.
def commit_alt(marker_id):
    """
    Height below which THIS marker no longer fits in frame, so looking for it
    is pointless and the aircraft should commit. Derived from the marker, never
    hardcoded -- a hardcoded 0.8 m is exactly what broke when the tag grew.
    Tracking the fine marker it comes out at 0.13 m, below touchdown: with two
    markers there is effectively no blind descent left.
    """
    return 2.0 * TAG_SIZES[marker_id] / (2 * math.tan(HFOV / 2))


TOUCHDOWN_ALT = 0.35

# BATTERY. The training environment reasons about a budget with a reserve held
# back to fly home, and rewards an abort (-10) far above a risky landing (-30
# or -80). The flight code did not model a budget at all, so "give up" could
# only ever be triggered by exhausting the search. Now it is triggered by
# either, which is what the policy was actually trained to decide.
RESULT_JSON = os.environ.get("MISSION_JSON", "px4_response.json")
T_BUDGET = 300.0                  # endurance proxy, seconds of flight
RTL_RESERVE = 0.20                # fraction kept back to climb and return


class PolicyBrain:
    """
    The trained PPO policy, driving the real aircraft.

    WHAT IT DOES AND DOES NOT DECIDE

    It sets the vertical rate and the search radius. It does NOT centre the
    aircraft on the marker -- a proportional law with a braking profile already
    does that to within half a metre, and asking a network to rediscover it
    would be swapping something exact for something approximate. And it does
    not decide whether a marker is ours: the identity check stays a hard layer
    outside the policy, so no amount of training can talk the aircraft into a
    stranger's garden.

    THE OBSERVATION MUST BE BUILT EXACTLY AS TRAINED

    Six channels, in the order and scaling of ApproachEnv._obs. Getting a
    divisor wrong here would not raise anything -- the policy would simply act
    on a world that does not exist, and it would look like bad flying.

    Sigma is passed through UNSCALED even though training inflated it by 1.75.
    Both are in metres and the mapping from metres of uncertainty to metres of
    search is what has to transfer; feeding the inflated number would make the
    aircraft search an area for an error it is not facing.
    """

    VZ_MAX = 2.0                  # ApproachEnv.VZ_MAX
    SEARCH_R_MAX = 25.0           # (a[1]+1)/2 * 25

    def __init__(self, path, t_budget):
        # Plain numpy weights from export_policy.py, verified against SB3 to
        # 2.4e-07. The flight process has pymavlink and the gz bindings; the
        # training stack lives in a different interpreter entirely, and the
        # weights are the only part that needs to fly.
        if path.endswith(".zip"):
            path = path.replace(".zip", "_weights.npz")
        self.w = dict(np.load(path))
        self.t_budget = float(t_budget)

    def act(self, alt, climb, d_bel, sigma, acquired, elapsed):
        obs = np.array([
            alt / 60.0,
            climb / self.VZ_MAX,
            min(d_bel, 100.0) / 100.0,
            sigma / 50.0,
            float(acquired),
            min(elapsed, self.t_budget) / self.t_budget,
        ], np.float32)
        x = obs
        n = int(self.w["n_layers"])
        for i in range(n - 1):
            x = np.tanh(self.w[f"w{i}"] @ x + self.w[f"b{i}"])
        a = np.clip(self.w[f"w{n-1}"] @ x + self.w[f"b{n-1}"], -1, 1)
        return (float(a[0]) * self.VZ_MAX,
                float(a[1] + 1.0) * 0.5 * self.SEARCH_R_MAX)


class Mission:
    def __init__(self, endpoint="udpin:0.0.0.0:14540", pool="sigma_pool.npz",
                 expect_id=ARUCO_ID, seed=0, scale=1.0, video=None,
                 search_k=2.5, expect_ids=None, policy=None, aim_mean=True, search_alt=SEARCH_ALT):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.m = mavutil.mavlink_connection(endpoint, source_system=255,
                                            source_component=190)
        if self.m.wait_heartbeat(timeout=25) is None:
            raise RuntimeError("no PX4 heartbeat -- is SITL running?")
        self._beating = True
        threading.Thread(target=self._beat, daemon=True).start()
        time.sleep(3.0)
        self.search_k = float(search_k)
        self.search_alt = search_alt
        self.aim_mean = bool(aim_mean)
        self._bel_sum, self._bel_n = np.zeros(2), 0
        self.brain = None if policy is None else PolicyBrain(policy, T_BUDGET)
        self.gimbal = GimbalAtt() if GIMBAL else None   # rig v2: the landing camera's own attitude
        self.eye = ArucoEye(expect_id=expect_id, expect_ids=expect_ids,
                            **(dict(topic=GIMBAL_TOPIC, hfov_deg=math.degrees(2.0)) if GIMBAL else {}))
        self.sig = SigmaSource(pool=pool, seed=seed, scale=scale)
        # Where the tag actually is, matching <pose> on the arucotag include in
        # aruco.sdf. It is NOT at the origin: the aircraft spawns at the origin
        # and must fly to the pad, or the transit and search phases never run.
        self.pad = np.array(PAD_XY, float)
        self.rec = None if video is None else Recorder(video)
        self.MASK = 3015          # velocity + absolute yaw; see send()
        self.yaw_hold = 0.0
        self.yaw = 0.0
        self.roll = self.pitch = 0.0
        self.t_boot = 0.0
        self.last_cmd, self.last_off, self.last_dned = (0.0, 0.0, 0.0), (np.nan, np.nan), (np.nan, np.nan)
        # MISSION_TRACE=path.csv: one row per control step, for diagnosing wobble. Off by default.
        tp = os.environ.get("MISSION_TRACE")
        self.trace = None if not tp else open(tp, "w")
        if self.trace:
            self.trace.write("t,tsim,x,y,z,vn,ve,climb,roll,pitch,yaw,phase,seen,ex,ey,dn,de,cmd_vn,cmd_ve,cmd_up,pad_n,pad_e,mid\n")
        self.climb = 0.0
        self.policy_vz = 0.0
        self.active_id = PAD_IDS[0]
        self.policy_radius = 25.0
        assert CAM_B is not None, "run calib_cam.py first (writes cam_B.npy)"

    # ---- plumbing --------------------------------------------------------
    def _beat(self):
        while self._beating:
            try:
                self.m.mav.heartbeat_send(self.mavutil.mavlink.MAV_TYPE_GCS,
                                          self.mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                                          0, 0, 0)
            except Exception:
                pass
            time.sleep(1.0)

    def send(self, vx, vy, vz_up):
        self.last_cmd = (float(vx), float(vy), float(vz_up))
        # Absolute yaw, not yaw RATE. The old mask ignored the yaw field and
        # commanded rate 0, which asks the aircraft not to rotate rather than
        # telling it where to point -- so small errors integrated freely and a
        # 12 s lateral leg drifted 10 degrees. That matters more here than on a
        # normal vehicle because the camera is body-fixed: drifting yaw slowly
        # rotates the whole pixel-to-world mapping and walks the tag across the
        # frame, which is visible in the recordings as the marker lock
        # flickering on and off while nothing else changes.
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            self.mavutil.mavlink.MAV_FRAME_LOCAL_NED, self.MASK,
            0, 0, 0, float(vx), float(vy), -float(vz_up), 0, 0, 0,
            float(self.yaw_hold), 0)

    def pos(self):
        # The NEWEST position and attitude, not the oldest queued. One blocking read
        # per loop returned the oldest waiting message: measured on the idle SITL the
        # controller flew on position data 0.65-0.73 s old -- as long as the airframe's
        # own 0.69 s lag -- and centring became a limit cycle between the +/-2.5 m/s
        # caps with +/-28 deg of roll (the wobble in the first Logan video). Drain what
        # is waiting; block for a new message only when none has arrived.
        p = None
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                break
            if msg.get_type() == "LOCAL_POSITION_NED":
                p = msg
        if p is None:
            p = self.m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1)
            if p is None:
                return None
        att = self.m.messages.get("ATTITUDE")
        if att is not None:
            self.yaw = att.yaw            # needed every frame: see CAM_B
            self.yawspeed = att.yawspeed  # rad/s: where a turn will stop (gz_learner's heading lock for the landing)
            self.roll, self.pitch = att.roll, att.pitch
        self.vel_ne = np.array([p.vx, p.vy])
        self.t_boot = p.time_boot_ms / 1000.0   # PX4 time = simulator time (lockstep)
        self.climb = -p.vz
        return np.array([p.x, p.y, -p.z])

    def cmd(self, c, *a):
        args = list(a) + [0] * (7 - len(a))
        self.m.mav.command_long_send(self.m.target_system,
                                     self.m.target_component, c, 0, *args)

    def arm_offboard(self):
        p = self.pos()                 # capture the heading to hold
        self.yaw_hold = self.yaw
        for _ in range(60):
            self.send(0, 0, 0); time.sleep(0.05)
        self.cmd(self.mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, 6)
        time.sleep(0.5)
        self.cmd(self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        t0 = time.time()
        while time.time() - t0 < 6:
            self.send(0, 0, 0)
            hb = self.m.recv_match(type="HEARTBEAT", blocking=False)
            if hb and (hb.base_mode &
                       self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                self.gimbal_down()
                return True
            time.sleep(0.05)
        return False

    def gimbal_down(self):
        """Rig v2: the gimbal straight down (pitch -90, roll 0, yaw 0; mode 2 = MAVLink targeting). Sent every control
        step, not once: PX4's gimbal module stabilises at 50 Hz, but its Gazebo bridge (GZGimbal) passes the angles
        to the joints only every 200 ms -- or whenever PX4 receives a command. So each step's command makes the
        simulated gimbal track PX4's stabilisation at the loop rate, as a real gimbal's own controller does."""
        if GIMBAL:
            self.cmd(self.mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL, -90, 0, 0, 0, 0, 0, 2)

    def disarm(self):
        self.cmd(self.mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)

    def return_to_depot(self, tmax=120, v_home=5.0, v_down=1.5, z_home=None):
        """
        Put the aircraft back at the launch point before every mission.

        The previous version only brought it DOWN, so in a batch the second and
        later missions started wherever the last one landed -- on the pad. They
        then decoded the tag during the climb and never flew a transit at all:
        the very artifact that moving the pad 120 m away was meant to remove,
        reintroduced by the loop that repeats the mission.
        """
        p = self.pos()
        if p is None:
            return
        if np.linalg.norm(p[:2]) < 1.5 and p[2] < 0.5:
            self.disarm(); return
        if not self.arm_offboard():
            return
        t0 = time.time()
        while time.time() - t0 < tmax:
            p = self.pos()
            if p is None:
                continue
            d = -p[:2]                       # the depot is the NED origin
            home = np.linalg.norm(d) < 1.5
            if home and p[2] < 0.4:
                break
            if home:
                self.send(0, 0, -v_down if p[2] > 3 else -0.5)
            else:
                # z_home: over what stands between (climbed to first, in place, before heading home)
                v = np.zeros(2) if z_home is not None and p[2] < z_home - 5.0 else np.clip(d * 0.5, -v_home, v_home)
                self.send(v[0], v[1], ((z_home or CRUISE_Z) - p[2]) * 0.5)
            time.sleep(0.05)
        self.disarm()
        time.sleep(2.0)

    def _approach(self, d, cap):
        """
        Fly at `d` (metres, NED) fast but arriving stopped.

        Two ideas, both cheap. Command only the speed the aircraft can still
        brake from within the remaining distance, so it is never asked to stop
        faster than it can -- that alone removes the overshoot. Then aim at
        where it will be one lag-time from now, d - v*tau, because a command
        issued today does not take effect for 0.69 s and steering at the
        present position is steering at the past.
        """
        d_eff = np.asarray(d, float) - self.vel_ne * TAU_LAT
        n = float(np.linalg.norm(d_eff))
        if n < 1e-3:
            return np.zeros(2)
        # ...and near the target, proportional. sqrt(2*A*n) is infinitely steep at
        # n = 0: 0.55 m/s for 0.1 m of error, so once over the pad (from ~35 m down)
        # noise flipped the command every few steps -- a 2.3 Hz roll buzz, command
        # reversing on 35% of steps. K_LIN takes over inside 2*A/K^2 = 4.7 m; with the
        # lag lead it is a PD loop with damping ratio ~1.0 on the measured plant.
        return d_eff / n * min(cap, math.sqrt(2.0 * A_MAX * n), K_LIN * n)

    def _begin_search(self, centre, alt=None):
        """
        Lay out an expanding-square search around `centre`, sized by sigma.

        THE ORIGINAL SEARCH PHASE MADE THINGS WORSE. It descended, and
        descending SHRINKS the footprint -- the aircraft was reducing its
        chance of seeing the pad with every metre. It has to hold altitude and
        cover ground.

        The radius comes from GeoAnchor's own sigma: at k=2.5 that covers ~95%
        of a two-dimensional error, and k is swept by sweep_search_k.py. This
        is the operational payoff of calibrating sigma at all. An overconfident sigma searches too small a box and
        aborts a delivery that was findable; a pessimistic one burns battery
        sweeping ground the pad was never on.
        """
        alt = self.search_alt          # the sweep is flown at the search height
        swath = 2 * 0.7 * math.tan(HFOV / 2) * max(alt, 5.0)
        radius = (self.policy_radius if self.brain is not None
                  else self.search_k * self.sig.sigma_m)
        legs = max(2, min(12, int(math.ceil(2 * radius / swath)) * 2))
        pts, q, d = [], np.array(centre, float), np.array([1.0, 0.0])
        for k in range(1, legs // 2 + 1):
            for _ in range(2):
                q = q + d * k * swath
                pts.append(q.copy())
                d = np.array([-d[1], d[0]])      # turn left 90 deg
        self.search_wps, self.wp_i = pts, 0

    # ---- the mission -----------------------------------------------------
    def run(self, verbose=True):
        self.sig.new_episode()
        believed = self.sig.believed_pad(self.pad)
        err = float(np.linalg.norm(self.sig.err_xy))
        if verbose:
            print(f"  scene {self.sig.scene}: sigma {self.sig.sigma_m:.1f} m, "
                  f"actual error {err:.1f} m")
            print(f"  GeoAnchor believes the pad is at "
                  f"({believed[0]:+.1f}, {believed[1]:+.1f}); it is at "
                  f"({self.pad[0]:+.1f}, {self.pad[1]:+.1f})")
        # Start from the depot, on the ground, every time.
        self.return_to_depot()
        if not self.arm_offboard():
            time.sleep(3.0)               # PX4 can still be settling after boot
            if not self.arm_offboard():
                # Same keys as every other outcome. A bare {"outcome": ...} made
                # the summary crash with a KeyError, which reads like a flight
                # failure and is actually a reporting one.
                return {"outcome": "arm_failed", "miss_m": None, "alt_m": None,
                        "ever_saw_marker": False, "sigma_m": self.sig.sigma_m,
                        "scene": self.sig.scene, "t_s": 0.0}

        t0 = time.time()
        phase = "climb"
        self.search_wps, self.wp_i, self.t_seen = [], 0, time.time()
        self.mark_ned = None
        self.n_lock, self.locked = 0, False
        self._bel_sum, self._bel_n = np.zeros(2), 0
        self.active_id = PAD_IDS[0]
        seen_ever = False
        # WHERE the marker was first decoded, not merely whether it was. The
        # 2.40 m rung is supposed to buy acquisition altitude -- 38.1 m against
        # 19.1 m -- and altitude is what buys search footprint (2.37*h across).
        # A run that lands either way hides the whole effect, so record it.
        acq_alt = acq_t = acq_id = None
        acq_span = acq_want = None
        crashed_into = None
        last_log = 0.0
        while time.time() - t0 < T_MAX:
            p = self.pos()
            if p is None:
                self.send(0, 0, 0); continue
            elapsed = time.time() - t0

            # THE SAME CRASH TEST THE POINT-MASS ENV USES (_crashed in
            # approach_env_v5): inside the trunk below the canopy, or inside
            # the canopy within its band. The trees carry no collision geometry
            # -- see make_trees.py for why -- so this is what records the hit.
            for k, tr in enumerate(TREES):
                d = float(np.linalg.norm(p[:2] - np.array(tr["ned"])))
                top, frac = tr["h"], tr["canopy_frac"]
                if (p[2] < top and d < tr["trunk_r"]) or \
                   (frac * top < p[2] < top and d < tr["canopy_r"]):
                    crashed_into = k
                    print(f"    CRASH into tree {k} at alt {p[2]:.1f} m, "
                          f"{d:.1f} m from its axis (canopy {tr['canopy_r']:.1f} m, "
                          f"top {top:.1f} m)")
                    break
            if crashed_into is not None:
                # Record it, then FLY HOME rather than cutting the motors here.
                # Disarming mid-air drops the airframe wherever it happens to
                # be, and the next mission in the batch would take off from
                # that spot instead of the depot -- which silently breaks the
                # seed pairing every comparison in this project relies on. The
                # run is already scored as a crash; the return is bookkeeping.
                phase = "crash"
                self.return_to_depot()
                break

            # a fresh GeoAnchor fix twice a second, as on the real aircraft
            self.sig.t_fix += 0.05
            if self.sig.t_fix >= 0.5:
                self.sig.fix()
                latest = self.sig.believed_pad(self.pad)
                self._bel_sum += latest
                self._bel_n += 1
                # AIM AT THE AVERAGE OF EVERY FIX, NOT THE NEWEST ONE.
                #
                # GeoAnchor's error direction wanders (mean cosine +0.25
                # between consecutive fixes) while its magnitude persists, so
                # the errors largely cancel when averaged. In the point-mass
                # study this cut the pad-estimate error from 22.1 m to 9.3 m,
                # a 58% reduction, and took an otherwise identical rule from
                # 76.7% to 100% delivered. It costs one running sum.
                believed = (self._bel_sum / self._bel_n if self.aim_mean
                            else latest)

            seen, off, ids, mid = self.eye.look()
            if seen:
                if not seen_ever:
                    acq_alt, acq_t, acq_id = float(p[2]), elapsed, int(mid)
                    # IS THIS DETECTION THE RIGHT SIZE?
                    #
                    # A marker at altitude h must span TAG_SIZES[id]*F_PX/h
                    # pixels. If the quad the detector returned is a different
                    # size, it is not the pad -- DICT_4X4_50 has short Hamming
                    # distances and aliasing at small scales can decode a valid
                    # id out of noise. That matters more than a missed
                    # detection: a false positive of OUR OWN id walks straight
                    # past the wrong-id safety layer and steers the aircraft.
                    c = self.eye.last_corners
                    if c is not None:
                        got = float(np.linalg.norm(c[0] - c[1]))
                        want = TAG_SIZES[mid] * F_PX / max(p[2], 1e-6)
                        acq_span, acq_want = got, want
                        print(f"    FIRST DECODE id{mid} at {p[2]:.1f} m: quad "
                              f"{got:.1f} px, geometry predicts {want:.1f} px"
                              f"{'   <-- MISMATCH' if got > 2*want or got < 0.5*want else ''}")
                seen_ever = True
                self.t_seen = time.time()
                self.active_id = mid
                self.n_lock += 1
                if not self.locked and self.n_lock >= LOCK_FRAMES and self.mark_ned is not None:
                    self.locked = True
                    print(f"    LOCKED on our pad after {self.n_lock} reads, at ({self.mark_ned[0]:+.1f}, "
                          f"{self.mark_ned[1]:+.1f}), alt {p[2]:.1f} m")

            if self.brain is not None:
                self.policy_vz, self.policy_radius = self.brain.act(
                    alt=p[2], climb=self.climb,
                    d_bel=float(np.linalg.norm(believed - p[:2])),
                    sigma=self.sig.sigma_m, acquired=seen,
                    elapsed=elapsed)

            # Battery reserve outranks everything except an in-progress
            # touchdown: once the reserve is gone the only safe outcome is to
            # go home, whether or not the pad was ever found.
            batt = max(0.0, 1.0 - elapsed / T_BUDGET)
            if batt <= RTL_RESERVE and phase not in ("abort", "landed") \
                    and p[2] > commit_alt(self.active_id):
                phase = "abort"

            if phase == "climb":
                self.send(0, 0, 3.0)
                if p[2] > CRUISE_Z:
                    phase = "transit"
            elif phase == "transit":
                # fly to where GeoAnchor THINKS the pad is
                d = believed - p[:2]
                if seen:
                    phase = "centre"
                elif np.linalg.norm(d) < 2.0:
                    if seen_ever:
                        phase = "centre"
                    else:
                        self._begin_search(believed, p[2]); phase = "search"
                else:
                    v = self._approach(d, V_LAT)
                    self.send(v[0], v[1], 0.0)
            elif phase == "search":
                # arrived where it believed the pad was, and it is not there.
                # Sweep the sigma-sized box AT ALTITUDE. Never land blind.
                if seen:
                    phase = "centre"
                elif self.wp_i >= len(self.search_wps):
                    phase = "abort"            # looked everywhere sigma allowed
                else:
                    d = self.search_wps[self.wp_i] - p[:2]
                    if np.linalg.norm(d) < 2.5:
                        self.wp_i += 1
                    v = self._approach(d, V_SEARCH)
                    vz = (self.policy_vz if self.brain is not None else
                          np.clip((self.search_alt - p[2]) * 0.4,
                                  -V_DESC, V_DESC))
                    self.send(v[0], v[1], vz)
            elif phase == "centre":
                # Checked FIRST and unconditionally: see BLIND_COMMIT_ALT.
                if p[2] < TOUCHDOWN_ALT:
                    phase = "landed"
                elif p[2] < commit_alt(self.active_id):
                    self.send(0, 0, -V_FLARE)
                elif not seen and self.locked and time.time() - self.t_seen > LOST_S:
                    # Committed: gone longer than a blink, but this pad was read LOCK_FRAMES
                    # times -- keep steering at where it is and keep going down, slower.
                    v = self._approach(self.mark_ned - p[:2], float(np.clip(0.3 * p[2], 0.5, V_CENTRE)))
                    self.send(float(v[0]), float(v[1]), -V_BLIND)
                elif not seen:
                    # Hovering here forever is how three missions burned the
                    # whole time budget: the aircraft glimpsed the tag at the
                    # edge of frame during transit, switched to centring, lost
                    # it again, and then had no way back out. Losing the marker
                    # at altitude means go and look for it again.
                    if p[2] < REACQUIRE_ALT:
                        self.send(0, 0, 0.5)          # back up and re-look
                    elif time.time() - self.t_seen > LOST_S:
                        self._begin_search(p[:2], p[2]); phase = "search"
                    elif self.mark_ned is not None:
                        # A blink, not a loss: keep steering at where the marker was
                        # last seen, holding height until it is back. Commanding a dead
                        # stop on every missed frame was the wobble -- all 90 command
                        # jumps over 1 m/s in the traced flight were seen/unseen flips,
                        # 2.5 m/s <-> 0, and at 40-60 m 27% of frames miss the marker.
                        v = self._approach(self.mark_ned - p[:2],
                                           float(np.clip(0.3 * p[2], 0.5, V_CENTRE)))
                        self.send(float(v[0]), float(v[1]), 0.0)
                    else:
                        self.send(0, 0, 0.0)
                else:
                    # Where the marker is, in world NED metres: the camera ray rotated
                    # by the FULL attitude, then run down to the ground. Heading alone
                    # ignored the 10-30 deg the aircraft tilts while braking -- at 50 m
                    # a 9 deg tilt moves the image centre 7.7 m over the ground. Replayed
                    # on the traced Logan flight: heading-only 5.5 m off (median), full
                    # attitude 0.87 m, best at zero lag once telemetry was fresh.
                    ray = cam_ray(off, self.roll, self.pitch, self.yaw, self.gimbal, self.eye.look_stamp)
                    d_ned = ray[:2] * (p[2] / max(ray[2], 1e-3)) - PAD_OFF_NED[mid]   # to the touchdown point
                    # The pad does not move, so smooth WHERE IT IS (local NED) and steer on
                    # that -- what PX4's landing-target estimator does. Raw, every frame
                    # steered: between 15 and 40 m the detector swapped between the coarse
                    # and medium markers on 133 of 354 steps and single-frame estimates
                    # jittered ~1 m, so the east command flipped sign step to step.
                    raw = p[:2] + d_ned
                    self.mark_ned = raw if self.mark_ned is None else self.mark_ned + MARK_ALPHA * (raw - self.mark_ned)
                    d_ned = self.mark_ned - p[:2]
                    self.last_off, self.last_dned = (ex, ey), (float(d_ned[0]), float(d_ned[1]))
                    # P on position error, D on measured velocity. Without the
                    # damping term the aircraft overshoots the pad, reverses,
                    # and oscillates +/-1.7 m indefinitely -- never satisfying
                    # the centred test, so it never descends. Exactly the
                    # under-damped failure the MuJoCo controller had.
                    # Speed cap scales with height: 1 m/s is fine on short
                    # final but crawls when the tag is first acquired 14 m away
                    # at cruise.
                    v = self._approach(
                        d_ned, float(np.clip(0.3 * p[2], 0.5, V_CENTRE)))
                    vx, vy = float(v[0]), float(v[1])
                    # Descend in proportion to how centred we are, rather than
                    # only when perfectly centred. Requiring perfect centring
                    # left the aircraft parked at 13.7 m -- an altitude where a
                    # 0.5 m tag spans ~19 px and detection is intermittent, so
                    # it could never get centred enough to descend. Sinking
                    # while roughly centred takes it to where the tag is big and
                    # detection is reliable, which makes centring easier.
                    # Descend in proportion to alignment, with a floor so the
                    # aircraft always makes progress. A zero floor and a tight
                    # 300 px tolerance had it holding 20 m for three minutes,
                    # shaving off centimetres of lateral error that were
                    # already far inside the pad.
                    # The floor keeps it descending through the cruise phase
                    # instead of perfecting a centring already well inside the
                    # pad. On short final it is removed: down there a metre of
                    # lateral error is what loses the tag, so being centred has
                    # to be earned before descending further.
                    # Alignment from the SMOOTHED touchdown estimate, in pixel-equivalents:
                    # the raw offset of whichever marker was decoded jumped with every
                    # coarse/medium swap and flicked the descent rate.
                    err_px = float(np.linalg.norm(d_ned)) * F_PX / max(p[2], 0.5)
                    floor = 0.2 if p[2] > REACQUIRE_ALT else 0.0
                    align = float(np.clip(1.0 - err_px / 600.0, floor, 1.0))
                    centred = err_px < CENTRE_PX
                    base = V_DESC if p[2] > 2.5 else V_FLARE
                    vz = -base * align
                    if self.brain is not None and p[2] > 2.5:
                        # the policy owns the descent rate, the alignment gate
                        # still owns permission to use it
                        vz = min(0.0, self.policy_vz) * align
                    self.send(vx, vy, vz)
                    if p[2] < 0.35:
                        phase = "landed"
            elif phase == "abort":
                self.send(0, 0, 1.0)
                if p[2] > CRUISE_Z:
                    break
            elif phase == "landed":
                break

            if self.trace:
                self.trace.write(",".join(str(v) for v in (
                    round(time.time() - t0, 3), round(self.t_boot, 3), *np.round(p, 3), *np.round(self.vel_ne, 3), round(self.climb, 3),
                    round(self.roll, 4), round(self.pitch, 4), round(self.yaw, 4), phase, int(bool(seen)),
                    *self.last_off, *self.last_dned, *np.round(self.last_cmd, 3), *self.pad,
                    mid if seen else -1)) + "\n")
                self.last_off, self.last_dned = (np.nan, np.nan), (np.nan, np.nan)
            if self.rec is not None:
                self.rec.tick(self.eye, phase, p, believed, seen, ids,
                              self.sig.sigma_m, self.pad, t_sim=self.t_boot)

            if verbose and elapsed - last_log > 4.0:
                print(f"    {phase:<8} ({p[0]:+6.1f},{p[1]:+6.1f}) alt {p[2]:5.1f}  "
                      f"marker={('id%d' % mid) if seen else 'n':>4} "
                      f"batt {batt:4.2f} ids={ids}")
                last_log = elapsed
            time.sleep(0.05)

        p = self.pos()
        miss = float(np.linalg.norm(p[:2] - self.pad)) if p is not None else None
        self.disarm()
        out = {"outcome": phase, "miss_m": miss, "search_k": self.search_k,
               "alt_m": None if p is None else float(p[2]),
               "ever_saw_marker": seen_ever, "sigma_m": self.sig.sigma_m,
               "acq_alt_m": acq_alt, "acq_t_s": acq_t, "acq_id": acq_id,
               "acq_span_px": acq_span, "acq_want_px": acq_want,
               "crashed_into": crashed_into, "locked": self.locked, "lock_reads": self.n_lock,
               "forest": [dict(t) for t in TREES],
               "scene": self.sig.scene, "t_s": time.time() - t0}
        return out

    def close(self):
        self._beating = False
        if self.rec is not None:
            path, n = self.rec.close()
            print(f"  video: {path}  ({n} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--pool", default="sigma_pool.npz")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--aim", choices=["mean", "latest"], default="mean",
                    help="aim at the running mean of fixes, or the newest one")
    ap.add_argument("--policy", default=None,
                    help="trained PPO .zip; it then sets descent and search "
                         "radius instead of the sigma rule")
    ap.add_argument("--search-k", type=float, default=2.5,
                    help="search box radius in units of sigma")
    ap.add_argument("--search-alt", type=float, default=SEARCH_ALT,
                    help="search altitude, m (forest study: 18)")
    ap.add_argument("--video", default=None,
                    help="write a side-by-side mission video to this .mp4")
    ap.add_argument("--seed0", type=int, default=0,
                    help="first sigma seed; trial i uses seed0 + i. Lets one "
                         "mission per process still walk the seed sequence.")
    ap.add_argument("--vary-forest", action="store_true",
                    help="respawn a different forest before each mission")
    ap.add_argument("--forest0", type=int, default=100,
                    help="first forest seed; trial i uses forest0 + i")
    ap.add_argument("--wrong-id", type=int, default=None,
                    help="expect a marker id that is NOT in the world, to prove "
                         "the safety layer forces an abort rather than a landing")
    a = ap.parse_args()
    exp = ARUCO_ID if a.wrong_id is None else a.wrong_id
    if a.wrong_id is not None:
        print(f"SAFETY-LAYER TEST: expecting marker id {exp}, world contains "
              f"id {ARUCO_ID}\n")
    results = []
    for i in range(a.runs):
        print(f"--- mission {i + 1}/{a.runs} ---")
        if a.vary_forest:
            replant_forest(a.forest0 + i)
        vid = None if a.video is None else (
            a.video if a.runs == 1 else a.video.replace(".mp4", f"_{i+1}.mp4"))
        ids = PAD_IDS if a.wrong_id is None else [exp]
        mi = Mission(pool=a.pool, expect_id=exp, seed=a.seed0 + i, scale=a.scale,
                     video=vid, search_k=a.search_k, expect_ids=ids, search_alt=a.search_alt,
                     policy=a.policy, aim_mean=(a.aim == "mean"))
        r = mi.run()
        mi.close()
        results.append(r)
        miss = "  n/a" if r["miss_m"] is None else f"{r['miss_m']:5.2f}"
        alt = " n/a" if r["alt_m"] is None else f"{r['alt_m']:4.2f}"
        aq = ("never" if r.get("acq_alt_m") is None else
              f"id{r['acq_id']} at {r['acq_alt_m']:.1f} m / {r['acq_t_s']:.0f} s")
        print(f"  => {r['outcome']}  miss {miss} m  alt {alt} m  "
              f"saw_marker={r['ever_saw_marker']}  first {aq}  ({r['t_s']:.0f} s)\n")
        time.sleep(3)
    # WRITE THE RESULTS FIRST, UNCONDITIONALLY. This used to sit inside the
    # `len(results) > 1` summary block, so running one mission per process --
    # which is how the forest sweep has to run -- wrote no JSON at all. Eighteen
    # trials produced nothing but log lines before this was noticed. The summary
    # is a nicety; the data is not.
    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=1)

    if len(results) > 1:
        ok = sum(1 for r in results if r["outcome"] == "landed"
                 and r["miss_m"] is not None and r["miss_m"] < 1.0)
        ab = sum(1 for r in results if r["outcome"] == "abort")
        print(f"SUMMARY: {ok}/{len(results)} landed on the pad, {ab} aborted, "
              f"{len(results) - ok - ab} other")
        # Acquisition altitude is the quantity the 2.40 m rung is meant to buy.
        aa = [r["acq_alt_m"] for r in results if r.get("acq_alt_m") is not None]
        ms = [r["miss_m"] for r in results if r.get("miss_m") is not None]
        if aa:
            print(f"  first acquisition: median {float(np.median(aa)):.1f} m "
                  f"(min {min(aa):.1f}, max {max(aa):.1f}), n={len(aa)}")
        if ms:
            print(f"  miss: median {float(np.median(ms)):.2f} m")
        print(f"  wrote {RESULT_JSON}")


if __name__ == "__main__":
    main()
