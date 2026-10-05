# Measured constants

Every number here was measured on this machine against PX4 SITL + Gazebo, and
every one replaced a value that had been assumed. Re-measure after changing the
airframe, the camera, the marker, or the PX4 build.

| what | value | measured by | stored in |
|---|---|---|---|
| lateral velocity lag | tau = 0.69 s | `shared/px4_gazebo/measure_tau.py` | `shared/px4_gazebo/px4_response.json` |
| vertical velocity lag | tau = 0.23 s | `shared/px4_gazebo/measure_tau.py` | `shared/px4_gazebo/px4_response.json` |
| hover attitude noise | roll ±0.05°, pitch ±0.04° | `shared/px4_gazebo/measure_tau.py` | `shared/px4_gazebo/px4_response.json` |
| camera focal length | fx = fy = 539.94 px | `camera_info` topic | `px4_mission.F_PX` |
| camera→body rotation | orthonormal 2×2 | `shared/px4_gazebo/calib_cam.py` | `shared/px4_gazebo/cam_B.npy` |
| reliable decode size | ≈35 px (≥95% decode) | `shared/px4_gazebo/measure_detect.py` | `shared/px4_gazebo/detect_curve.json` |
| pad → PX4 NED frame | +90°, residual 0.05 m | `shared/px4_gazebo/find_pad.py` | `px4_mission.PAD_XY` |

## The three that mattered most

**Reliable decode size.** `archive/sigma/compare_physics.py` showed this single constant
decides whether the trained policy delivers or aborts: at 20 px it delivered
98%, at 34 px it delivered 0%. Nothing else came close, the velocity lag
included. It is now a measured curve, not a threshold — 14% decode at 25 px,
88% at 30–35 px, 100% above 35 px, and back down to 45% above 400 px where the
marker outgrows the frame.

**Camera→body rotation.** A rotation's singular values must both be 1.0. The
previous hand-fitted matrix had 0.81 and 0.62, so it skewed and shrank every
offset it converted, and it silently baked in the yaw it was fitted at. The
replacement is constrained orthonormal and comes out at 0.964 / 0.947, which is
the model checking itself.

**Hover attitude noise.** Worth measuring because of what it ruled OUT. PX4
holds attitude to ±0.05° in hover, so the visible wobble was never the
firmware's PID gains — tuning `MC_*` would have been tuning the wrong loop.
The fault was outer-loop saturation, fixed in `_approach()`.

## Re-measuring

```
./shared/px4_gazebo/launch_px4.sh &                 # SITL + Gazebo, headless
python3 shared/px4_gazebo/measure_tau.py            # velocity lag + attitude baseline
python3 shared/px4_gazebo/find_pad.py               # Gazebo ENU -> PX4 NED transform
python3 shared/px4_gazebo/calib_cam.py              # camera mounting rotation
python3 shared/px4_gazebo/measure_detect.py         # decode rate vs apparent size
```

`shared/px4_gazebo/calib_cam.py` and `shared/px4_gazebo/find_pad.py` both print a residual. If it is not small,
the fit is wrong — do not use the result.
