#!/usr/bin/env python3
"""
x500.py -- register the Holybro X500 with the aviary, using the REAL parameters
read from the PX4 Gazebo model on this machine, not estimates.

Source of truth:
  /home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/models/x500_base/model.sdf
  /home/sai/thesis_2.0/PX4-Autopilot/Tools/simulation/gz/models/x500/model.sdf

Measured from those files:
  base_link mass 2.0000 kg + 4 rotors x 0.0161 kg = 2.0644 kg total
  ixx 0.02167, iyy 0.02167, izz 0.04000 kg m^2
  rotor positions (+/-0.1740, +/-0.1740, 0.06) -> arm 0.2461 m
  motorConstant  8.54858e-06   (thrust = kf * omega^2, omega in rad/s)
  momentConstant 0.016         (km/kf)
  maxRotVelocity 1000.0 rad/s  -> 8.55 N per motor, 34.19 N total
  weight 20.25 N -> THRUST/WEIGHT = 1.69

WHAT I HAD GUESSED, AND HOW WRONG IT WAS:
                    guessed      real
  mass (kg)           1.600     2.064    28% light
  ixx                0.02500   0.02167   close
  izz                0.04500   0.04000   close
  arm (m)             0.250     0.246    close
  thrust/weight        2.30      1.69    36% OPTIMISTIC  <- the important one
  km/kf              0.0251    0.0160

The thrust-to-weight error matters most: 1.69 is a far less agile aircraft than
2.3, with much less margin for aggressive manoeuvres. Every controller gain
tuned against the guessed model was tuned for a livelier vehicle than exists.

UNIT NOTE, EASY TO GET WRONG: PX4/Gazebo gives thrust as kf*omega^2 with omega
in RAD/S. The aviary's HOVER_RPM = sqrt(GRAVITY/(4*KF)) treats its "rpm" as
whatever unit makes that identity hold, so KF must be converted to the aviary's
convention rather than copied across.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/RL_projects/MuJoCo-drones-gym"))

from multi_drone_mujoco.envs.base_aviary import DRONE_PARAMS
from multi_drone_mujoco.utils.enums import DroneModel

G = 9.81
MASS = 2.0 + 4 * 0.0161          # base_link + 4 rotors, from the SDF
ARM = 0.2461                     # from rotor poses (+/-0.1740, +/-0.1740)
KF_SI = 8.54858e-06              # N per (rad/s)^2, PX4 motorConstant
MAX_OMEGA = 1000.0               # rad/s, PX4 maxRotVelocity
T2W = 4 * KF_SI * MAX_OMEGA ** 2 / (MASS * G)     # = 1.69

# Convert PX4's rad/s thrust constant into the aviary's "rpm" convention, which
# is defined implicitly by HOVER_RPM = sqrt(GRAVITY / (4*KF)). Choosing the
# hover point to match PX4's fixes the conversion.
HOVER_OMEGA = (MASS * G / (4 * KF_SI)) ** 0.5     # rad/s at hover, = 769.6
HOVER_RPM_AVIARY = HOVER_OMEGA * 60.0 / (2 * 3.141592653589793)   # = 7349
KF = MASS * G / (4.0 * HOVER_RPM_AVIARY ** 2)

X500 = {
    "mass": MASS,
    "arm_length": ARM,
    "thrust2weight_ratio": T2W,
    "ixx": 0.02167, "iyy": 0.02167, "izz": 0.04000,
    "kf": KF,
    "km": KF * 0.016,            # PX4 momentConstant
    "prop_radius": 0.1396,       # rotor visual is 0.2792 m long = diameter
    "max_speed_kmh": 60.0,
    "gnd_eff_coeff": 11.36859,   # non-dimensional, carried over from the CF2
    "drag_coeff_xy": 9.1785e-7 * (MASS / 0.027) ** (2 / 3),
    "drag_coeff_z": 10.311e-7 * (MASS / 0.027) ** (2 / 3),
    "dw_coeff_1": 2267.18, "dw_coeff_2": 0.16, "dw_coeff_3": -0.11,
    "collision_h": 0.12, "collision_r": 0.30, "collision_z_offset": 0.0,
}

X500_MODEL = DroneModel.RACE
DRONE_PARAMS[X500_MODEL] = X500
