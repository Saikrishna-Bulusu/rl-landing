# Fast simulator, MJX twin, Aerial Gym, PX4 SITL: which is best for us?

Written 4 Oct 2026 for Sai. Desk study plus two quick measurements on this laptop. Nothing was installed, the repo was not modified, PX4 and Gazebo were not started.

How to read this: part 1 is the short answer in plain words. Part 2 answers "is our fast simulator a Gymnasium environment". Parts 3 to 5 are the evidence. Part 6 is the comparison table. Part 7 is the recommendation. Part 8 is what would really close the sim-to-real gap. Part 9 lists sources and things I could not verify. Part 10 is three questions to check you can explain it.

---

## 1. The short answer

**Can we use Aerial Gym instead of our own simulators? No, not for either job, and on this laptop it cannot even start.**

Four reasons, in plain words.

1. **It will not run on your GPU.** Aerial Gym is built on NVIDIA's Isaac Gym Preview 4. That needs Python 3.8 or older, and its closed-source physics library has no code for new GPUs. Your RTX 5070 Laptop is a new "Blackwell" chip (compute capability 12.0, which I confirmed with nvidia-smi). An NVIDIA staff member answered a user with an RTX 5080 (same chip family) in April 2026: there is "no workaround", the library is closed and no update will be released. The route that does work is Isaac Lab, a different program with a different install and much heavier hardware advice (16 GB of GPU memory, you have 8 GB).
2. **It does not have the two things we need.** For wind we need PX4's own control loops, because our learner works by nudging PX4's velocity loop. Aerial Gym uses its own textbook controller instead of PX4's, and has no wind model at all (a maintainer's answer to a user: add it yourself with a small change). For dynamic obstacles we need a world about a kilometre wide with a helicopter 1,000 m away. Aerial Gym's worlds are small rooms about 12 x 7 x 5 m, and moving obstacles slow it down a lot (its own example warns about it).
3. **The "flash it into PX4" path is a different kind of policy.** The PX4 neural module replaces PX4's entire control chain with one small network that outputs the four motor commands. Our deciders and our wind learner do not do that. They sit above PX4 and talk to it over MAVLink. Our decider needs about 220 inputs including a 20-second forecast from a planner. That planner runs on the companion computer, not the flight controller.
4. **Its sim-to-real evidence is real but narrow.** Flights under a motion-capture system, one 1.2 kg drone tracking a square (the PX4 module paper), and a cluttered corridor at up to 2.83 m/s with an ArduPilot drone (the Aerial Gym paper). Neither paper describes a wind test or an outdoor flight (my reading of the text, not a claim that none was done), and neither does satellite-image localisation. It does not test what our stack's real gaps are.

**What it is good for:** (a) a published recipe to measure your own drone (weight, inertia, thrust, motor lag) that we should copy when hardware arrives; (b) a citation showing that simulation-trained networks can run on a PX4 flight controller; (c) possibly later, training small camera-based avoidance policies for the "unknown movers" direction, if we move to Isaac Lab and a bigger GPU.

**What actually closes our gap:** not a new simulator but climbing the ladder with measurements. Fast sim, then MJX (already matches Gazebo), then PX4 SITL with the real companion computer in the loop, then hardware hover logs in real wind to refit the MJX twin. See part 8.

---

## 2. Is our fast simulator a Gymnasium or PettingZoo environment?

First, what those names mean, in everyday terms. Gymnasium is a standard plug shape for simulators. If a simulator has the same plug (`reset()` and `step(action)` that return observation, reward, terminated, truncated, info), then any training library can be connected to it. PettingZoo is the same idea for several learners acting in one world, with dictionaries keyed by agent name.

**Neither of our simulators has that plug.** Both have the same *idea* (reset, then step with an action), but they are batched custom code.

| | Fast simulator `shared/fast_sim/urban_env.py` | MJX twin `conditions/wind/land_env.py` + `conditions/wind/x500_mjx.py` |
|---|---|---|
| Language and numbers | PyTorch, `UrbanEnv` class, thousands of drones as tensors | JAX, plain functions on a state dictionary, `jax.vmap` over 4,096 drones, `lax.scan` over time |
| reset | `env.reset()` starts all `n` flights together, returns a dictionary of tensors | `reset(key, w10)` returns one state; vmapped |
| step | `env.step(cmd)` takes an (n, 3) velocity command in m/s, returns only the observation dictionary | `step(st, act, key, ...)` returns `(new_state, reward, done)` |
| reward | not in the environment; computed in `shared/fast_sim/urban_rl.py` (`generation()`) | inside `step` |
| end of flight | `env.done`, `env.outcome`, `env.summary()`; no reset of single drones, finished drones are frozen until everyone is done ("synchronous generations") | `done` mask. In `ppo.py` training a finished flight is replaced at once by a fresh one (`vreset` + `where_tree`: an automatic reset); only in `land_env.evaluate()` are finished flights frozen with `jnp.where` |
| spaces | none declared | none declared |
| random numbers | its own `torch.Generator`s so paired comparisons line up | explicit JAX keys |
| action meaning | the learner picks 1 of 14 candidates; the candidate list is built by a guidance generator outside the env (`urban_agents`), held for 2 steps (0.5 s) | 2 continuous feed-forward accelerations plus a 5-way vertical choice, every 0.04 s |

Closest standard name: the MJX twin follows the "functional" style used by JAX libraries such as Brax and gymnax (my characterisation, not checked against their docs). The fast simulator is a hand-rolled vector environment.

Old fact worth knowing: the earlier single-drone files in the repo (`archive/sigma/approach_env.py`, `archive/sigma/outdoor_env.py`, `archive/tello/cage_env.py`) *are* `gymnasium.Env` classes. Only the later GPU simulators dropped the plug, because one drone per Python call is far too slow for thousands of flights.

**PettingZoo does not fit.** It is for several *learning* agents. In our world exactly one agent learns (our drone). The helicopter, other drones, people and the magpie are scripted. PettingZoo's dictionary-per-agent format also works against GPU tensors.

**What wrapping as Gymnasium would take** (effort is my estimate, not measured):

- Write a class subclassing `gymnasium.vector.VectorEnv` (the installed Gymnasium 1.3.0 has it, with `AutoresetMode.NEXT_STEP / SAME_STEP / DISABLED`). It needs `num_envs`, declared observation and action spaces, and `reset(seed, options)` and `step(actions)` returning `(obs, reward, terminated, truncated, infos)`.
- For the fast simulator: move the reward from `urban_rl.py` into the environment, bring in the 14-candidate generator and the 2-step hold, put the action mask for duplicate candidates in `info`, and use `AutoresetMode.DISABLED` (our flights are whole generations, there is no single-flight reset). Roughly 1 to 2 days.
- For the MJX twin: wrap with Gymnasium's `JaxToTorch` or `ArrayConversion` wrapper, which exist in 1.3.0 (checked). But importing them fails in the `mujoco_drones` environment today, because they need the package `array_api_compat`, which is not installed there (I did not install it). About 1 day. Not needed for our own JAX PPO. Note the MJX training loop already auto-resets finished flights (see table), which matches Gymnasium's "same step" style of reset more than the "disabled" style used for the fast simulator.

**What it buys:** the ability to plug in off-the-shelf trainers (rl_games, Sample Factory, CleanRL), standard logging, and an easier story for reviewers. **What it does not buy:** more speed, a smaller sim-to-real gap, or better results. Our DAgger-then-PPO with the planner's forecast as an input is custom anyway. I would do it only if a reviewer or a colleague asks to reuse the environments.

---

## 3. What Aerial Gym is, and what it is not

Source: its README, its documentation site, and its paper (Kulkarni, Rehberg, Alexis, "Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots", IEEE Robotics and Automation Letters, vol. 10, no. 4, pp. 4093-4100, April 2025, arXiv 2503.01471, DOI 10.1109/LRA.2025.3548507).

**What it is.** A GPU simulator for many multirotors at once, from NTNU's Autonomous Robots Lab. It sits on NVIDIA Isaac Gym for physics, and adds its own depth/LiDAR ray-casting built on NVIDIA Warp. Controllers (the Lee geometric controller family) run on the GPU in PyTorch. Training examples use rl_games, Sample Factory and CleanRL. License BSD-3.

**API.** Its `Task` classes follow the Gymnasium shape (reset, step returning obs, rewards, terminations, truncations, infos) but everything is batched torch tensors. The paper says its environments "conform to the Gymnasium standard", but in the code (checked on `main`) the base class `BaseTask` is a plain `ABC`, the PX4 task imports the old `gym.spaces`, and the rl_games runner's `ExtractObsWrapper` subclasses the old `gym.Wrapper`. So it is "Gymnasium-shaped and batched", not a `gymnasium.vector.VectorEnv` subclass. It also has `reset_idx(env_ids)` for resetting single drones, which ours lacks.

**Physics.**
- Rigid body in Isaac Gym's PhysX engine, rotors as forces, motors as first-order lag with separate rise and fall time constants.
- The x500 config in the repo uses the same thrust constant (8.54858e-6) and the same 12.5 ms up / 25 ms down motor lag that PX4's Gazebo model uses (we use the same numbers in `x500_mjx.py`). Good, but that is the motor only.
- Controllers: Lee geometric attitude, rate, velocity, position, acceleration. **Not PX4's cascade** (PX4 uses PID loops with tilt limits, thrust margins and its own allocation).
- Drag: a simple linear plus quadratic term in the robot config, off by default for the x500.

**Wind and disturbance.** The x500 config has a `disturbance` block, and the code (`apply_disturbance` in `base_multirotor.py`) can add, with some probability per step, a random force and torque. In the shipped x500 config this block is switched off (`enable_disturbance = False`, probability 0, all limits 0). Limits of 0.1 N and 0.05 N m appear in the same file, but inside the `robot_asset` block, which the disturbance code does not read, and the PX4 sim-to-real task config does not turn it on (checked, corrected from the first draft). There is no wind field, no height profile, no turbulence filter, no gusts. A user asked about wind (GitHub issue #47, July 2025, still open). The maintainer's reply: subtract a wind velocity inside the existing drag term, and you must keep and update a wind tensor yourself. So it is a recipe, not a feature.

**Dynamic obstacles.** Possible. You write a velocity into the obstacle's state tensor each step (issue #2 reply, and `examples/dynamic_env_example.py`). But the example itself warns "a dynamic environment will slow down the simulation by a lot", and the docs say dynamic scenes need Isaac Gym's own cameras, because the Warp depth path must rebuild its bounding-volume tree whenever meshes move (docs and issue #22 reply). The `dynamic_env` bounds are about 12 x 7 x 5 m. There is no notion of a 1 km world, a helicopter at 1,000 m, or CASA-style rules.

**Static obstacles.** Strong: random trees, panels, thin objects, walls, forest-like scenes, randomised per environment, inside those small rooms.

**Sensors.** Depth cameras, segmentation, LiDAR (Ouster, OS-Dome) and small time-of-flight sensors by Warp ray-casting, an IMU model with noise and bias drift. Paper, RTX 3090: roughly 1,600 to 2,750 frames per second for 270 x 480 depth images over 128 to 2,048 environments. Throughput falls as image size grows. Depth images are the speed limit if you use them.

**Throughput claims.** Paper Table I, RTX 3090: about 4.4 million physics samples per second at 65,536 instances (4.43 x 10^6), with obstacle-free scenes and constant RPM commands. The same table lists the L2F simulator at 14,063 x 10^6 samples per second at 65,536 instances (a number the L2F authors reported, on a laptop GPU), so Aerial Gym is not the fastest pure-physics simulator; its selling point is the sensors. Our MJX twin is quoted in `conditions/wind/README.md` at about 21 million physics steps per second on 4,096 drones, with PX4's full cascade at 250 Hz, so the two numbers are not the same unit of work. **Not reproduced:** I re-ran a hover-only test today (4,096 drones, 64 decisions of 10 physics steps each, `control_step` under `jax.vmap`, `nice -n 10`, GPU 9% busy) and got about 5.5 million physics steps per second (0.48 s per 64-decision round), roughly 4 times below the README. Treat 21 M as an unconfirmed README figure until someone reruns the README's own benchmark. The README of Aerial Gym and its RL-training docs page say state-based control policies train "in under a minute" (RTX 3090) and vision-based navigation "in about an hour" (RTX 3090 for the depth-image task; about an hour on a laptop RTX 3080 Ti for the LiDAR navigation policy of the Unified Autonomy Stack).

**Maturity.** GitHub's "updated" stamp is 2 Oct 2026, but that moves with stars and issue activity: the last code push was 28 Jun 2026. 776 stars, 22 open issues (both checked via the GitHub API today). Two issues opened 5 Sept 2026 report possible bugs in the navigation reward (#64, #65). I did not verify them. Treat as "someone should check before trusting its navigation reward".

### 3.1 Can it run on your machine?

| Question | Answer | Source |
|---|---|---|
| Does Isaac Gym Preview 4 run on an RTX 5070 Laptop (Blackwell, sm_120)? | No. Its `libPhysXGpu_64.so` has kernels only up to SM_80 and no PTX to recompile. NVIDIA staff: "There is no workaround for this", the binary is closed, no update will come. (Test case was an RTX 5080; same architecture family as your 5070.) | NVIDIA forum thread 367941 (Apr 2026) |
| Python version | Isaac Gym Preview 4 needs Python 3.6, 3.7 or 3.8 (NVIDIA's download page also lists Ubuntu 18.04 or 20.04, driver 470+, and a minimum of 8 GB of GPU memory). Aerial Gym's install page says python=3.8, PyTorch 1.13.1 + CUDA 11.7 (or "the newest version of PyTorch your driver supports"). Ubuntu 24.04 ships Python 3.12. Blackwell support arrived in PyTorch 2.7 with CUDA 12.8 (its release notes), and PyTorch 2.7.0 requires Python 3.9 or newer (PyPI metadata; the first draft said 3.10, which was wrong). 3.9 or newer is above Isaac Gym's 3.8 ceiling, so the two cannot be reconciled. A user report on an RTX 5090 (unitree_rl_gym issue #96, 19 Apr 2026, open) shows the same Python-3.8 versus sm_120 PyTorch clash. | Aerial Gym docs, install page; NVIDIA Isaac Gym download page; PyTorch v2.7.0 release notes; PyPI torch 2.7.0; unitree_rl_gym issue #96; Isaac Gym page says "legacy software ... no longer supported" |
| Is there an Isaac Lab version? | Partly. Isaac Lab v2.3.2 (Isaac Sim 5.1) added a multirotor asset, a thruster actuator and a manager-based "ARL drone" task, contributed by the Aerial Gym authors (PR #3760). That is the drone *building blocks*. The Isaac Lab multirotor API page lists no drag, wind or built-in controllers. I did not check whether the Warp renderer, the Lee controllers and the environments were ported. | Isaac Lab release notes; Isaac Lab v2.3.2 API page; Aerial Gym docs home |
| Can Isaac Lab run on this laptop? | Uncertain. Isaac Sim 5.1's requirements page names Linux driver 580.65.06 as the tested version (you have 580.178.04, newer) and lists 16 GB of GPU memory as the minimum, warning that less "may be insufficient" for complex rendering. You have 8 GB. State-only training without rendering may fit; I could not test it (no installs allowed). The newest Isaac Sim docs name driver 595.58.03 as the tested version, and the Isaac Lab 3.0 beta notes require Python >=3.12,<3.13 (two separate pages). | Isaac Sim 5.1.0 and latest requirements pages; Isaac Lab release notes |
| What does the Aerial Gym README say? | A cached copy says Isaac Lab support is "under development"; the docs home and the Isaac Lab notes say it is available. The FAQ still says "working on". The sources disagree, so check the repo before relying on it. | README, docs home, FAQ |

---

## 4. The PX4 neural-network route (the "flash it into the firmware" part)

Sources: arXiv 2505.00432 (Hegre, Rehberg, Kulkarni, Alexis, "A Neural Network Mode for PX4 on Embedded Flight Controllers", 4-page workshop paper, ICRA 2025 Workshop on 25 Years of Aerial Robotics, 1 May 2025), PX4 documentation source on GitHub (`docs/en/neural_networks/`), Aerial Gym docs page 9 (sim2real), Aerial Gym issue #49.

| Question | Answer |
|---|---|
| What is it called? | `mc_nn_control` module in PX4, with a flight mode named "Neural Control". |
| Status and version | Marked **experimental**, badge "PX4 v1.17". It was developed in a fork (branch `for_paper`) and later merged into PX4's main branch (issue #49 reply, Aug 2025). PX4 v1.17.0 was released 13 May 2026; v1.18.0-rc1 on 10 Sept 2026. |
| Build requirement | Ubuntu 24.04 or newer (PX4 docs). You have 24.04. |
| Boards | Pre-made build targets: `px4_sitl_neural` (simulation), `px4_fmu-v6c_neural` (Pixhawk 6C) and `mro_pixracerpro_neural`. Tested on a Pixracer Pro (STM32H743, Cortex-M7, 460 MHz, 2 MB flash, 1 MB RAM). The PX4 docs say the module takes roughly 50 KB (of flash) and boards that are nearly full must drop other modules; the paper separately says the network was made small enough for the 50 KB of RAM available to it. |
| What the network sees | 15 numbers: position error (3), the first two rows of the rotation matrix (6), linear velocity (3), angular velocity (3). |
| What it outputs | 4 numbers: one thrust per motor, rescaled to PX4's normalised motor commands (`actuator_motors`). |
| What it replaces | **The whole chain**: position, velocity, attitude and rate controllers plus the control allocator (the "mixer"). PX4's EKF, mode logic, failsafes and drivers stay. You can switch back to a normal flight mode in the air. |
| How fast | The module runs each time a new gyro sample arrives; 650 Hz on the Pixracer Pro (paper). Network inference 93.4 microseconds; whole loop 137.6 microseconds. (A different experiment in the Aerial Gym paper ran a motor policy on a ModalAI Voxl 2 Mini, in a custom PX4 module, at 250 Hz, under a Qualisys motion-capture system.) |
| Network size | 2 hidden layers, 64 and 32 neurons, ReLU (paper). The module checks at start that the model has exactly 15 inputs and 4 outputs. |
| How it gets onto the board | Train in Aerial Gym (rl_games PPO), convert PyTorch to TensorFlow Lite Micro (TFLM), turn the file into a C array with `xxd`, paste into `control_net.cpp`, rebuild, reflash. The network is compiled into the firmware: new network = new build. |
| Setpoints | Position setpoints from a PX4 `TrajectorySetpoint`. The PX4 comparison table says `mc_nn_control` does **not** accept offboard setpoints; RAPTOR (another PX4 module, v1.18) does, over MAVLink. |
| Known bug (Aerial Gym docs, sim2real page; not in the PX4 pages) | The network can output NaN motor values on a real drone if it gets no trajectory setpoints; start the setpoint module first. |
| Evidence it transfers | One 1.2 kg drone, motion-capture indoors, tracking a square. Position and velocity "largely consistent" between sim and real, but motor commands differ more, blamed on wrong motor time constants. No wind, no outdoor, no GNSS-denied test. |
| Other module | RAPTOR (`mc_raptor`, PX4 v1.18, experimental): 2,084-parameter network, trained with RLtools (not PyTorch), outputs motor commands, setpoints over MAVLink, loads weights from the SD card, claimed tested on 10+ real quadrotors from 32 g to 2.4 kg (PX4 docs). Its paper is Eschmann, Albani, Loianno, "RAPTOR: A Foundation Policy for Quadrotor Control", arXiv 2509.11481 (Sept 2025, revised April 2026); I read only its abstract, which says 10 real quadrotors (32 g to 2.4 kg), a 2,084-parameter policy, and tests that include indoor/outdoor flight and wind disturbances. The PX4 RAPTOR page reports figure-eight tracking up to 12 m/s in "moderate winds (~5 m/s)". So unlike `mc_nn_control`, RAPTOR does have outdoor and wind evidence, but it is a low-level motor policy too (not our layer) and was not checked further. |

**Contrast with our learners.**

| | PX4 neural module | Our decider | Our wind flyer |
|---|---|---|---|
| Runs on | Flight controller (STM32) | Companion computer | Companion computer |
| Rate | about 650 Hz | every 0.5 s | 25 Hz (0.04 s) |
| Outputs | 4 motor thrusts | 1 of 14 velocity commands | acceleration feed-forward (<= 3 m/s^2 per axis) on PX4's velocity loop, plus a vertical choice |
| Replaces | the whole PX4 cascade | nothing in PX4 | nothing; adds to PX4's velocity loop |
| Inputs | 15 floats | 221 floats in the "eyes" variant, which adds the planner's 20 s forecast per candidate (comment and a run of `urban_rl.inputs`); 165 without it | 50 floats (`land_env.OBS_DIM` = 50; `conditions/wind/README.md` still says 49, which is stale) |
| Needs to know | own state only | map, intruders, planner rollouts | pad estimate, house, tree |

Technical note: the wind flyer's network (80,649 parameters in the policy, counted from `ppo.init_params`: 50-256-256-7 plus 2 log-std values) would physically fit on a flight controller, and the decider's (about 126,000 for the policy with planner view, 112,000 without; counted from `urban_rl.Net`; the value network is separate) too. The issue is not size. Their inputs come from sensors and planners that live on the companion, and they act through PX4's velocity interface. Moving them onto the flight controller would mean re-writing the sensing there, and PX4's stock module takes no outside setpoints. Our design is the same pattern NTNU itself uses for its other policies: a Sample-Factory-trained policy (the `rl_nav` repository: a ROS 2 node that outputs body-frame accelerations and yaw rate, and imports `mavros_msgs`; the README does not itself say "companion computer", that part is my reading). In the same spirit the Aerial Gym paper's velocity and acceleration command policies flew on an ArduPilot drone with onboard odometry, outside the flight controller.

---

## 5. Why wind and dynamic obstacles are different problems

**Dynamic obstacles** are about *information and time scale*. The decider acts every 0.5 s. What matters is when it sees the helicopter, what the planner forecasts, what the rules force. The drone's body dynamics hardly matter at this level, so our fast simulator replaces them with a first-order lag measured from PX4 (time constants 0.69 s horizontal, 0.23 s vertical). The gap that matters is sensing and behaviour: detection ranges, localisation error, real helicopter and bird behaviour. A rotor-accurate simulator adds nothing here, and a small-room simulator takes away the thing that matters (km scale).

**Wind** is about *the body and PX4's control loops*. The learner adds feed-forward to PX4's velocity loop, so the simulator must reproduce PX4's loops (tilt limit, thrust margin, saturation, allocation), rotor lag and drag in wind. That is exactly what `x500_mjx.py` does, and its README shows it matches PX4 in Gazebo (step-response times within about 0.02 to 0.05 s, tilt 44.1 vs 45.6 deg at 8 m/s, hover tilt in 15.8 m/s wind 22.5 vs 22.6 deg). Aerial Gym would give us a faster body simulator but without PX4's loops and without wind, so we would first have to rebuild both.

Evidence the ladder works: on 30 Sept a learner with a harsh reversal price pulsed climb/descend about 1.5 times a second, held PX4 at its thrust limit and sank in Gazebo (comment in `land_env.py`). That was found by the Gazebo rung, with real PX4 code. No amount of physics speed in a simulator without PX4's code would have found it.

---

## 6. Comparison table

"Port" means moving our work to that tool. Items marked (est.) are my estimates, not measurements. "Measured" values for our fast sim were taken on 4 Oct 2026 on this laptop with other GPU users present, so they are rough.

| | Our fast sim (`urban_env.py`) | Our MJX twin (`x500_mjx.py`, `land_env.py`) | Aerial Gym (Isaac Gym build; Isaac Lab port noted) | PX4 SITL + Gazebo |
|---|---|---|---|---|
| API type | Custom batched torch class; `reset()`/`step(cmd)` return an obs dictionary; no rewards, spaces or autoreset | Pure JAX functions, vmapped; `(state, reward, done)` | Gymnasium-shaped batched torch Tasks (`reset`, `step`, `reset_idx`); not a `VectorEnv` subclass; rl_games / Sample Factory / CleanRL wrappers | Real programs talking MAVLink; one flight at a time |
| Physics | Kinematic: velocity follows the command with a first-order lag (0.69 s xy, 0.23 s z), measured on PX4 | Rigid body in MJX, rotor model, drag, PX4 cascade at 250 Hz | Isaac Gym PhysX rigid body, rotor forces, first-order motors, Lee controllers | Gazebo physics + real PX4 code |
| PX4 faithfulness | Low-order fit to PX4's response | High: PX4's position, attitude, rate and allocation laws re-written; checked against Gazebo step responses | Low to medium: same motor constants as PX4's Gazebo x500, but not PX4's controllers; PX4 route replaces the cascade | Highest: the actual firmware (SITL) |
| Wind | None in the env. (Gazebo skies showed wind harmless for the full stack: 10/10 flights, project notes) | Mean wind with log profile, MIL-F-8785C turbulence, 1-cosine gusts; same model as the Gazebo plugin | No wind field. Random force/torque kicks; a user-added drag-relative-to-wind recipe (issue #47, open) | Gazebo wind plus our drag plugin |
| Dynamic obstacles at km scale | Yes: helicopter seen to 1,000 m, drones to 200 m, people, cars, magpie, fog, crane; CASA rules | No (a 30 m yard) | No: rooms about 12 x 7 x 5 m; moving meshes slow it; no km scale | Yes but only tens of scripted flights |
| Static obstacles | Logan LiDAR height map plus unmapped boxes | Yard with house (10 x 8 m) and tree | Strong: random trees, panels, walls, forests (small rooms) | Gazebo Logan world (photo floor, draped roofs, cranes, poles) |
| Sensors | Probability models: detection curve, GeoAnchor error replay, idealised voxel map, measured bird rig | Pad estimate noise (0.03 + 0.004 h m, 0.05 m bias), velocity noise | Warp depth, LiDAR, segmentation, IMU; fast for small images | Real rendered camera and depth (slow, YOLO in the loop) |
| Scale | Measured: 32,768 drones, peak 182 MiB of PyTorch tensors (about 200 MiB as seen by the driver); 4,096 drones 77 MiB | 4,096 drones (README) | Paper: 65,536 instances (RTX 3090); depth images limit it | 1 flight; never two SITLs at once (project notes) |
| Throughput on this 8 GB RTX 5070 Laptop, Ubuntu 24.04 | Measured and re-measured by the checker (4 Oct, `StraightLander`, corridor world, 200 lock-steps each): about 300 lock-steps/s at 1,024, 4,096 and 32,768 drones on a quiet GPU (328 / 302 / 288), and 91 to 280 when other jobs were using the GPU. A lock-step is 0.25 s of flight, so 4,096 drones give about 1.2 million drone-steps/s and 32,768 give about 9 million; the speed is flat in the batch size, so the sim is launch-bound and bigger batches are cheaper per drone. One 300 s flight is 1,200 lock-steps (0.5 s decisions are every second lock-step). The first draft's 98 lock-steps/s was measured under GPU contention. | README: about 21 million physics steps/s at 4,096 drones; a PPO round of 4,096 x 64 decisions about 0.3 s. Not reproduced: a hover-only test by the checker gave about 5.5 million steps/s (0.48 s for the 64-decision rollout alone) | Cannot run (Isaac Gym). Paper numbers are for an RTX 3090 and are not comparable. Isaac Lab route untested. | About 1.3 to 3.4 s of wall time per simulated second (project notes) |
| Sim-to-real evidence | Inputs measured from PX4 and Gazebo flights; no hardware | Matches Gazebo flight by flight; no hardware | Real flights: Aerial Gym paper: ArduPilot drone with onboard odometry (velocity and acceleration policies, corridor navigation up to 2.83 m/s), plus a Voxl 2 Mini motor policy under motion capture; PX4-module paper: 1.2 kg drone under motion capture. Neither describes wind or outdoor tests | Real firmware; no real air, no real sensors |
| Level of the stack it trains | Decision layer (14 candidates, 0.5 s) | Flying layer (25 Hz feed-forward on PX4's velocity loop, vertical choice) | Motor thrusts, accelerations or velocity setpoints, depth navigation | Validates all layers, trains none |
| Porting effort for us | Gymnasium wrapper about 1 to 2 days (est.). Port to Aerial Gym: not meaningful | Gymnasium wrapper about 1 day (est.). Port to Aerial Gym: rebuild PX4's cascade in torch, add wind, move to Isaac Lab, redo the Gazebo validation: a multi-week project with a bigger GPU needed (est.) | n/a | Already built |

---

## 7. Recommendation

### For wind: keep the MJX twin. Do not move to Aerial Gym.

Reasons.
1. Our learner adds to PX4's velocity loop. Only the MJX twin reproduces that loop, and it is already checked against Gazebo.
2. Aerial Gym lacks both pieces (PX4's cascade, a wind model), so a move would first mean rebuilding them, then redoing the Gazebo validation.
3. Isaac Gym Preview 4 does not run on your GPU, with no workaround. The Isaac Lab route needs a different install and more GPU memory than you have.
4. The mc_nn_control route would turn the wind flyer into a different research question: an end-to-end motor policy that replaces PX4's loops, giving up the safety net that PX4's cascade provides. Its evidence is indoor with no wind. High risk for the ACRA timeline.
5. The real wind gaps are not inside any simulator: the wind model's realism (MIL-F-8785C versus a Logan back yard), the body drag area (0.04 m^2 is an assumption), and PX4's estimator in gusts. Only real hover logs in wind close these (part 8).

Cheap upgrade worth doing: stress the twin with the unknowns (vary drag area, turbulence strength, motor lag), as Aerial Gym's sim-to-real recipe suggests, and report the spread.

### For dynamic obstacles: keep the fast simulator. Do not port.

Reasons.
1. The decision layer's gap is sensing and behaviour at kilometre scale. Aerial Gym's world is a small room, and moving obstacles are its slow path.
2. It has no CASA rules, helicopter model, localisation-error replay, or planner interface. We would write all of it again.
3. Our fast simulator is also cheap: it uses about 200 MiB of GPU memory at 32,768 drones (182 MiB of tensors at peak), which leaves room for other jobs on the 8 GB card.
4. The Gazebo rung already tests the same stack with real PX4, real depth and real camera images.

Optional: write the Gymnasium wrapper (part 2) if you want to reuse trainers or share the environments. It will not change results.

### When would Aerial Gym become interesting?

Only if the project adds a policy that learns from *depth images* (for example the "unknown movers" step in the hierarchical RL plus visual odometry direction). Then Aerial Gym's Warp depth and its Isaac Lab drone task are a real candidate, but only after (a) a bigger GPU or a confirmed 8 GB fit, (b) a check that the Isaac Lab port includes what you need. Until then, citing it as related work is enough.

---

## 8. What would actually close our sim-to-real gap

The ladder: fast sim, then MJX, then SITL, then (HITL), then hardware. For each rung: what it catches, what it misses, what to do next.

| Rung | Catches | Misses | Next step |
|---|---|---|---|
| 1. Fast sim | Logic errors, bad rewards, rule conflicts, thousands of encounters | Real sensor errors, real intruder behaviour, timing | Keep. Add latency to every decider (the memory note "offline checks need latency"). Log which inputs came from measurements and which are assumed (the header already does). |
| 2. MJX twin | Control-loop and wind interactions, thrust-limit effects | Wind realism, drag area, real estimator, delays | Already matches Gazebo. Report sensitivity to drag, turbulence, motor lag. Plan to refit with hardware logs. |
| 3. PX4 SITL + Gazebo | Real PX4 code: EKF, commander, failsafes, mode logic; camera and depth in the loop | Real air, sensor noise, vibration, GNSS-denied reality | Already used for tens of flights. Count disagreements with the fast sims and keep that table: it is your measured gap. |
| 4. Companion-in-the-loop (new, cheap) | Run the same decider and planner on the real companion computer (Jetson or similar) against SITL over the real MAVLink link. Catches timing, memory, a 2 s planner stalling, message loss. | Same as SITL | Do this before hardware. No new simulator needed. |
| 5. HITL | Real flight-controller hardware running PX4 | Little for us | Low value: PX4's docs say HITL for multicopters is supported only with Gazebo Classic and is community-maintained; our stack uses the newer Gazebo. A flight controller on the bench with a real offboard link (SIH on hardware, I did not read that page) may be the cheaper check. |
| 6. Hardware | Everything real | Needs flight approval and a safe site | Start with tethered or net-protected hover in real wind. Log mass, inertia, hover RPM, motor lag with Aerial Gym's recipe (hover flight, read RPM, thrust constant = hover thrust per motor / RPM squared). Refit MJX parameters. Record real wind with an anemometer. Then test the wind flyer at low speed. |

**The three gaps that matter for our paper, and which rung closes each:**

1. Localisation (GeoAnchor on real 2021 imagery versus real camera images): only hardware or recorded real flight video. Not Aerial Gym, not any simulator.
2. Wind and drag: real hover logs in wind (rung 6), then refit rung 2.
3. Real intruders (helicopter, other drones, magpie): cannot be simulated away; you report conditional results, as the fast simulator's header already says (encounter sets, not days in Logan).

**Where Aerial Gym fits:** (a) the system-identification recipe for rung 6; (b) a citation that simulation-trained policies run on PX4 hardware (`mc_nn_control`, tracking only, indoors); (c) a possible depth-image training ground later; (d) an optional independent cross-check of the MJX twin, but only after an Isaac Lab setup that your 8 GB may not support. It does not belong in the main path.

**Suggested sentence for the paper (limitations):** "Our simulators are validated flight by flight against PX4 SITL in Gazebo; no real-flight data exists yet. We therefore treat all reported rates as simulation results and identify wind, drag area and real sensor error as the parameters to refit from the first hardware logs."

---

## 9. Sources, and what I could not verify

Opened and used:
- Aerial Gym repo and README: https://github.com/ntnu-arl/aerial_gym_simulator (code files read via raw.githubusercontent.com: `x500_config.py`, `base_multirotor.py`, `position_setpoint_task_sim2real_px4.py`, `dynamic_env_example.py`, `dynamic_environment.py`, `base_task.py`)
- Aerial Gym documentation: https://ntnu-arl.github.io/aerial_gym_simulator/ (home, getting started, robots and controllers, simulation components, RL training, sensors, sim2real `/9_sim2real/`, FAQ)
- Aerial Gym paper: Kulkarni, Rehberg, Alexis, IEEE RA-L 2025, arXiv 2503.01471, DOI 10.1109/LRA.2025.3548507 (DOI from the repo's citation block)
- PX4 neural module paper: Hegre, Rehberg, Kulkarni, Alexis, arXiv 2505.00432
- PX4 docs source (main branch): https://github.com/PX4/PX4-Autopilot/tree/main/docs/en/neural_networks (`index.md`, `mc_neural_network_control.md`, `nn_module_utilities.md`, `tflm.md`, `raptor.md`), and `docs/en/simulation/hitl.md`. The hosted docs site (docs.px4.io) blocked the fetch tool, so I read the same pages as source files.
- PX4 releases: https://github.com/PX4/PX4-Autopilot/releases (v1.17.0, v1.18.0-rc1)
- Aerial Gym issues #1, #2, #13, #22, #35, #47, #49, #64, #65: https://github.com/ntnu-arl/aerial_gym_simulator/issues
- NVIDIA developer forum, "Isaac Gym Preview 4 incompatible with RTX 5080 (Blackwell SM_120)": https://forums.developer.nvidia.com/t/isaac-gym-preview-4-incompatible-with-rtx-5080-blackwell-sm-120-libphysxgpu-64-so-missing-sm-120-kernels/367941
- Isaac Gym page (deprecation): https://developer.nvidia.com/isaac-gym
- Isaac Lab release notes: https://isaac-sim.github.io/IsaacLab/develop/source/refs/release_notes.html ; multirotor asset API: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab_contrib/isaaclab_contrib.assets.html
- Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html and the "latest" page
- NTNU companion-computer deployment: https://github.com/ntnu-arl/rl_nav and https://github.com/ntnu-arl/unified_autonomy_stack (README only; paper arXiv 2605.12735 not opened)
- Our code: `shared/fast_sim/urban_env.py`, `shared/fast_sim/urban_rl.py`, `conditions/wind/x500_mjx.py`, `conditions/wind/land_env.py`, `conditions/wind/ppo.py`, `conditions/wind/README.md`; Gymnasium 1.3.0 and PettingZoo 1.27.0 as installed.

Measured by the first author today: `urban_env.UrbanEnv` with `StraightLander`, corridor world, n = 1,024 / 4,096 / 16,384 / 32,768, `nice -n 10`, with 1.8 to 2.0 GB of GPU memory already used by another process. The checker re-ran n = 1,024 / 4,096 / 32,768 (200 steps each, three runs): 328 / 302 / 288 lock-steps/s on a quiet GPU and 91 to 280 under contention, so the first draft's 98 lock-steps/s at 4,096 was a contended figure and the throughput depends heavily on what else is on the GPU. MJX numbers are quoted from the README; the checker's own hover-only re-run gave 5.5 million steps/s, not 21 million (unresolved).

Not verified or uncertain:
- Whether Isaac Lab (Isaac Sim 5.1) runs usefully on 8 GB; whether the Isaac Lab port includes Aerial Gym's Warp renderer, Lee controllers and environments; whether Isaac Lab has a built-in wind model (I found none).
- The two Sept 2026 bug reports (#64, #65) are unconfirmed reports.
- Other users' RTX 50 experience with Isaac Gym: one NVIDIA-staff answer, on an RTX 5080 (opened and confirmed). A second report (unitree_rl_gym issue #96, RTX 5090) was opened by the checker: it shows the Python 3.8 / sm_120 PyTorch clash, not the PhysX-kernel problem, and has no maintainer answer.
- RAPTOR's paper claims: read only through the PX4 documentation and the arXiv abstract.
- All "est." porting efforts and the parameter counts (my arithmetic) are estimates.
- Table I in the Aerial Gym paper does list L2F at 14,063 x 10^6 samples/s at 65,536 instances (reported by the L2F authors, laptop GPU); the first draft said it mixed units and did not quote it. It is now quoted in part 3.

---

## 10. Check yourself (try to answer in your own words)

1. Why does our wind learner need PX4's control loops in the simulator, while our decider does not?
2. What is the difference between "the policy runs inside PX4's flight controller" and "the policy runs on the companion computer and sends velocity commands"? Which one are we, and why can't we flash our decider?
3. Aerial Gym's sim-to-real flights were indoors with motion capture. Name one of our real gaps that those flights say nothing about.


---

## Verification

Checked 4 Oct 2026 by a second agent. Plain-words summary first: the conclusion of the report stands (do not move to Aerial Gym; keep the MJX twin for wind and the fast simulator for the dynamic obstacles). What changed is a handful of numbers and one or two overstated details. Every external source below was opened (arXiv, DOI/Crossref, GitHub API and raw files, NVIDIA and PX4 pages, PyPI); every number about our code was recomputed from the repo, read-only. Nothing was installed, nothing in the repo was changed, PX4 and Gazebo were not started. Scratch scripts are in `research/v/` (`bench.py`, `mjxbench.py`, `dim.py`).

### Tally

VERIFIED 31 (rows V1 to V31 below; each row groups several sub-claims, e.g. V9 covers nine issues), CORRECTED 14 (C1 to C14), REMOVED 0. Nothing was deleted: two items I could not confirm are kept but flagged (C13, the README MJX throughput, which I could not reproduce; U1, two project-note figures).

### CORRECTED (the text above was edited)

| # | What the first draft said | What is right | Source opened |
|---|---|---|---|
| C1 | Wind flyer has 49 inputs, about 80,000 parameters | `land_env.OBS_DIM` is 50 (`conditions/wind/README.md` still says 49, stale). Policy has 80,649 parameters (50-256-256-7 plus 2 log-std), counted by running `ppo.init_params` | `conditions/wind/land_env.py`, `conditions/wind/ppo.py`, run on CPU |
| C2 | Decider has "about 220 inputs" and about 130,000 parameters, always including the planner's forecast | 221 inputs only in the "eyes" variant (with the planner's 20 s forecast); 165 without. Policy parameters 126,222 with eyes, 111,886 without | `shared/fast_sim/urban_rl.py`, run (`v/dim.py`) |
| C3 | Fast simulator measured at about 98 lock-steps/s (about 0.4 M drone-decisions/s) at 4,096 drones | That number was taken with other jobs on the GPU. Re-run: 328 / 302 / 288 lock-steps/s at 1,024 / 4,096 / 32,768 on a quiet GPU, 91 to 280 when busy. A lock-step is 0.25 s, so these are drone-steps (not decisions): about 1.2 M/s at 4,096, about 9 M/s at 32,768. The "launch-bound, flat in batch size" reading is confirmed | `v/bench.py`, 3 runs |
| C4 | "Updated 2 Oct 2026" for the Aerial Gym repo | That is GitHub's activity stamp; the last code push was 28 Jun 2026. Stars (776) and open issues (22) are right | GitHub API, `ntnu-arl/aerial_gym_simulator` |
| C5 | x500 config disturbance block adds random force/torque "up to 0.1 N, 0.05 N m in the PX4 sim-to-real config" | In the shipped x500 config the block is disabled with zero limits; the 0.1 / 0.05 values sit in the unused `robot_asset` block; the PX4 sim-to-real task config does not enable it. Conclusion (no wind model) unchanged | `x500_config.py`, `base_multirotor.py`, `position_setpoint_task_sim2real_px4_config.py` on `main` |
| C6 | Blackwell needs PyTorch 2.7+ which needs Python 3.10+ | PyTorch 2.7.0 requires Python >=3.9. Still above Isaac Gym's 3.8 ceiling, so the clash and the conclusion stand | PyPI torch 2.7.0 metadata; PyTorch v2.7.0 release notes |
| C7 | MJX twin: "finished flights frozen with `jnp.where`" | Frozen only in `land_env.evaluate()`. In `ppo.py` training a finished flight is replaced at once by a fresh one (automatic reset) | `conditions/wind/ppo.py` lines 126-146, `conditions/wind/land_env.py` `evaluate` |
| C8 | `JaxToTorch` / `ArrayConversion` "exist in 1.3.0" (implied usable) | They exist, but importing them fails in the `mujoco_drones` env because `array_api_compat` is not installed | run in `mujoco_drones` env |
| C9 | Aerial Gym "no wind, no outdoor" flights; indoor motion capture for the 1.2 kg drone in the Aerial Gym column | The paper's velocity/acceleration and corridor policies flew an ArduPilot drone with onboard odometry; the Voxl 2 Mini motor policy used motion capture; the 1.2 kg drone is the separate PX4-module paper. Neither paper describes wind or outdoor tests (reworded as "does not describe", not "none") | arXiv 2503.01471 PDF text; arXiv 2505.00432 PDF text |
| C10 | Isaac Sim 5.1 "requires driver 580.65.06 or later"; Python >=3.12 attributed to the Isaac Sim 6 docs | The requirements page names 580.65.06 (and 595.58.03 on the latest page) as the tested driver; Python >=3.12,<3.13 comes from the Isaac Lab 3.0 beta notes, a different page | Isaac Sim 5.1.0 and latest requirements pages; Isaac Lab release notes |
| C11 | README gives the "under a minute / under an hour" training times on RTX 3090 or laptop 3080 Ti | README gives the two claims; the hardware is in the RL-training docs page (3090 for navigation and position control; a laptop 3080 Ti for the LiDAR task of the Unified Autonomy Stack) | Aerial Gym `README.md`, `docs/6_rl_training.md` |
| C12 | Table I of the Aerial Gym paper "mixes units for L2F"; not quoted | The L2F row is 14,063 x 10^6 samples/s at 65,536 instances (reported by the L2F authors, laptop GPU). Now quoted, with its caveat | arXiv 2503.01471 PDF, Table I |
| C13 | MJX twin about 21 M physics steps/s; PPO round about 0.3 s (README) | **Not reproduced.** A hover-only test (4,096 drones, vmapped `control_step`, 64 decisions) gave about 5.5 M steps/s at 9% GPU load, 0.48 s per rollout, about 4 times below the README. Kept in the report as an unconfirmed README figure with this note. Not the same unit of work as Aerial Gym's 4.4 M in any case | `v/mjxbench.py` (also an unrolled variant, slower at 0.9 M/s) |
| C14 | `rl_nav` is a policy "on the companion computer" sending acceleration commands "over ROS 2 / MAVROS" | The README confirms Sample Factory, a ROS 2 node, body-frame acceleration + yaw-rate output and a `mavros_msgs` dependency; "companion computer" is the report's reading, now labelled as such | `ntnu-arl/rl_nav` README |

### UNCONFIRMED (kept, flagged)

- U1. "Gazebo about 1.3 to 3.4 s of wall time per simulated second" and "never two SITLs at once": project notes only (Sai's memory files), no repo file to re-derive them from. Left as "(project notes)". (Wind harmless 10/10 flights is confirmed from `conditions/weather/weather_report.json`: calm, breeze and gusty each 10 of 10 landed.)

### VERIFIED

External sources (all opened):

| # | Item | Source opened |
|---|---|---|
| V1 | Aerial Gym paper: title, Kulkarni / Rehberg / Alexis, IEEE RA-L, 2025 (vol. 10 no. 4, pp. 4093-4100, April), arXiv 2503.01471, DOI 10.1109/LRA.2025.3548507 | arxiv.org/abs/2503.01471; Crossref via doi.org |
| V2 | Paper: built on Isaac Gym, Warp ray-casting, RL Games / Sample Factory / CleanRL, 4.43 M samples/s at 65,536 (RTX 3090, constant RPM, obstacle-free), 1,592 to 2,750 fps for 270 x 480 depth, 2.83 m/s maximum speed in the cluttered corridor, ArduPilot + ROVIO platform, Voxl 2 Mini motor policy in a custom PX4 module at 250 Hz | arXiv 2503.01471 PDF, text extracted |
| V3 | PX4 NN paper: title, Hegre / Rehberg / Kulkarni / Alexis, ICRA 2025 workshop, arXiv 2505.00432; 650 Hz, 93.4 us, 137.6 us, 1.2 kg quad, Pixracer Pro STM32H743 460 MHz 2 MB flash 1 MB RAM, 64 + 32 ReLU, 15 in / 4 out, motion capture, square, motor time constants blamed, rl_games PPO | arxiv.org/abs/2505.00432 and PDF |
| V4 | PX4 docs (main branch): `mc_nn_control` experimental, "PX4 v1.17" badge, Ubuntu 24.04+, three build targets, 15-input breakdown (3+6+3+3), 4 outputs, replaces the whole cascade and allocator, setpoints from `TrajectorySetpoint`, no offboard setpoints (comparison table), `xxd` + `control_net.cpp`, model check for 15 / 4, about 50 KB | github.com/PX4/PX4-Autopilot `docs/en/neural_networks/*` |
| V5 | RAPTOR: PX4 v1.18 badge, 2,084 parameters, RLtools, SD card / MAVLink FTP, 10+ real platforms 32 g to 2.4 kg; paper identity (Eschmann, Albani, Loianno, arXiv 2509.11481) | PX4 `raptor.md`; arxiv.org/abs/2509.11481 |
| V6 | PX4 v1.17.0 released 13 May 2026; v1.18.0-rc1 10 Sept 2026 | GitHub releases API |
| V7 | PX4 HITL for multicopters: Gazebo Classic, community maintained | PX4 `docs/en/simulation/hitl.md` |
| V8 | Aerial Gym repo: 776 stars, 22 open issues, BSD-3-Clause; Isaac Lab "under development" in README, "available" on docs home (Isaac Lab v2.3.2), "working on" in the FAQ | GitHub API; `README.md`; docs `index.md`, FAQ |
| V9 | Issues: #64 and #65 (opened 5 Sept 2026, reward bugs, unconfirmed), #47 wind (July 2025, open, maintainer's drag recipe), #49 (merged into PX4 main, Aug 2025), #2 and #22 (dynamic obstacles; Warp bounding-volume rebuild is slow), #13 (Isaac Gym unsupported), #1 and #35 (GPU questions) | GitHub API issue pages and comments |
| V10 | x500 config: thrust constant 8.54858e-6, motor time constants 0.0125 up and 0.025 down; drag/damping zero by default; Warp and Isaac Gym sensor notes; Lee geometric controllers (attitude, rate, position, velocity, acceleration); Ouster OS0-128-style LiDAR; "dynamic simulation slows by a lot" warning; dynamic-env bounds about 11-12 x 5-8 x 4-5 m | Aerial Gym repo files and docs |
| V11 | Aerial Gym install page: python=3.8, PyTorch 1.13.1, CUDA 11.7, "newest PyTorch your driver supports"; sim2real page: hover flight, thrust constant = (weight x 9.81 / 4) / rpm^2; NaN-motor warning | Aerial Gym `docs/2_getting_started.md`, `docs/9_sim2real.md` |
| V12 | Code API: `BaseTask(ABC)` with `reset`, `reset_idx`, `step`; PX4 task imports `gym.spaces`; rl_games runner `ExtractObsWrapper(gym.Wrapper)` | Aerial Gym `main` source files |
| V13 | NVIDIA forum 367941: RTX 5080, kernels only SM_50 to SM_80, no PTX, staff reply 27 Apr 2026 "no workaround", closed source, no update | forums.developer.nvidia.com thread |
| V14 | Isaac Gym page: "legacy software", "no longer supported", replacement Isaac Lab; download page: Python 3.6 to 3.8, Ubuntu 18.04 / 20.04, 8 GB minimum | developer.nvidia.com/isaac-gym and `/download` |
| V15 | Isaac Lab v2.3.2 (Isaac Sim 5.1): multirotor / thruster actuator / ARL drone task via PR #3760 (contributors include the Aerial Gym maintainer); the multirotor asset page lists no drag, wind or controllers | Isaac Lab release notes; v2.3.2 API page |
| V16 | Isaac Sim 5.1 requirements: 16 GB GPU memory with a warning below that; Ubuntu 22.04 / 24.04 | docs.isaacsim.omniverse.nvidia.com 5.1.0 requirements |
| V17 | Blackwell support arrives in PyTorch 2.7 (CUDA 12.8) | PyTorch v2.7.0 release notes |
| V18 | `unified_autonomy_stack` paper arXiv 2605.12735: "The Unified Autonomy Stack: Toward a Blueprint for Generalizable Robot Autonomy", Dharmadhikari et al., 12 May 2026 (the first draft had not opened it); `rl_nav` description | arxiv.org/abs/2605.12735; GitHub |
| V19 | unitree_rl_gym issue #96 (RTX 5090, opened by the checker) | GitHub API |
| V20 | Laptop facts: RTX 5070 Laptop compute capability 12.0, driver 580.178.04, Ubuntu 24.04.5, 8,151 MiB | `nvidia-smi`, `/etc/os-release` |

Our code and measurements (re-derived, read-only):

| # | Item | How |
|---|---|---|
| V21 | Fast simulator: `UrbanEnv.reset()` / `step(cmd)` return only an observation dictionary; no reward, spaces or per-drone reset in the class; `summary()`; own `torch.Generator` streams; synchronous generations | `shared/fast_sim/urban_env.py` |
| V22 | 14 candidates, hold of 2 decisions, 0.25 s lock-step so 0.5 s per decision; 300 s = 1,200 steps; reward computed in `urban_rl.generation()`; duplicate-candidate mask `repeats()`; velocity command (n, 3) m/s | `shared/fast_sim/urban_env.py`, `shared/fast_sim/urban_rl.py` |
| V23 | TAU_XY 0.69 s, TAU_Z 0.23 s (measured on PX4) | `shared/fast_sim/urban_env.py` header and constants |
| V24 | MJX twin: `reset(key, w10)`, `step(st, act, key, ...)` returning `(state, reward, done)`, vmapped over the drones, `lax.scan` over time; decision 0.04 s (10 physics steps of 4 ms), PX4 inner loops at 250 Hz; feed-forward limit 3 m/s^2; 2 continuous + 5-way vertical output (7 head outputs) | `conditions/wind/land_env.py`, `conditions/wind/x500_mjx.py`, `conditions/wind/ppo.py` |
| V25 | MJX constants: mass 2.5843 kg, KT 8.54858e-6, w = 150 + 850 u, lag 12.5 / 25 ms, rotor drag 8.06428e-5, CdA 0.04 / 0.05 (labelled an assumption), 30 m yard, house 10 x 8 m, noise 0.03 + 0.004 h m, 0.05 m bias | `conditions/wind/x500_mjx.py`, `conditions/wind/land_env.py` |
| V26 | README Gazebo-versus-MJX checks: tilt 44.1 / 45.6 deg at 8 m/s, hover tilt 22.48 / 22.6 deg in 15.8 m/s wind, step-response time differences 0.02 to 0.05 s | `conditions/wind/README.md` |
| V27 | 30 Sept reversal-price comment (about 1.5 Hz pulsing, thrust limit, sank in Gazebo) | `conditions/wind/land_env.py` lines 43-49 |
| V28 | Earlier single-drone environments are Gymnasium classes: `ApproachEnv(gym.Env)`; `OutdoorLandingEnv` and `CageLandingEnv` through `BaseAviary(gym.Env)` in `MuJoCo-drones-gym` | repo files plus `~/RL_projects/MuJoCo-drones-gym` |
| V29 | Installed Gymnasium 1.3.0, PettingZoo 1.27.0; `VectorEnv` with `num_envs`, single and batched spaces, `AutoresetMode` NEXT_STEP / SAME_STEP / DISABLED | run in `mujoco_drones` env |
| V30 | 32,768 drones need under 200 MiB of GPU tensors (peak 182 MiB; 202 MiB seen by the driver); 4,096 drones 77 MiB | `v/bench.py` |
| V31 | Gazebo weather: calm, breeze and gusty skies landed 10 of 10 | `conditions/weather/weather_report.json` |


### Three most important corrections

1. The measured speed of our fast simulator was wrong by about 3 times (98 vs about 300 lock-steps/s), because the first run shared the GPU. It also was called "decisions" when it is 0.25 s steps. And the MJX figure of 21 M steps/s could not be reproduced (5.5 M/s in a quick hover test). Any speed comparison to Aerial Gym's 4.4 M/s should wait for a proper rerun.
2. Observation sizes and parameter counts for our own networks were off: wind flyer 50 inputs not 49 (README is stale); decider 221 inputs only with the planner view (165 without), so "needs about 220 inputs including the forecast" is true only for that variant.
3. Aerial Gym's x500 "disturbance" is switched off and the quoted 0.1 N / 0.05 N m limits are not used; wind is simply absent. Also its code is not Gymnasium-based (old `gym`) despite the paper's wording, and the repo's last code push is 28 June 2026, not 2 October.
