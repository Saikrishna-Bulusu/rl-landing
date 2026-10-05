# Which learning algorithm for the wind learner, and why

Written for Sai first (plain words, one everyday example per idea), then the details for the paper.
Date: 4 Oct 2026. Independently checked the same day (see "Verification" at the end; the larger corrections are marked CORRECTED in the text, the full list is there). Repo read only; all scripts and numbers below were made under `papers/research/scripts/alg/`.

---

## 0. The short answer

1. **Keep PPO as the learner for the paper's decision layer** (the 1-of-5 vertical choice, every 0.04 s). No alternative has earned a switch. The main reason is not "PPO is best". It is that the usual reason to leave PPO (it wastes data) does not apply here, because our data is almost free and studying it is not (section 1).
2. **Today's check found a hole you must close before a reviewer finds it.** A 15-line non-learning controller (a "disturbance observer": it guesses how hard the wind pushes right now and pushes back) lands **80.6%** of flights at 18 m/s. Your learned PPO flyer lands **44 to 55%** (8 seeds). Same drone, same yards, same channel to PX4. So the learned flyer is not the best flyer. Section 2 has the table, section 4 says what to do.
3. **The off-policy family (SAC, TD3, FastTD3, PQL) and the world-model family (DreamerV3, TD-MPC2) are not recommended** as replacements. Reasons: they fit our two-kind action badly, our discount (0.9998) is far outside their comfort zone, and 25 decisions per second is a known trouble for them. FastTD3 (for continuous parts) and PQN (for the discrete part) are the right *challengers* to test.
4. **One experiment settles it** (section 5): about 2 GPU hours on a quiet laptop (about 7 to 9 if the GPU is shared like in the old logs), plus 2 to 3 days of coding.

### Words used in this file (each explained once, in everyday terms)

| Word | Plain meaning |
|---|---|
| policy | the rule the drone follows to pick its next move (here a small neural network) |
| on-policy | learn only from data collected by the *current* version of the policy, then throw it away |
| off-policy | keep old data in a diary (replay buffer) and keep learning from it, even though it came from older versions |
| actor / critic | actor = picks moves; critic = a second network that scores "how good is this situation", so the actor knows whether a move helped |
| discount (gamma) | how far ahead the learner cares. 0.99 at 25 decisions/s looks about 4 s ahead; our 0.9998 looks about 200 s ahead |
| hybrid action | one move made of two kinds of choice: here 2 smooth numbers (acceleration push) plus 1 pick from 5 (vertical mode) |
| update-to-data (UTD) | how many learning steps per new piece of data |
| feed-forward (FF) | a push added *before* the error shows up, instead of reacting after |
| disturbance observer (DOB) | a small calculator that estimates the unseen push (wind) from how the drone is actually accelerating |
| non-learning baseline | a hand-built controller with a few knobs and no training, used to check the learning earns its keep |

---

## 1. On-policy and off-policy, with a driving-school picture

**On-policy (PPO).** A learner driver and an instructor. After each lesson the instructor reviews *only today's drive*, tells the learner what to change, and the notes are then thrown away, because they describe a driver who no longer exists. Next lesson, new drive, new notes. Wasteful, but the notes always match the current driver, so the advice is reliable.

**Off-policy (SAC, TD3, FastTD3, PQL).** The learner keeps a diary of every drive ever made and re-studies old pages again and again. Far less waste. The risk: a page written by last week's driver may mislead today's driver, and studying many old pages while also changing your habits is how learners get confused (the "deadly triad" in the FastTD3 paper [4]).

**Why thousands of simulated drones change the maths.** With 4,096 learner drivers on the road at once, every second brings 4,096 fresh pages. Throwing yesterday's notes away costs nothing. Off-policy's big advantage (re-using pages) matters when pages are expensive: a real robot, a slow simulator, camera images. It matters much less when pages are cheap.

**Our numbers (measured today on the RTX 5070 Laptop, quiet GPU, `papers/research/scripts/alg/prof.py`, `td3bench.py`):**

| What | Time |
|---|---|
| Simulate 4,096 drones for one decision (includes running the policy) | about 3.7 to 4.3 ms (the checker measured 3.7) |
| One PPO round: 262,144 fresh decisions (4,096 drones x 64 steps): simulate 0.276 s + learn 0.241 s | 0.52 s (about 500k decisions/s) |
| One FastTD3-sized learning step (batch 32,768, distributional critic, bf16) | about 30 ms |
| The same in plain fp32 | about 63 ms (CORRECTED by the checker: 62 to 64 ms in four runs; the first version said 96 ms and did not reproduce) |
| A smaller step (batch 8,192, bf16 / fp32) | 6.6 ms / 13.6 ms |

Read the second block this way. **One learning step on 32,768 stored decisions (30 ms) costs about as much as simulating 28,000 brand-new decisions (7 simulation steps).** Re-studying a page costs roughly the same as writing a new one (with bf16; in fp32 it costs about twice as much, 63 ms). That is why FastTD3's headline (faster than PPO) was shown on humanoids in Isaac and MuJoCo Playground, where each simulated step is expensive [4], and why it is not automatic on our cheap, lightweight simulator.

Resulting throughput if one environment step of 4,096 drones is followed by 2 learning steps (my arithmetic from the numbers above, not a run of full FastTD3):

| Learner | Decisions per second | Note |
|---|---|---|
| PPO (measured) | 500k to 615k | 0.43 to 0.52 s per round; see the caveat below |
| FastTD3-size, batch 32,768, bf16 | about 64k | 4.3 + 2 x 30 ms per 4,096 decisions |
| FastTD3-size, batch 32,768, fp32 | about 31k | 4.3 + 2 x 63 ms |
| FastTD3-lite, batch 8,192, bf16 | about 230k | 4.3 + 2 x 6.6 ms |

Off-policy sees about 2.5 to 20 times fewer fresh decisions per second. It wins only if each decision teaches it more than 2 to 25 times as much. PPO already reaches the hardest wind level of its curriculum quickly: in the logged stage-1 run `fly_hp_s0` the wind first reaches W17 at round 261 in all eight flyer seeds (about 68M decisions, roughly 2 minutes of quiet-GPU time; all eight logs checked).

**Caveat on speed I could not resolve.** Your logged runs took 2,058 to 2,143 s for 1,000 stage-1 rounds in six of the eight seeds (2.1 s per round, 125k decisions/s, matching "130k at 4,096"; seed 5 took 1,800 s and seed 2's log stops at round 800 after 1,043 s) and 637 to 666 s for 600 stage-2 rounds. In isolation I measure 0.43 to 0.52 s per round. The likely causes are other jobs sharing the GPU and the validation evals (each 512-flight level takes about 7 s, so one eval every 100 rounds adds about 28 s), but I did not prove that. Plan the experiment on the quiet-GPU numbers and re-measure on the day.

---

## 2. What the code does today, and what the new check found

### Today's learner (read from `conditions/wind/ppo.py` and `conditions/wind/land_env.py`)

| Item | Value |
|---|---|
| Algorithm | PPO, plain JAX, 394 lines. Clip 0.2, 4 epochs x 8 minibatches, GAE lambda 0.95, entropy 0.003, LR 3e-4 annealed to a tenth (stage 2 of the paper's decider runs at 1e-4), KL early stop at 1.5 x 0.02 |
| Parallelism | 4,096 drones x 64 steps per round (262k decisions) |
| Network | tanh MLP 256-256, 50 inputs (CORRECTED from 49: `E.OBS_DIM` prints 50 and the checkpoint's first weight matrix is 50 x 256; the repo README also says 49), one head: 2 Gaussian means (state-independent log-std) + 5 logits. Separate value net 256-256 |
| Action | acceleration feed-forward added to the rule's velocity setpoint (|a| <= 3 m/s^2 per axis) + a vertical mode (rule's own, down 1.5, down 0.6, hold, up 1.0 m/s). It is a push on top of PX4's velocity controller, not a replacement |
| Start | the learner starts as the rule: the "rule's own" logit has bias 4.0 (p about 0.93), feed-forward about 0 |
| Reward | landed +10 to +15, crash -15 and hard -10 by default; the paper decider (`stage2_c100h100g`) uses crash -100 and hard -100 (CORRECTED: the first version said hard -30, which belongs to the older `stage2_c100` recipe), timeout -5, time -0.002/step, shaping (potential-based, at gamma 0.9998 in the paper decider), reversal price on climb/descend flips (exists, but 0 in the paper decider) |
| Curriculum | W10 drawn from [0, W] or the top 3 m/s; W starts 4 and rises 1 m/s when 95% of the top band lands (the flag `--w-ok 0.95` in every queue script; the code default is 0.8), or after 150 rounds stuck |
| Two stages | stage 1 flyer (`--no-decide`), stage 2 decider on the frozen flyer (`--flyer ... --no-fly`). All three stage-2 seeds sit on the same single frozen flyer, `fly_s0_best.npz`, not on the `fly_hp` family below |

Observed seed spread of PPO (1,024 paired flights per point, standard error about 1.5 points at 50%, 0.7 at 95%; ladder JSONs in `conditions/wind/runs/`):

| Learner | W15 | W18 | W20 |
|---|---|---|---|
| Rule (no learning) | 77.9% | 5.6% | 0.0% |
| PPO flyer only, 8 seeds (`fly_hp_s0..s7_best`, recipe with `--hp-tau 4`) | 98.6 to 99.4% | **43.8 to 54.9%** | 5.0 to 7.6% |
| The one flyer under the paper's decider (`fly_s0_best`, no hp-tau) | 99.3% | 45.9% | 4.4% |
| PPO flyer + PPO decider, 3 seeds (`stage2_c100h100g_s0..2_best`, the paper's decider; all on `fly_s0_best`) | 98.6 to 99.1% | 93.8 to 96.2% (crash 0.8 to 3.0%) | 77.4 to 85.8% (crash 3.4 to 11.1%) |
| An early joint run, both heads at once (`both_s0_best`, older recipe) | 88.3% | 40.3% | 8.1% |

So PPO reproduces well from seed to seed (stage 1 spread about 11 points at W18, stage 2 about 2.4 points; standard deviation across the three stage-2 seeds is about 1.4 points). Careful: the three stage-2 seeds share one frozen flyer, so their 2.4-point spread does not include flyer-to-flyer variation. That flyer (45.9% at W18) lies inside the 8-seed range of the other recipe, so it is a typical flyer, but the paper should say it is one flyer. The joint early run is evidence that the hybrid head was hard to train at once, which is the real reason the work is split in two stages. (Different, older recipe, so treat it as a hint, not a measurement of the hybrid head.)

### New today: a non-learning controller beats the learned flyer

Prototype `papers/research/scripts/alg/dob.py`, `dob2.py`, `dob3.py` (not in the repo). In plain words, every 0.04 s it does three things:

1. Look at how fast the drone's velocity changed over the last 4 readings (what the learner already sees in its history): the real acceleration.
2. Work out how much acceleration the propellers were giving, from the lean (tilt) of the drone: about g x tilt.
3. The difference is the wind's push. Add the opposite as the acceleration feed-forward: FF = -k x push, with k a single number and the same +-3 m/s^2 limit as the learner. The rule and PX4 do everything else.

Gain k was picked on the validation yards (seed 20000, W18: k 0.5 / 1.0 / 1.5 / 2.0 gave 78.1 / 81.1 / 64.8 / 23.3% landed; the checker's independent re-implementation gave 78.0 / 80.9 / 65.8 / 23.1%), and scored on the test yards (seed 10000, the same 1,024 paired flights as the PPO ladders):

| W10 (m/s) | 9 | 12 | 15 | 18 | 20 |
|---|---|---|---|---|---|
| DOB feed-forward + the rule's own vertical, landed | 100% | 100% | 100% | **80.6%** | 18.3% |
| (crashed / gave up at 90 s) | 0 / 0 | 0 / 0 | 0 / 0 | 0.3% / 19.1% | 0.5% / 81.2% |
| Rule | 100% | 100% | 77.9% | 5.6% | 0.0% |
| PPO flyer, 8 seeds | 100% | 99.9 to 100% | 98.6 to 99.4% | 43.8 to 54.9% | 5.0 to 7.6% |
| PPO flyer + PPO decider, 3 seeds | | 98.2 to 99.5% | 98.6 to 99.1% | 93.8 to 96.2% | 77.4 to 85.8% |

Checks I ran so this is not an accident (validation yards, W18):

| Check | Landed |
|---|---|
| Wrong sign (k = -0.5) | 0.0% (so the push estimate is real, not luck) |
| Clean | 81.3% |
| Attitude noise 0.6 degree | 81.6% |
| Attitude noise 1.7 degree | 80.1% |
| Attitude bias 0.6 degree | 81.2% |

What this does and does not show:

- It shows the flyer's job (cancel the gust's push) has a simple answer that a one-knob formula finds, and that PPO did not find a policy this good. Capacity is not the issue (the formula is a function of inputs the network sees). The cause is probably credit assignment: the reward for a good push arrives 30 to 90 s later, as a landing.
- It does **not** yet show that the decider on top of DOB beats 96% at W18 (untested), nor that DOB transfers to PX4 + Gazebo (the velocity estimate there has latency and noise that the MJX twin only approximates; never flown). The prototype used the true attitude with added noise, and a stateless slope over 4 readings with no filter tuning, so a tuned version has room.
- The learned decider does buy something DOB cannot: 19% of DOB's flights give up at W18 (they wait for the rule's stillness gate), and 81% at W20. Deciding when to commit is the discrete choice and nothing in the non-learning list can make it.
- It costs the decider a little at W12 (98.2 to 99.5% against 100% for rule and DOB) and its crash rate at W20 is 3.4 to 11.1% against 0.5% for DOB. Say both in the paper.

---

## 3. Option by option

### 3A. How each option handles our action, how fast it trains with thousands of GPU drones, how stable it is

| Option | Hybrid action (2 smooth + 1 of 5) | Wall-clock with thousands of GPU envs | Stability (from sources and our history) |
|---|---|---|---|
| **PPO** (today) | Native. Gaussian for the 2 smooth numbers and a categorical for the pick; log-probabilities add. Already in `ppo.py` | Measured 0.52 s per 262k decisions on a quiet GPU. Logged runs under sharing: 35 min (stage 1), 11 min (stage 2) | Needed two fixes (slide after the peak: LR annealing + KL early stop; hover leak: shaping at gamma). Seed spread small (see section 2) |
| **SAC** [2] | Smooth part native. Discrete part needs SAC-discrete [35] or hybrid SAC [34]; neither is a stock library piece | Same cost class as TD3 plus entropy handling (my estimate: 1.0 to 1.2 x FastTD3). On one Isaac Sim locomotion task (Unitree A1 velocity), a careful write-up found it slower, less consistent and slightly below PPO, even after fixing action bounds [31]. In PQL's six tasks, DDPG with n-step beat SAC with n-step in every one [5] | FastTD3's own SAC variant "FastSAC" trains faster than SAC but stays slower than FastTD3 and "tends to be unstable" [4]. Squashed (tanh) actions are sensitive to the action bounds [31] |
| **TD3** [3] | Smooth part native, deterministic. Discrete part needs a relaxation or P-DQN-style (a smooth parameter per discrete choice) [33] | Small-batch TD3 under-uses 4,096 drones | Three stabilisers (clipped double Q with two critics, delayed policy updates, target-policy smoothing; target networks too) [3] |
| **FastTD3** [4] | Same as TD3 | Authors: solves HumanoidBench tasks in under 3 h on an A100; faster than PPO on IsaacLab and MuJoCo Playground humanoids, "particularly on rough terrain with domain randomization". Recipe: batch 32,768, 2 to 8 updates per 128 to 4,096 env steps, 101-atom distributional critic, gamma 0.99, target mix 0.1, critic 1024-512-256 [4][4b]. Our measured cost per update: 30 ms (bf16) / 96 ms (fp32) | Authors report stable runs without layer-norm tricks [4]. They also found that a reward tuned for PPO gave an undeployable gait with FastTD3, so rewards must be retuned per algorithm [4] |
| **PQL** [5] | Same as TD3 (DDPG family) | Three processes in parallel (act, value learner, policy learner). The FastTD3 authors dropped the asynchronous part because of "high implementation complexity" [4]. In one JAX process on one laptop GPU it adds nothing over FastTD3 | Mixed exploration noise per env is the useful idea; FastTD3 found "no significant gains" from it but kept it [4] |
| **PQN** (discrete only) [6] | Discrete part only (our 5 modes). Smooth part would stay with the DOB or a separate learner | No replay buffer, no target network, layer-norm; reported up to 50 times faster than classic DQN [6] | Our extra danger: Q-learning is shown to "collapse with small time steps" (DQN and DDPG) [29], and we decide at 25 Hz |
| **DreamerV3** [7] | Handles discrete and continuous actions in general; mixing both per step in one head: not verified | A learned world model is an extra network training pass on top of the policy. FastTD3's comparison plots show DreamerV3 and TD-MPC2 as slow in wall-clock (they come from 48-hour HumanoidBench runs) [4] | Fixed hyperparameters across 150+ tasks is its selling point [7] |
| **TD-MPC2** [8] | Continuous control; discrete not mentioned in what I read | Same story as DreamerV3 [4] | One set of hyperparameters across 104 tasks [8] |

### 3B. Known quadrotor use, cost in our plain-JAX code, risk, verdict

| Option | Known quadrotor use (only sources I opened) | Cost in our code (my estimates) | Risk for THIS problem | Verdict |
|---|---|---|---|---|
| **PPO** | Zero-shot Crazyflie flight with a PPO framework [14]; PPO beat DQN agents for VTOL landing on an offshore docking station in simulated waves (numerical experiments only) [15]; the FastTD3 paper says practitioners mostly use PPO for deployable policies [4] (it separately cites champion-level drone racing as a sim-to-real success of RL, without naming the algorithm there; that Nature paper was not opened); RL beat optimal control in drone racing, trained within minutes on a workstation [12] | 0 (exists) | Low. Known failure modes are understood and fixed. Weakness: needs fresh data every round, which we have | **Keep** for the decision layer |
| **SAC** | SAC vs TD3 for low-level quadcopter control, with dynamic entropy tuning [38] | About 300 new lines, plus about 150 for a hybrid head. 1 to 2 days | Medium to high: hybrid head, action bounds, gamma | No |
| **TD3** | 18 s of training on a consumer laptop for a Crazyflie rotor-speed policy, flown in real [10]. TD3-based "WA-TD3" for strong-wind tracking, over 62% better tracking accuracy (abstract) [18] | About 250 lines | As FastTD3, minus the big-batch speed | Only as FastTD3 |
| **FastTD3** | None found for quadrotors (search was not exhaustive) | About 250 to 350 lines: GPU replay buffer, twin distributional critic, mixed noise, jitted collect-and-update. 1 to 1.5 days | Medium to high: (a) discount 0.9998 vs its 0.99 [4b]; (b) 25 Hz time-step trouble [29]; (c) replay vs wind curriculum; (d) reward retune [4]; (e) discrete part not covered | **Challenger** for any continuous (flyer) learning |
| **PQL** | None found | Not practical (asynchronous processes) | | No; FastTD3 is its single-process descendant [4] |
| **PQN** | None found for drones | About 120 to 150 lines; it can reuse the PPO rollout scan | Medium: time-step trouble [29], gamma, reward scale for a -100 crash | **Challenger** for the decider |
| **DreamerV3** | Vision-based drone racing to 9 m/s in sim and hardware-in-the-loop [13] | Thousands of lines or a second framework; no DreamerV3 or TD-MPC2 package is installed in the `mujoco_drones` environment (checked with pip list) | High. We already own the simulator at about 500k decisions/s, so learning a model of it buys almost nothing | No |
| **TD-MPC2** | Original TD-MPC for fixed-wing attitude under varying wind [37] (that is TD-MPC, not TD-MPC2, and not a quadrotor) | As DreamerV3 | High, same reason | No |

### 3C. Structure choices and the non-learning baselines reviewers will ask for

| Option | What it is, in our terms | Evidence | Cost in our code | Verdict |
|---|---|---|---|---|
| **Residual vs end-to-end** | We are already residual: the learner adds a push to the rule's velocity setpoint and may override the rule's vertical speed. End-to-end would output setpoints from scratch (or rotor commands) | Residual RL on top of a PID cascade cut position deviation by about 50% in simulation and by 35% on average (53% at the maximum) in an outdoor test with wind above 13 m/s [17]; residual policies fix imperfect controllers [16]; commanding body rates and thrust transfers better than commanding rotors [11] | Residual: done. End-to-end ablation: one flag and about 3 runs | **Keep residual.** Optional end-to-end ablation |
| **Imitation, then RL** | Copy a teacher first, then improve by RL | DAgger [20]; drone acrobatics learned by imitating a privileged MPC expert [19]; seeding an off-policy buffer with demonstrations [21]. We already do a version: the "rule" logit starts at p about 0.93 | Warm-start the flyer from the DOB by regression: about 30 lines | **Use as warm start only** (arm in section 5) |
| **INDI** | Incremental controller: uses the measured angular acceleration (from the gyro) to correct rotor commands directly | [23][24] (tracking to 12.9 m/s flight speed and 2.1 g with 6.6 cm RMS; tests with a drag plate and a rope pull, not wind) [24]; hierarchical DRL + INDI flown on a 50 g Crazyflie in fan wind up to 3.5 m/s: 94.6% vs 36.0% overall success against DRL + PID across 80 experiments [28] (CORRECTED from 94.7% vs 55.0%) | The real thing replaces PX4's rate loop: not reachable through our MAVLink velocity + acceleration channel (Gazebo test impossible). An acceleration-level incremental version is the DOB form below | Cite. Do not implement the firmware form |
| **L1 adaptive control** | A fast estimate of the unknown disturbance, passed through a low-pass filter so the correction stays smooth | L1Quad: tested for 11 kinds of uncertainty on agile quadrotors [25] | About 60 lines, same slot as the DOB | Optional second baseline |
| **Disturbance observer + feed-forward** | Section 2's controller | General theory [26]; estimators with PID or MPC position control against gusts up to 12 m/s [27] | About 15 lines (done) | **Mandatory baseline.** Prototype done |
| **Neural-Fly** | Learn offline a shared description of how aerodynamic force depends on the drone's state, then adapt a few coefficients online in real time | 12 minutes of flight data, wind tunnel up to 12.1 m/s, beats INDI, L1 and a nonlinear baseline [22] | 2 to 3 days (collect rollouts at many winds in MJX, learn the basis, add online least squares) | Later. Wins mainly if the wind force depends strongly on the drone's own velocity |

A note you should be ready to say aloud: **every published wind test I opened is at or below about 13 m/s** (Neural-Fly 12.1 [22], Hentzen 12 [27], residual RL above 13 outdoors [17], GustPilot 3.5 [28]). Our hard levels are W10 = 18 and 20 m/s, which is 23.6 and 26.2 m/s at 30 m (1.31 x W10). Near that, the drone leans so far it runs short of thrust. Published methods have no claim there, so "show me an INDI or L1 number at 18 m/s" can only be answered by our own baseline.

---

## 4. Recommendation, with reasons

**R1. Keep PPO as the learner of the paper's decision layer (the discrete 1-of-5).** Reasons, each checkable:

1. The decider is where the win is (5.6% to 96% at 18 m/s with the PPO flyer) and PPO handles a mixed or discrete action with a few lines already written and tested.
2. Our discount is 0.9998 (about 200 s ahead, because landing pay-off comes 30 to 90 s later). PPO gets this through advantage estimates (GAE). Off-policy methods bootstrap their estimates from the next step's estimate, and the instability of that grows as the discount nears 1 [30]. FastTD3's own default is 0.99, about 4 s at 25 Hz [4b]. Switching would force a change to the *problem*: slower decisions (action repeat) or n-step returns.
3. 25 decisions per second is a known trouble for Q-learning (DQN and DDPG collapse at small time steps [29]).
4. The reward took a week to get right (shaping leak, reversal price, crash price). The FastTD3 authors found rewards tuned for PPO misbehave under their algorithm [4]. A new learner re-opens that.
5. Cheap data, costly study (section 1). The speed argument for off-policy is weak here.
6. Reproducible: 8 flyer seeds within 11 points at W18 and 3 decider seeds within 2.4 points (the three decider seeds share one frozen flyer), all landed on the first reasonable recipe after the two fixes.

**R2. Fix the flyer question now.** The flyer is where a 15-line baseline beats a learned policy. Choose one of these, in this order of effort:

- (a) Run the experiment of section 5, arm P1: PPO decider on a DOB flyer. If it is at or above the paper's 94 to 96% at W18, drop the learned flyer from the main story. Simpler system, no stage-1 training, a flyer you can explain with one equation.
- (b) If the paper is already frozen, add the DOB row to the table and describe the PPO flyer's role honestly: the learned flyer helps over the rule, but a simple observer helps more at the same channel; the decider, not the flyer, is the contribution.
- (c) Residual on top of the DOB (PPO learns a small correction after being warm-started from it): only if (a) leaves a gap at W20.

**R3. Do not adopt SAC, PQL, DreamerV3 or TD-MPC2.** Adopt FastTD3 and PQN only as challengers in the experiment. Rank for a future real-robot fine-tuning stage (not in scope): FastTD3 first (off-policy reuse matters once samples are real and costly).

**What would change this recommendation:**

- The challenger beats PPO by 3 or more points at W18 and W20 (about 2 standard deviations of PPO's stage-2 spread) at equal wall-clock, with no more crashes: then switch the decider's learner.
- The project moves to real-flight fine-tuning, a slow simulator (Gazebo in the training loop) or camera images in the loop: then sample reuse is worth its cost and FastTD3 or a world model returns to the table.
- A reviewer's reading of the 25 Hz decision rate requires a slower decider anyway (e.g. 5 Hz): Q-learning's time-step trouble [29] shrinks and PQN becomes more competitive.

**How to say it to the professor, in four sentences.** We trained the decider with PPO because our simulator produces over 500 thousand decisions a second, which makes re-using old data unnecessary, and because our landing reward arrives 30 to 90 seconds later, which suits PPO's long-horizon estimates. We checked this against the strongest off-policy recipe for GPU simulation (FastTD3 and a discrete counterpart) at equal wall-clock on the same task and seeds, and against a non-learning disturbance observer on the same channel. The disturbance observer is a strong flyer, so the learned part of our system is the decision, which a controller cannot make. We report all three with seeds and paired flights.

---

## 5. THE experiment that settles it

**Question.** On the same task, with the learner as the only difference, does anything beat PPO at equal wall-clock for the decision layer? And does a non-learning flyer make the learned flyer unnecessary?

**Fixed for every arm.** `land_env.py` as it stands (same reward, curriculum from W4 to W20 stepping 95%, observations, 4,096 drones, 25 Hz), validation yards seed 20000, test yards seed 10000 with 1,024 paired flights per wind, W = 9, 12, 15, 18, 20, checkpoint chosen on validation exactly as `ppo.py` does (best mean validation over W9, 12, 15, 18, scored on 512 flights per level every 100 rounds; the final ladders use 1,024).

| Arm | Flyer | Vertical decider | Learner | Seeds | Status |
|---|---|---|---|---|---|
| R | rule | rule | none | n/a | done (README ladder) |
| N | DOB feed-forward | rule | none | n/a | prototype done: 80.6% at W18 |
| P0 | PPO flyer (`fly_hp_s*`) | PPO | PPO | 3 | done (paper's decider) |
| **P1** | **DOB feed-forward (frozen)** | **PPO** | **PPO** | 3 | to run |
| **Q1** | **DOB feed-forward (frozen)** | **challenger** | **PQN-style Q-learner** | 3 | to run |

**Main comparison: P1 against Q1.** Same flyer, same env, same reward, same curriculum, same wall-clock. This isolates the algorithm. Side comparisons: P1 against P0 (does a learned flyer earn its keep? you want P1 at least as good), P1 against N (does the decider earn its keep? yes if above about 80.6% at W18).

**The challenger, concretely.** A PQN-style discrete Q-learner [6] on the 5 vertical modes: parallel environments, no replay buffer, layer-norm, lambda-returns, the same rollout scan as PPO. Pilots (1 seed, 8-minute budget each) choose among: decision repeat r = 1 (25 Hz) or 5 (5 Hz), and gamma 0.99, 0.995, 0.9998. The pilots are the price of giving the challenger a fair chance; report them. If a pilot shows the Q-learner needs 5 Hz to work, say so: it supports R1 point 3.

**Equal wall-clock.** Budget = the median total seconds of the three P1 runs (training + validation evals) on the same quiet GPU that day. Q1 gets exactly that many seconds, validation evals on the same schedule (every 60 s of wall time for both). Report every curve twice: against wall seconds and against decisions consumed.

**What is measured, all with 3 seeds per arm.**

- Landed percent at W18 and W20 on the paired test flights, plus W9, W12, W15 as sanity.
- Crash percent and give-up percent.
- Wall-clock to first reach 90% landed at W18 on validation yards.
- Seed-to-seed range, shown as dots, not just a mean.

**Decision rule, to agree with your advisor before running:** switch the decider's learner only if Q1's median beats P1's median by at least 3 points at both W18 and W20 with crash percent not higher, or matches P1 in half the wall-clock. Otherwise PPO stays and Q1 goes in the appendix. Power note: PPO's stage-2 seed SD is about 1.4 points and 1,024 flights give a standard error of 0.7 points at 95%, so 3 seeds cannot separate differences below about 3 points. Say so in the paper.

**Tracked in ClearML, project `rl-landing/wind`.** Same `Task.init` and series names as `ppo.py`'s `clearml_log` (so old and new overlay): "validation landed / W9, W12, W15, W18", "outcomes (training flights)", "curriculum", "speed / decisions/s". One change: pass the **wall seconds** as the iteration number for every scalar (the `iteration` argument of `report_scalar`), so curves from different algorithms line up on one axis. Task names `cmp_{ppo|pqn|dob}_{flyer}_s{seed}`, tags `alg:ppo`, `alg:pqn`, `base:dob`, `stage:2`. Upload the best checkpoint as an artifact as `ppo.py` does.

**Non-learning baseline in MJX.** Arm N is the DOB (section 2). Prototype is `papers/research/scripts/alg/dob.py`. To go into the repo it needs: the gain tuned on validation, a small sweep (gain k, and a filter time constant on the 4-reading slope: 3 x 5 settings, 15 evals of about 7 s each, 2 minutes), and a decision whether the real PX4 channel gives a usable acceleration estimate (it needs flying in Gazebo for that, outside this experiment). Optional second baseline: an L1-style version with a low-pass filter (about 60 lines).

**Optional phase 2 (only if reviewers ask for a flyer comparison, or P1 leaves a gap at W20).** Residual flyer on top of the DOB, PPO vs FastTD3, same protocol, 3 seeds, equal wall-clock: PPO residual (warm-started by regressing onto the DOB output) against FastTD3 residual (batch 8,192 to 32,768, bf16, replay N x 4,096 with N about 500 transitions = 0.9 GB, gamma and action-repeat chosen by pilots).

### Expected hours on this laptop (RTX 5070 Laptop, 8 GB shared)

| Step | GPU hours (quiet) | My time |
|---|---|---|
| DOB as a frozen-flyer hook in a copy of `ppo.py` / `land_env.py`, smoke test, sweep of k and filter | 0.2 | 0.5 day |
| P1: 3 PPO seeds, about 8 to 11 min each (stage-2 length, 600 rounds) | 0.5 | |
| Q1 code (PQN-style, about 150 lines, reusing the PPO rollout) | | 1 to 1.5 days |
| Q1 pilots: up to 6 settings x 1 seed x 8 to 11 min | 0.8 | |
| Q1 finals: 3 seeds x 8 to 11 min | 0.5 | |
| Ladders (8 winds x 1,024 flights, about 1 min each) for 6 to 9 checkpoints | 0.15 | |
| **Total, main experiment** | **about 2.2** | **2 to 3 days** |
| Same if the GPU is shared the way it was in the logs (about 3.5 to 4 x slower) | about 7 to 9 | |
| Phase 2 (if run): FastTD3 code 1.5 to 2 days; 6 pilots + 6 finals at about 13 min | about 2.6 | 2 days |

Housekeeping before any run (from your standing rules): `nvidia-smi` shows more than 2 GB free; `nice -n 10`; `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the laptop is on AC power; run arms one after another (a second job on the GPU would make the equal-wall-clock claim false); never start PX4 or Gazebo for this; run a 2-minute smoke test that the agent is actually in the loop (the discrete mode histogram is not all "rule's own") before spending the hours.

---

## 6. What I did not or could not check

- Most papers were read at abstract level. Read in more depth: FastTD3 (full text, parsed from the PDF), PQL (searched, not read in full), Learning to Fly in Seconds (partial). Of the FastTD3 numbers, batch 32,768, the 1024-512-256 critic, 2 to 8 updates per 128 to 4,096 environment steps and the no-layer-norm stability remark are in the paper text; gamma 0.99, tau 0.1 and 101 atoms come from the authors' `hyperparams.py` on GitHub [4b], not from the paper text.
- Every code-size and time estimate marked "my estimate" is mine, not measured.
- The 30 and 96 ms update times come from a benchmark with random data and a stand-in for the distributional loss (same matrix sizes, `papers/research/scripts/alg/td3bench.py`). They measure cost, not learning.
- I could not reproduce your logged 130k decisions/s; I measure 500k to 615k in isolation (cause unproven).
- The DOB prototype was tested only in MJX, with the observation's lean (true attitude plus the noise tests above). It has not been flown in Gazebo, and one setting of its gain and a 4-reading slope are all that was tuned.
- DreamerV3's handling of one step with a discrete and a continuous head together was not verified. I found no quadrotor use of FastTD3, PQL or TD-MPC2, but my search was not exhaustive.
- The champion-level drone racing paper (Nature, 2023) was not opened (its Nature page redirected to a login); the claim that it used PPO is not repeated here.

---

## 7. Check yourself (answer in your own words before the meeting)

1. Why does re-using old data not save time on our laptop, even though it saves samples? (Hint: compare 30 ms of studying with 4.3 ms of simulating.)
2. Why can a disturbance observer fly better than the learned flyer, and why can it not replace the decider?
3. What single result would make you switch from PPO, and how many seeds do you need to see it?

---

## 8. Sources (all opened during this work; URL or DOI)

**Learners and parallel simulation**

- [1] Schulman et al. 2017, PPO. https://arxiv.org/abs/1707.06347
- [2] Haarnoja et al. 2018, Soft Actor-Critic algorithms and applications. https://arxiv.org/abs/1812.05905
- [3] Fujimoto, van Hoof, Meger 2018, TD3. https://arxiv.org/abs/1802.09477
- [4] Seo et al. 2025, FastTD3. https://arxiv.org/abs/2505.22642 (full text read)
- [4b] FastTD3 default hyperparameters, authors' repository. https://raw.githubusercontent.com/younggyoseo/FastTD3/main/fast_td3/hyperparams.py
- [5] Li et al. 2023, Parallel Q-Learning. https://arxiv.org/abs/2307.12983
- [6] Gallici et al. 2024, Simplifying deep temporal difference learning (PQN). https://arxiv.org/abs/2407.04811
- [7] Hafner et al. 2023, DreamerV3. https://arxiv.org/abs/2301.04104
- [8] Hansen, Su, Wang 2023, TD-MPC2. https://arxiv.org/abs/2310.16828
- [9] Rudin et al. 2021, Learning to walk in minutes. https://arxiv.org/abs/2109.11978
- [29] Tallec, Blier, Ollivier 2019, Making deep Q-learning robust to time discretization. https://arxiv.org/abs/1901.09732
- [30] François-Lavet, Fonteneau, Ernst 2015, How to discount deep RL. https://arxiv.org/abs/1512.02011
- [31] Raffin 2025, SAC on massively parallel simulators (blog). https://araffin.github.io/post/sac-massive-sim/
- [32] Neunert et al. 2019 (CoRL 2019; arXiv Jan 2020), Continuous-discrete RL for hybrid control in robotics. (Not cited in the text above; kept as background, as are [1], [9], [36] and [39].) https://arxiv.org/abs/2001.00449
- [33] Xiong et al. 2018, P-DQN. https://arxiv.org/abs/1810.06394
- [34] Delalleau et al. 2019, Discrete and continuous action representation (hybrid SAC). https://arxiv.org/abs/1912.11077
- [35] Christodoulou 2019, SAC for discrete action settings. https://arxiv.org/abs/1910.07207
- [39] Zakka et al. 2025, MuJoCo Playground. https://arxiv.org/abs/2502.08844

**Quadrotor learning**

- [10] Eschmann, Albani, Loianno, Learning to fly in seconds (arXiv 2023; published in IEEE RA-L, accepted April 2024). https://arxiv.org/abs/2311.13081 (TD3 confirmed in the full text)
- [11] Kaufmann, Bauersfeld, Scaramuzza 2022, Benchmark comparison of learned control policies for agile quadrotor flight. https://arxiv.org/abs/2202.10796
- [12] Song et al. 2023, Reaching the limit in autonomous racing. https://arxiv.org/abs/2310.10943
- [13] Romero et al., Dream to Fly (DreamerV3 for drones). https://arxiv.org/abs/2501.14377
- [14] Chen et al. 2024, What matters in learning a zero-shot sim-to-real RL policy for quadrotor control (PPO). https://arxiv.org/abs/2412.11764
- [15] Ali, Gupta, Hashim 2024, Deep RL for sim-to-real policy transfer of VTOL-UAVs offshore docking operations. https://arxiv.org/abs/2406.00887
- [16] Silver et al. 2018, Residual policy learning. https://arxiv.org/abs/1812.06298
- [17] Ishihara et al. 2023, Wind resistance of cascaded PID quadcopters with residual RL. https://arxiv.org/abs/2308.01648
- [18] Liu et al. (first author Huidong Liu), Disturbance-aware hybrid learning for robust and adaptive UAV flight in extreme winds (WA-TD3), IJCAI 2026. https://www.ijcai.org/proceedings/2026/844
- [19] Kaufmann et al. 2020, Deep drone acrobatics. https://arxiv.org/abs/2006.05768
- [20] Ross, Gordon, Bagnell 2011, DAgger. https://arxiv.org/abs/1011.0686
- [21] Ball et al. 2023, Efficient online RL with offline data (RLPD). https://arxiv.org/abs/2302.02948
- [36] Kumar et al. 2021, RMA. https://arxiv.org/abs/2107.04034
- [37] Olivares et al. 2024, Model-free vs model-based RL for fixed-wing attitude control under wind (TD-MPC). https://arxiv.org/abs/2409.17896
- [38] Mahran et al., Dynamic entropy tuning in RL low-level quadcopter control (SAC vs TD3); IEEE ICCTA 2024, arXiv posted Dec 2025. https://arxiv.org/abs/2512.18336

**Non-learning baselines**

- [22] O'Connell et al. 2022, Neural-Fly. https://arxiv.org/abs/2205.06908
- [23] Smeur, Chu, de Croon 2016, Adaptive incremental nonlinear dynamic inversion for attitude control of micro aerial vehicles. https://doi.org/10.2514/6.2016-1390
- [24] Tal, Karaman, Accurate tracking of aggressive quadrotor trajectories using INDI and differential flatness. https://arxiv.org/abs/1809.04048
- [25] Wu et al., L1Quad. https://arxiv.org/abs/2302.07208
- [26] Chen, Yang, Guo, Li 2016, Disturbance-observer-based control and related methods, IEEE Trans. Industrial Electronics 63(2). https://doi.org/10.1109/TIE.2015.2478397
- [27] Hentzen et al. 2019, Disturbance estimation and rejection for high-precision multirotor position control. https://arxiv.org/abs/1908.03166
- [28] Habel et al. 2026, GustPilot: a hierarchical DRL-INDI framework for wind-resilient quadrotor navigation. https://arxiv.org/abs/2603.19966

**Our own measurements (`papers/research/scripts/`)**

- `papers/research/scripts/alg/prof.py`, `prof2.py` (PPO round timing), `td3bench.py` (update cost), `evtime.py` (eval cost), `dob.py`, `dob2.py`, `dob3.py` (DOB prototype and checks). Run with `/home/sai/miniconda3/envs/mujoco_drones/bin/python`, `nice -n 10`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. DOB test result: `papers/research/scripts/alg/dob_test.json`.
- Logged PPO runs and ladders: `conditions/wind/runs/fly_hp_s*_best_ladder.json`, `stage2_c100h100g_s*_best_ladder.json`, `rule_ladder.json`, `*.jsonl` (wall times).

---

## Appendix. The DOB feed-forward in 8 lines (what `dob.py` does)

```
a_meas   = (v[-1] - v[0]) / (3 * 0.04)        # horizontal acceleration from the last 4 velocity readings
z_h      = lean of the last 2 readings         # horizontal part of the drone's up-axis
a_thrust = 9.81 * z_h / sqrt(1 - |z_h|^2)      # horizontal acceleration the propellers give
push     = a_meas - a_thrust                   # what the wind adds
a_ff     = clip(-k * push, -3, 3) m/s^2        # same limit as the learned flyer; k = 1.0 chosen on validation yards
```
It is sent through the existing acceleration feed-forward input (`act[:2] = a_ff / 3`), with `fly=True, decide=False`, so PX4's velocity controller and the rule's vertical speed stay in charge of everything else.

---

## Verification

Checked on 4 Oct 2026 by a second agent that did not write the report. Method: every source opened at its arXiv page (title, first author, date, venue, from the page metadata) and its full text for the specific claim; the two DOI items through the Crossref DOI registry; IJCAI through the proceedings page; repo numbers re-read or recomputed read-only; GPU numbers re-run (GPU was quiet, 261 MiB used before, `nice -n 10`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, bytecode writing off, nothing installed, no PX4 or Gazebo). Scratch files are in `research/chk2/`.

**Totals: 52 items VERIFIED, 12 CORRECTED, 2 REMOVED** (40 source entries: 35 verified, 5 corrected, 0 removed; 26 claims about our code and numbers: 17 verified, 7 corrected, 2 removed).

### A. Sources (what was opened, and the result)

| Item | Result | What I opened and what it says |
|---|---|---|
| [1] PPO | VERIFIED | arxiv.org/abs/1707.06347: Schulman et al., 2017. Not cited in the text. |
| [2] SAC | VERIFIED | arxiv.org/abs/1812.05905: Haarnoja et al., 2018, Soft Actor-Critic Algorithms and Applications. |
| [3] TD3 | VERIFIED (wording tightened) | arxiv.org/abs/1802.09477: Fujimoto, van Hoof, Meger, ICML 2018. Abstract: minimum of two critics, delayed policy updates, link to target networks (target-policy smoothing is the third trick in the body). |
| [4] FastTD3 | CORRECTED | arxiv.org/pdf/2505.22642 (full text parsed): Seo et al., 2025. Confirmed: under 3 h on one A100; faster than PPO in IsaacLab and MuJoCo Playground, "particularly on rough terrain with domain randomization"; batch 32,768; critic 1024-512-256; UTD "2, 4, 8 updates per 128 to 4096 environment steps"; deadly triad remark; no layer-norm needed; mixed noise "no significant gains" but kept; PQL async dropped for "high implementation complexity"; DreamerV3, SAC, TD-MPC2 curves from 48-hour HumanoidBench runs; PPO-tuned reward gives an undeployable FastTD3 gait (and the reverse); FastSAC faster than SAC, slower than FastTD3, "tends to be unstable". WRONG in the report: it said the paper cites champion-level drone racing among PPO users. The paper (intro) cites it only as a sim-to-real success of RL and names no algorithm there. Text fixed. |
| [4b] hyperparams.py | VERIFIED | raw.githubusercontent.com/younggyoseo/FastTD3/main/fast_td3/hyperparams.py: gamma 0.99, tau 0.1, batch_size 32768, num_updates 2, num_atoms 101, critic_hidden_dim 1024, actor_hidden_dim 512. |
| [5] PQL | VERIFIED | arxiv.org/pdf/2307.12983: Li et al., ICML 2023. Three processes (actor, V-learner, P-learner); six Isaac Gym tasks; "DDPG(n) outperforms SAC(n) in all tasks"; mixed exploration. |
| [6] PQN | VERIFIED | arxiv.org/pdf/2407.04811 abstract: Gallici et al., 2024; no replay buffer or target network, LayerNorm, "up to 50x faster than traditional DQN". |
| [7] DreamerV3 | VERIFIED | arxiv.org/abs/2301.04104 and PDF: Hafner et al., 2023; over 150 tasks, a single configuration; text covers discrete and continuous actions (one head mixing both: still unverified, as the report says). |
| [8] TD-MPC2 | VERIFIED | arxiv.org/abs/2310.16828: Hansen, Su, Wang, 2023 (ICLR 2024); 104 online RL tasks, single set of hyperparameters; continuous control. |
| [9] Rudin et al. | VERIFIED | arxiv.org/abs/2109.11978: Rudin et al., 2021, CoRL 2021. Not cited in the text. |
| [10] Learning to fly in seconds | VERIFIED | arxiv.org/pdf/2311.13081: Eschmann, Albani, Loianno; arXiv 2023, accepted RA-L April 2024; TD3; "after only 18 seconds of training on a consumer-grade laptop"; RPM control; microcontroller deployment. Venue added to the source line. |
| [11] Benchmark comparison | VERIFIED | arxiv.org/pdf/2202.10796: Kaufmann, Bauersfeld, Scaramuzza, ICRA 2022. Body-rate and thrust policies transfer more robustly than single-rotor thrust. |
| [12] Reaching the limit | VERIFIED | arxiv.org/pdf/2310.10943: Song et al., Science Robotics 2023. "within minutes of training on a standard workstation". |
| [13] Dream to Fly | VERIFIED | arxiv.org/pdf/2501.14377: Romero et al., 2025, ICRA 2026. DreamerV3, up to 9 m/s, simulation and a hardware-in-the-loop setup with rendered images. |
| [14] What matters | VERIFIED | arxiv.org/pdf/2412.11764: Chen et al., 2024, RA-L. SimpleFlight, PPO-based, zero-shot on Crazyflie. |
| [15] VTOL docking | CORRECTED | arxiv.org/pdf/2406.00887: Ali, Gupta, Hashim, 2024. It is landing on an offshore docking station in simulated waves, numerical experiments only, not a "moving ship deck"; PPO learns the better policy against the DQN variants. Title and text fixed. |
| [16] Residual policy learning | VERIFIED | arxiv.org/abs/1812.06298: Silver et al., 2018. |
| [17] Ishihara | CORRECTED | arxiv.org/pdf/2308.01648: Ishihara et al., 2023. Outdoors with wind above 13 m/s the mean deviation fell 35% (maximum 53%); the "about 50%" is the simulation figure and the abstract headline. The report said "halved ... outdoors". Fixed. |
| [18] WA-TD3 | VERIFIED | www.ijcai.org/proceedings/2026/844: Liu et al. (first author Huidong Liu), IJCAI 2026 (35th). Abstract: "over 62% improvement in tracking accuracy under strong winds". First author added. |
| [19] Deep drone acrobatics | VERIFIED | arxiv.org/pdf/2006.05768: Kaufmann et al., RSS 2020; demonstrations from an optimal controller with privileged information (MPC in the text). |
| [20] DAgger | VERIFIED | arxiv.org/abs/1011.0686: Ross, Gordon, Bagnell, AISTATS 2011 (arXiv 2010). |
| [21] RLPD | VERIFIED | arxiv.org/pdf/2302.02948: Ball et al., 2023; offline data (including a few demonstrations) used inside online RL. |
| [22] Neural-Fly | VERIFIED | arxiv.org/pdf/2205.06908: O'Connell et al., Science Robotics 2022. 12 minutes of flight data; wind up to 12.1 m/s; improvements of 66%, 42%, 35% over the nonlinear, L1 and INDI controllers. |
| [23] Smeur et al. | VERIFIED | DOI 10.2514/6.2016-1390 through Crossref: Smeur, Chu, de Croon, AIAA GNC Conference, 2016. (The publisher page returned 403; Crossref is the DOI registry.) |
| [24] Tal and Karaman | VERIFIED | arxiv.org/pdf/1809.04048: 12.9 m/s, 2.1 g, 6.6 cm RMS; drag plate and a tensioned wire/rope pull; no wind test. |
| [25] L1Quad | VERIFIED | arxiv.org/pdf/2302.07208: Wu et al.; "eleven types of uncertainties". |
| [26] Chen et al. DOB | VERIFIED | DOI 10.1109/TIE.2015.2478397 through Crossref: Chen, Yang, Guo, Li, IEEE Trans. Industrial Electronics 63(2), 1083-1095, 2016. |
| [27] Hentzen | VERIFIED | arxiv.org/pdf/1908.03166: Hentzen et al., IROS 2019; MPC and PID with an estimator, gusts up to 12 m/s. |
| [28] GustPilot | CORRECTED | arxiv.org/pdf/2603.19966: Habel et al., 2026. Abstract: 80 real-flight experiments, DRL-INDI 94.6% vs DRL-PID 36.0% overall success, wind up to 3.5 m/s, 50 g Crazyflie. The report's "94.7% vs 55.0%" is not in the paper. Fixed. |
| [29] Tallec et al. | VERIFIED | arxiv.org/pdf/1901.09732: 2019; "Deep Q-learning and DDPG collapse with small time steps" (abstract). |
| [30] Francois-Lavet et al. | VERIFIED (caveat) | arxiv.org/pdf/1512.02011: NIPS 2015 DRL workshop; DQN only; raising the discount toward its final value is linked to network instability. The report's reading "instability grows as the discount nears 1" is a fair paraphrase for DQN, not a general proof; keep the wording soft in the paper. |
| [31] Raffin blog | CORRECTED | araffin.github.io/post/sac-massive-sim/ (10 Feb 2025). Quote confirmed ("takes more time to train, is not consistent, and the performance is slightly below PPO's" after the quick fix), but it is one Isaac Sim locomotion task (Unitree A1), not "Isaac Lab" in general. Fixed. |
| [32] Neunert | VERIFIED | arxiv.org/abs/2001.00449: CoRL 2019 (arXiv Jan 2020). Not cited in the text. |
| [33] P-DQN | VERIFIED | arxiv.org/abs/1810.06394: Xiong et al., 2018; hybrid discrete-continuous action space. |
| [34] Hybrid SAC | VERIFIED | arxiv.org/abs/1912.11077: Delalleau et al., 2019; proposes Hybrid SAC. |
| [35] SAC-discrete | VERIFIED | arxiv.org/abs/1910.07207: Christodoulou, 2019. |
| [36] RMA | VERIFIED | arxiv.org/abs/2107.04034: Kumar et al., RSS 2021. Not cited in the text. |
| [37] Olivares | VERIFIED | arxiv.org/pdf/2409.17896: ICINCO 2024; the model-based agent is Temporal Difference MPC (TD-MPC), fixed-wing, simulation. |
| [38] Mahran | VERIFIED | arxiv.org/pdf/2512.18336: SAC (stochastic) against TD3 (deterministic) with dynamic entropy tuning; published IEEE ICCTA 2024, arXiv Dec 2025. Year added. |
| [39] MuJoCo Playground | VERIFIED | arxiv.org/abs/2502.08844: Zakka et al., 2025. Not cited by number in the text. |

Not confirmable, left as the report's own hedge: "no quadrotor use of FastTD3, PQL or TD-MPC2 found". The checker searched once for FastTD3 on quadrotors and found none (one 2026 quadcopter paper tests DDPG/TD3/PPO/SAC in Simulink, not FastTD3). The Nature drone-racing paper stays unopened (login redirect, not followed).

### B. Software and environment claims

The report makes no version or CUDA claims. One environment claim, "no DreamerV3 or TD-MPC2 package here", was replaced by a checked fact: `pip list` of `mujoco_drones` shows jax 0.10.2 (CUDA 12 plugin), mujoco and mujoco-mjx 3.13.0, torch 2.14.0, stable_baselines3 2.9.0, clearml 2.1.12, and no Dreamer or TD-MPC package. Old claim: REMOVED ("official code is outside our environment (not verified)").

### C. Claims about our code and numbers

| Item | Result | Evidence |
|---|---|---|
| PPO constants (clip 0.2, 4 epochs x 8 minibatches, lambda 0.95, entropy 0.003, LR 3e-4 to a tenth, KL stop 1.5 x 0.02, gamma 0.9998, 394 lines) | VERIFIED | `conditions/wind/ppo.py` read; stage 2 runs at LR 1e-4 (added) |
| Network (256-256 tanh, 2 means + 5 logits, state-independent log-std, separate value net, "rule's own" bias 4.0 so p = 0.93) | VERIFIED | `init_params`; 54.6 / (54.6 + 4) = 0.93 |
| Number of network inputs | CORRECTED | 50, not 49 (`E.OBS_DIM` run; weight shape 50 x 256). The repo README also says 49 and should be fixed. |
| Action limits and vertical modes | VERIFIED | `A_FF = 3.0`; `MODES = [rule, -1.5, -0.6, 0, +1.0]`; decision step 0.04 s |
| Reward constants | CORRECTED | Paper decider: crash -100 AND hard -100, shaping at 0.9998, no reversal price (checkpoint meta, `conditions/wind/queue/wind_flip2_30sep.sh`). The report said hard -30. |
| Curriculum 95% | VERIFIED (note added) | Code default is 0.8; every queue script passes `--w-ok 0.95`. |
| Stage structure | CORRECTED | The three stage-2 seeds all use one frozen flyer, `fly_s0_best.npz` (hp-tau 0), not the `fly_hp` family; its own ladder is 99.3 / 45.9 / 4.4% at W15 / 18 / 20. Added to the tables. |
| Ladders: rule 77.9 / 5.6 / 0.0; 8 flyer seeds; 3 decider seeds; joint run 88.3 / 40.3 / 8.1; spreads 11 and 2.4 points; SD 1.4 | VERIFIED | all `conditions/wind/runs/*_ladder.json` read (W12 of flyer seed 6 is 99.9, noted) |
| Standard errors 1.5 and 0.7 points | VERIFIED | sqrt(0.25/1024) = 1.56; sqrt(0.95 x 0.05/1024) = 0.68 |
| Validation seed 20000, test seed 10000, best checkpoint by mean over W9-18 | VERIFIED (detail added) | `ppo.py` lines 366-379; validation uses 512 flights per level |
| ClearML names and project | VERIFIED | `clearml_log`, `Task.init(project_name="rl-landing/wind")` |
| Log times | CORRECTED (minor) | 2,058 to 2,143 s in six of eight seeds (seed 5 1,800 s; seed 2 stops at round 800); stage 2 637 to 666 s; 125k decisions/s |
| W17 reached at "round 280, 73M decisions" | CORRECTED | round 261 in all eight seeds, about 68M decisions |
| DOB W18 test (80.6%) and W20 (18.3%), seed 10000 | VERIFIED | `dob_test.json` read; independent re-implementation written from the report's description (`papers/research/scripts/chk2/mydob.py`): 80.4% and 18.0%. W9, 12, 15: 100% each |
| DOB gain sweep 78.1 / 81.1 / 64.8 / 23.3 | VERIFIED | independent: 78.0 / 80.9 / 65.8 / 23.1 (within about 1 point; GPU run-to-run noise) |
| DOB sanity checks (wrong sign 0.0, clean 81.3, noise 0.6 deg 81.6, 1.7 deg 80.1, bias 81.2) | VERIFIED | re-ran `dob3.py`: 0.0, 81.2, 81.6, 80.4, 81.7 |
| Rule at W18 on test yards | VERIFIED | independent run 5.3% (report 5.6%, README 5.5%; three flights of difference) |
| PPO round timing 0.43 to 0.52 s, 500 to 615k decisions/s | VERIFIED | re-run `prof.py`: rollout 0.236 + update 0.203 = 0.44 s, 596k decisions/s |
| Update cost bf16 30 ms, batch 8,192 6.6 / 13.6 ms | VERIFIED | re-run: 30.1, 6.7, 13.5 ms |
| Update cost fp32 at batch 32,768: 96 ms | CORRECTED | 62 to 64 ms in four runs |
| Throughput table, "2 to 25 times fewer decisions" | CORRECTED | fp32 row about 31k/s (was 21k); ratio about 2.5 to 20. bf16 (64k) and lite (230k) rows re-derived and kept |
| "30 ms is about 28,000 new decisions" | VERIFIED | 30 / 4.3 = 7 steps x 4,096 = 28.6k |
| Discount horizons 4 s and 200 s; 1.31 x W10 = 23.6 and 26.2 m/s | VERIFIED | arithmetic |
| Experiment hours table (2.2 h; 7 to 9 h shared; phase 2 2.6 h) and replay 0.9 GB | VERIFIED | arithmetic re-done (replay about 0.84 GB at 102 floats per transition) |
| "FastTD3 paper cites drone racing among PPO users" | REMOVED | see [4] |
| "official code is outside our environment" | REMOVED | see B |

### D. What the checker could not settle

- Why the report's fp32 timing was 96 ms and ours 63 ms (probably a shared GPU at the time).
- Whether the DOB survives the real PX4 velocity estimate (never flown; the report says so).
- Run-to-run GPU noise: reruns differ by up to about 1 point at the steep part of the gain curve (k = 1.5).

