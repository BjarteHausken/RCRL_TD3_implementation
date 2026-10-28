# RCRL-TD3

Reinforcement learning for a double-integrator agent that navigates a pillar-filled corridor from a random start to a random goal, reaching the goal as fast as possible while avoiding collisions with walls and pillars. The repository contains:

- a **TD3 baseline** that maximizes a progress-based reward, and
- an **RCRL (Reachability Constrained Reinforcement Learning) agent** that learns the same reward while explicitly learning a reachability safety value, so collision avoidance comes from the learned constraint rather than from reward shaping alone.

The RCRL implementation is an adaptation, not a reproduction, of the paper's experiments: the paper's Algorithm 1 is a deterministic actor-critic template, which the paper evaluates in both on-policy and off-policy variants. This repository uses TD3 as its off-policy base. The RCRL safety objective, statewise Lagrangian, and dual update are preserved.

## Repository structure

```text
agents/
  td3.py                     Plain TD3 baseline (hyperparameters matched to RCRL-TD3)
  rcrl_td3.py                TD3 + RCRL safety critic, statewise or fixed-scalar multiplier
environment/
  __init__.py                create_env factory (standard corridor parameters)
  dynamics.py                Exact double-integrator discretization (dt = 0.1 s)
  environment.py             Gymnasium corridor environment (+ batched step_batch)
  layouts/generate_layouts.py  Random pillar-layout generator
  manual_control.py          Manual keyboard driving for environment sanity checks
layouts/                     Layout sets: <set>_{train,test,validation}.npy
models/                      Training runs as <name>/ (checkpoints + config.json)
utils.py                     Replay buffers (plain, and RCRL with h/h')
train.sh                     Wrapper: ./train.sh <rcrl|rcrl-vec|td3|td3-vec> [trainer flags]
train_scripts/
  train_rcrl_td3.py          Trainer for the RCRL agent
  train_rcrl_td3_vec.py      Vectorized RCRL trainer (parallel envs, device buffer)
  train_td3.py               Trainer for the plain TD3 baseline
  train_td3_vec.py           Vectorized TD3 trainer (parallel envs, device buffer)
evaluate.py                  Headless evaluation over all test layouts
evaluate_visual.py           Rendered single-episode evaluation
evaluate_gif.py              Headless single-episode gif rendering
evaluate_value.py            Interactive reachability-value explorer (RCRL-TD3)
evaluation.py                Batched evaluation used by trainers and evaluate.py
test.py                      Environment checks (gymnasium check_env, margins)
```

## Requirements

Python with NumPy, PyTorch, Gymnasium, and Matplotlib. There is no checked-in lockfile. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy torch gymnasium matplotlib
```

All commands below run from the repository root.

## Environment

A fully enclosed corridor of length 30 m with cylindrical pillars (radius 0.3–0.7 m); each layout has its own width (4–8 m) and pillar count (3–12). The agent starts at a random position and must reach a randomly placed goal.

| | |
| --- | --- |
| Dynamics | double integrator, `dt = 0.1 s`, speed clamped at 3.0 m/s |
| Action | 2D normalized acceleration command in `[-1, 1]`, scaled by `max_acceleration = 2.0` m/s² in the environment |
| Observation | velocity (2) + 64 lidar rays of 5.0 m range + goal-relative coordinates (2) → 68-dim |
| Termination | wall/pillar collision, or goal reached (distance to goal `<= goal_radius = 0.5 m`) |
| Truncation | 1000-step time limit (bootstrapped; configurable via `max_steps`) |
| Reward | `2.5 * (d_old - d_new) - 0.1` per step (`d` = distance to the goal), +50 on reaching the goal |

Actions are emitted normalized to `[-1, 1]` (the actor's `tanh` output scaled by `max_action = 1.0`) and multiplied by `max_acceleration` inside `step`/`step_batch`, so the true acceleration limit is exactly `±2.0 m/s²`. The corridor is closed at both ends; driving into either end wall terminates the episode like any other collision. Checkpoints trained before these environment changes (open goal end, double-scaled actions) are not compatible and must be retrained.

The state constraint is `h(s) = vehicle_radius - min(lidar_distances)`, where `vehicle_radius = 0.2 m`; see [Constraint convention](#constraint-convention).

The goal is a point inside the corridor (see [Randomized episodes](#randomized-episodes)); the agent perceives it only through the two relative goal coordinates `(goal_x - x, goal_y - y)`, which are the last two components of the observation.

**Observation normalization (on by default).** Each observation component is divided by its known range: velocities by the 3.0 m/s speed limit, lidar distances by the 5.0 m range, and the goal-relative coordinates by the loaded layout's corridor dimensions (30 m length and its own 4–8 m width). Every component then lands in roughly `[-1, 1]` or `[0, 1]`. The scaling is fixed — no learned or running statistics — so training, evaluation, and manual control always see the same input, and the observation space bounds reflect it. Disable it with `--no-normalize-obs` in any trainer or evaluation script; the choice is recorded in `config.json`, and a checkpoint must be evaluated with the same setting it was trained with.

## Layouts

Pillar layouts live in `layouts/` as `<set>_{train,test,validation}.npy` (the default `one` set has 30 train, 10 test, and 10 validation layouts). Each layout stores its own corridor `width` and its `(x, y, radius)` pillars; new sets draw a width from 4–8 m, 3–12 pillars per layout, and a radius from 0.3–0.7 m per pillar. Generate a new set with:

```bash
python -m environment.layouts.generate_layouts --name myset
```

Use `--min-pillars`/`--max-pillars`, `--min-width`/`--max-width`, and `--min-radius`/`--max-radius` to change the ranges. When a layout is loaded, `CorridorEnvironment` adopts its width, and the randomized start and goal are sampled on that same geometry.

All training and evaluation scripts select the set with `--layout-set` (default `one`); the set name is recorded in each run's `config.json`, so a checkpoint should be evaluated with the same `--layout-set` it was trained on.

## Run training

Run from the repository root as a module:

```bash
python -m train_scripts.train_rcrl_td3 --name my_rcrl_run
```

The default is `--multiplier scalar`, a single fixed Lagrange multiplier shared by all states and set with `--initial-lambda` (never updated during training). To use the paper's statewise `lambda(s)` network instead, pass `--multiplier network`:

```bash
python -m train_scripts.train_rcrl_td3 --name my_rcrl_network --multiplier network
```

The plain TD3 baseline (same randomized start+goal episode regime, same
actor/critic learning rates, and the same gradient clipping, but no safety
critic or multiplier) has its own trainer:

```bash
python -m train_scripts.train_td3 --name my_td3_run
```

### Vectorized training

`train_rcrl_td3_vec.py` and `train_td3_vec.py` run the same algorithms with
the same update-to-data ratio, but with `--num-envs` parallel environments
(default 32): one batched actor forward pass per vectorized step, a
device-resident replay buffer, and one gradient update per collected
transition, run back-to-back. Checkpoints and `config.json` have the same
format as the sequential trainers (with `"trainer": "vectorized"` added), so
evaluation is unchanged:

```bash
python -m train_scripts.train_rcrl_td3_vec --name my_rcrl_vec --num-envs 32
python -m train_scripts.train_td3_vec --name my_td3_vec --num-envs 32
```

All four trainers are also available through the wrapper script, which
selects the module by agent name and passes the flags through unchanged:

```bash
./train.sh rcrl --name my_rcrl_run
./train.sh rcrl-vec --name my_rcrl_vec --num-envs 32
./train.sh td3 --name my_td3_run
./train.sh td3-vec --name my_td3_vec --num-envs 32
```

For a small end-to-end check:

```bash
python -m train_scripts.train_rcrl_td3 --name rcrl_smoke --max-timesteps 8 \
  --batch-size 2 --replay-size 32 --eval-freq 0 --log-freq 1 \
  --output-dir /tmp/rcrl_runs
```

Checkpoints are written under `models/<name>/`. The run's `config.json` records the agent type in its `agent` field (`rcrl_td3` or `td3`) alongside the checkpoint prefix.

## Evaluation

Evaluate a checkpoint on all test layouts:

```bash
python evaluate.py --name my_rcrl_run
python evaluate_visual.py --name my_rcrl_run
```

`evaluate.py` reports per-layout outcomes and the same aggregates as the in-training evaluation: mean reward, mean steps, success rate, and collision rate. Each test layout is run for `--eval-episodes-per-layout` episodes (default 20), each with a freshly sampled feasible start and goal — the same randomized regime as training. Episodes are seeded by index, so evaluation runs are reproducible, and rates are reported with 95% Wilson confidence intervals.

Evaluation is batched: all episodes are stepped in parallel through `environment.step_batch`, with one actor forward pass over the stacked observations per timestep instead of one pass per episode. Finished episodes leave the batch immediately, so the loop stops as soon as every episode is done. The sampled starts, goals, and stopping rules match the sequential evaluation. Because the actor runs in a batch, floating-point accumulation differs slightly from single-row forwards, so individual episode rewards can shift marginally while the episode outcomes and step counts remain the same.

The evaluation scripts read the `agent` field in the run's `config.json` to select the agent type (TD3 or RCRL-TD3), so `--agent` is not needed; a run whose `config.json` does not record a valid agent type cannot be evaluated. For RCRL-TD3 the scripts also reconstruct the exact agent (`--multiplier`, `--initial-lambda`, and `--max-lambda`), so a checkpoint loads against the same network structure and multiplier parameterization it was saved with. The evaluation sampling margin and speed bound are also read from the run's `config.json`.

`evaluate_visual.py` renders a single episode and additionally accepts `--test-layout` (`random` or a zero-based index) to choose the test layout.

`evaluate_gif.py` renders one episode headlessly and writes only a gif, with no window and no evaluation report:

```bash
python evaluate_gif.py --name my_rcrl_run --test-layout 3
```

The default output is `gifs/<name>_<layout-set>_layout<INDEX>.gif`; use `--output PATH` to choose another path and `--gif-max-width` to bound the frame size. The episode is seeded by layout index, so the same model and layout always reproduce the same start, goal, and animation.

### Reachability-value explorer

`evaluate_value.py` plots the learned RCRL reachability value for a policy on the corridor, at a chosen velocity:

```text
V_h(s) = Q_h(s, pi(s))
```

where `Q_h` is the online safety critic and `pi` is the deterministic actor. The map uses a diverging scale centered on zero: negative values (predicted safe) run blue, positive values (predicted unsafe) run red, zero is neutral gray, and cells whose vehicle footprint overlaps a wall or pillar are left blank. The observation at every map cell is built exactly as `CorridorEnvironment._get_observation` builds it, including the lidar cast at that cell and the observation normalization recorded in the checkpoint.

```bash
python evaluate_value.py --name my_rcrl_run
python evaluate_value.py --name my_rcrl_run --layout-index 3 --speed 1.5 --heading 45
python evaluate_value.py --name my_rcrl_run --save value.png
```

The speed and heading sliders choose the velocity slice: every position on the map is evaluated at that same velocity. The goal sliders move the goal, and the layout slider switches pillar sets (`--layout-type train|test|val`, default `test`). Left-clicking the map pins a state marker, whose value, policy action, and raw constraint `h(s)` are shown in the upper-left corner; hovering reads the value under the cursor. The color scale rescales to each slice unless `--vmax` fixes a symmetric range. `--save` renders one headless figure instead of opening a window.

The script requires a checkpoint whose `config.json` records `"agent": "rcrl_td3"`; plain TD3 runs have no safety critic. A blue or red cell is a prediction of the learned critic, not a runtime safety guarantee (see [Important limits](#important-limits)).

## Manual control and tests

```bash
python environment/manual_control.py   # drive manually (needs a display)
python test.py                                   # check_env, random-state margins, goal termination/reward
```

`test.py` is an exploratory script (not a pytest suite). Re-run it when changing environment behavior.

## Constraint convention

The environment defines the state constraint as:

```text
h(s) = vehicle_radius - min(lidar_distances)
```

Therefore `h(s) <= 0` is safe and `h(s) > 0` means the sensed vehicle footprint overlaps a wall or pillar. The replay buffer stores both `h(s)` before the action and `h(s')` after it. The current-state value is essential: reachability is about the worst violation from the current state onward, not only the next collision signal.

## How TD3 is adjusted to RCRL

### 1. Learn a reachability safety value

Alongside TD3's twin reward critics, `SafetyCritic` learns the action-conditioned value `Q_h(s, a)`. It estimates the discounted worst constraint value that will occur when action `a` is taken and then the policy is followed. Its sign has a direct interpretation:

- `Q_h(s, a) <= 0`: predicted persistently safe trajectory.
- `Q_h(s, a) > 0`: predicted future constraint violation.

The target in `agents/rcrl_td3.py` is:

```text
Q_h_target = (1 - gamma) h(s)
             + gamma max(h(s), Q_h_target(s', pi_target(s')))
```

This is the discounted RCRL self-consistency update. In contrast to a cost-return critic, `max` retains the worst safety violation instead of summing costs. The safety target uses the unperturbed target actor; TD3's target-action noise is used only for the reward critic. On collision or goal terminals, bootstrapping is replaced by the recorded terminal `h(s')`; time-limit truncations still bootstrap.

### 2. Optimize reward subject to a statewise safety constraint

The actor minimizes the surrogate RCRL Lagrangian:

```text
L_actor = mean(-Q(s, pi(s)) + lambda(s) Q_h(s, pi(s)))
```

A predicted unsafe value is positive, so it increases the actor loss. A ReLU or sign reversal is intentionally not used: both would change the constraint direction. During this update critic weights are frozen, while gradients still flow from their outputs through the action into the actor.

### 3. Learn a state-dependent multiplier

`MultiplierNetwork` produces `lambda(s)`, rather than one global penalty. It uses Softplus to remain nonnegative and clips its output to `--max-lambda`. The multiplier is updated by gradient ascent on `lambda(s) Q_h(s, pi(s))`, implemented as minimization of its negative with the actor and safety critic detached. Thus it grows at states predicted to be unsafe and stays bounded for infeasible states.

`--multiplier scalar` replaces the network with a single fixed Lagrange multiplier shared by all states. It is set by `--initial-lambda` and never updated during training (so `--multiplier-lr` and `--multiplier-freq` have no effect in this mode), and the value is recorded in `config.json` like any other hyperparameter.

### 4. Separate update time scales

The reward critic and safety critic update every gradient step. The actor updates every `--policy-freq` steps (default 2), and the statewise multiplier every `--multiplier-freq` steps (default 6, network mode only). Every gradient step clips the global gradient norm of its network to `--grad-clip` (default `1.0`; `0` disables clipping). The default learning rates also make critic/safety updates faster than actor updates, and multiplier updates slowest:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--critic-lr` | `1e-4` | TD3 reward-critic learning rate |
| `--safety-lr` | `1e-4` | reachability safety-critic learning rate |
| `--actor-lr` | `2e-5` | policy learning rate |
| `--grad-clip` | `1.0` | max global gradient norm for all networks (`0` disables) |
| `--multiplier-lr` | `1e-6` | statewise dual-ascent learning rate |
| `--policy-freq` | `2` | actor/target update interval |
| `--multiplier-freq` | `6` | multiplier update interval |
| `--max-lambda` | `100` | upper multiplier bound |
| `--initial-lambda` | `10.0` | initial multiplier output; the fixed value in `--multiplier scalar` mode |
| `--multiplier` | `scalar` | `scalar` for one fixed multiplier, `network` for the statewise `lambda(s)` network |

### 5. Preserve stable TD3 reward learning

The reward critic remains TD3: twin Q-functions, clipped target-policy noise, delayed actor updates, and Polyak target averaging:

```text
target = tau * online + (1 - tau) * target
```

The actor, reward critic, safety critic, and each optimizer are saved in every RCRL checkpoint; in `--multiplier network` mode the multiplier network and its optimizer are saved as well (the fixed scalar needs no checkpoint file, it lives in `config.json`).

## Randomized episodes

RCRL imposes its constraint on the states in the support of the initial-state distribution (Prop. 4.1 of the paper), and the dual update only learns `lambda(s)` for states that appear in the replay buffer. Every episode therefore starts from a randomly sampled state, so the policy and the safety value are trained over the feasible set as a whole instead of from a single fixed start.

One `env.reset(...)` call does all of the following, in this order:

1. select a **layout** uniformly from the training layouts,
2. sample a **start state** via `CorridorEnvironment.sample_random_state` on that layout — a position with `vehicle_radius + margin` clearance from the walls and `pillar_radius + vehicle_radius + margin` clearance from every pillar, and a velocity drawn uniformly from a disk of radius `--random-reset-max-speed`, and
3. sample a **goal** the same way (with zero velocity), resampled until it is at least `2 * goal_radius` away from the start.

Sampling after the layout is selected guarantees the start and goal are feasible for the layout the episode actually runs on.

Full RCRL training runs from step 0 — there is no separate exploration or warm-up phase; exploration comes from the per-step action noise (`--expl-noise`).

| Option | Default | Purpose |
| --- | ---: | --- |
| `--random-reset-margin` | `0.3` | extra clearance (m) beyond the collision boundary for sampled start and goal states |
| `--random-reset-max-speed` | `1.0` | maximum speed (m/s) of the sampled start velocity |

Evaluation follows the same regime: one sampled start and one sampled goal per test layout, seeded by layout index for reproducibility.

## Important limits

RCRL's formal guarantees assume a deterministic Markov decision process, exact multi-time-scale updates, bounded/projection-compatible parameterizations, and an initial-state distribution that covers the largest feasible set (the randomized episode starts in [Randomized episodes](#randomized-episodes) widen this distribution). This corridor experiment uses neural networks, finite replay training, and observations made from velocity plus lidar rather than the full physical state. Treat success and collision rates as empirical evidence, not a formal persistent-safety guarantee.

The actor has no runtime safety filter. It returns its learned action directly, so safety depends on the learned safety critic and the data encountered during training.

## Reference

Yu, D., Ma, H., Li, S. E., & Chen, J. (2022). *Reachability Constrained Reinforcement Learning*. ICML 2022. https://proceedings.mlr.press/v162/yu22d.html
