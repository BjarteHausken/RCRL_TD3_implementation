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
models/                      Training runs as <name>/ (checkpoints + config.json + training_log.jsonl)
utils/                       Shared support modules
  replay_buffer.py           Replay buffers (plain, and RCRL with h/h')
  training_log.py            Structured JSONL training metrics written by every trainer
  plot_training.py           Training-curve plots from the JSONL logs
  test.py                    Environment checks (gymnasium check_env, margins)
train.sh                     Wrapper: ./train.sh <rcrl|rcrl-vec|td3|td3-vec> [trainer flags]
train_scripts/
  train_rcrl_td3.py          Trainer for the RCRL agent
  train_rcrl_td3_vec.py      Vectorized RCRL trainer (parallel envs, device buffer)
  train_td3.py               Trainer for the plain TD3 baseline
  train_td3_vec.py           Vectorized TD3 trainer (parallel envs, device buffer)
evaluation_scripts/
  evaluate.py                Headless evaluation over all test layouts
  evaluate_compare.py        Side-by-side comparison of several runs
  evaluate_visual.py         Rendered single-episode evaluation
  evaluate_gif.py            Headless single-episode gif rendering
  evaluate_value.py          Interactive reachability-value explorer (RCRL-TD3)
  evaluate_feasibility.py    Learned feasible set vs actual largest feasible set
  feasibility.py             Ground-truth viability kernel and set metrics
  evaluation.py              Batched evaluator used by the trainers and scripts
```

## Requirements

Python with NumPy, PyTorch, Gymnasium, Matplotlib, and Pillow. There is no checked-in lockfile, but `requirements.txt` lists the dependencies; Matplotlib is imported by `environment.py` even for headless runs. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Layout data is not tracked. Before training, evaluating, or running `utils/test.py`, generate the default layout set:

```bash
python -m environment.layouts.generate_layouts --name one
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
| Reward | `(2.5 * (d_old - d_new) - 0.1) / reward_scale` per step (`d` = distance to the goal), `+50 / reward_scale` on reaching the goal; `reward_scale` is `1.0` by default |

Actions are emitted normalized to `[-1, 1]` (the actor's `tanh` output scaled by `max_action = 1.0`) and multiplied by `max_acceleration` inside `step`/`step_batch`, so the true acceleration limit is exactly `±2.0 m/s²`. The corridor is closed at both ends; driving into either end wall terminates the episode like any other collision. Checkpoints trained before these environment changes (open goal end, double-scaled actions) are not compatible and must be retrained.

The state constraint is `h(s) = vehicle_radius - min(lidar_distances)`, with `vehicle_radius = 0.2 m`; see [Constraint convention](#constraint-convention). There is no separate safety margin in training, so `h(s) > 0` coincides with the collision boundary. Evaluation can optionally apply a wider runtime clearance with `--safety-margin` (at least the vehicle radius): the margin metrics use that threshold, and the actor's lidar is shortened by the excess over the vehicle radius so the policy acts as if obstacles were that much closer. The environment, collision checks, rewards, and the true lidar used for metrics are unchanged.

Rewards are divided by a fixed `reward_scale` (`--reward-scale`, default `1.0` in every trainer and in the environment) so that episodic returns and safety values have comparable magnitudes in the RCRL actor loss; larger divisors shrink returns, and the multiplier bounds must be scaled with the divisor (see [Reward scaling](#reward-scaling)). `reward_scale` is recorded in `config.json`, and evaluation uses the recorded value unless overridden.

The goal is a point inside the corridor (see [Randomized episodes](#randomized-episodes)); the agent perceives it only through the two relative goal coordinates `(goal_x - x, goal_y - y)`, which are the last two components of the observation.

**Observation normalization (on by default).** Each observation component is divided by its known range: velocities by the 3.0 m/s speed limit, lidar distances by the 5.0 m range, and the goal-relative coordinates by the loaded layout's corridor dimensions (30 m length and its own 4–8 m width). Every component then lands in roughly `[-1, 1]` or `[0, 1]`. The scaling is fixed — no learned or running statistics — so training, evaluation, and manual control always see the same input, and the observation space bounds reflect it. Disable it with `--no-normalize-obs` in any trainer or evaluation script; the choice is recorded in `config.json`, and a checkpoint must be evaluated with the same setting it was trained with.

## Layouts

Pillar layouts are generated rather than tracked: they live in the gitignored `layouts/` directory as `<set>_{train,test,validation}.npy`. The default `one` set has 30 train, 10 test, and 10 validation layouts; create it before running any script:

```bash
python -m environment.layouts.generate_layouts --name one
```

Generation uses the current randomized generator, so a freshly generated set is not byte-identical to the layouts used for earlier training runs. Generate a differently named set with `--name myset`. Each layout stores its own corridor `width` and its `(x, y, radius)` pillars; new sets draw a width from 4–8 m, 3–12 pillars per layout, and a radius from 0.3–0.7 m per pillar.

Use `--min-pillars`/`--max-pillars`, `--min-width`/`--max-width`, and `--min-radius`/`--max-radius` to change the ranges. When a layout is loaded, `CorridorEnvironment` adopts its width, and the randomized start and goal are sampled on that same geometry.

All training and evaluation scripts select the set with `--layout-set` (default `one`); the set name is recorded in each run's `config.json`, so a checkpoint should be evaluated with the same `--layout-set` it was trained on.

## Run training

Run from the repository root as a module:

```bash
python -m train_scripts.train_rcrl_td3 --name my_rcrl_run
```

The default is `--multiplier network`, the paper's statewise `lambda(s)` trained by dual ascent. To use a single fixed Lagrange multiplier shared by all states instead, pass `--multiplier scalar`; it is set by `--initial-lambda` and never updated during training:

```bash
python -m train_scripts.train_rcrl_td3 --name my_rcrl_scalar --multiplier scalar
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

## Training metrics and curves

Every trainer also writes `models/<name>/training_log.jsonl`, one JSON object per line. stdout output is unchanged. Each record has a `kind` and a `step` (environment transitions):

| `kind` | Written by | Contents |
| --- | --- | --- |
| `run` | all trainers, once | agent, trainer, layout set, seed, reward scale, step budget |
| `episode` | sequential trainers, per episode | reward, length, collision, goal; RCRL also margin steps and `max_h` |
| `progress` | all trainers, every `--log-freq` | replay size and RCRL multiplier/safety diagnostics; vectorized trainers also add per-window reward, length, collision, goal, and margin aggregates |
| `eval` | all trainers, every `--eval-freq` | validation reward, steps, success and collision with 95% CIs, margin metrics, and RCRL lambda statistics |

`--log-freq 0` disables only `progress` records; `episode` and `eval` records are still written. A rerun with the same `--name` overwrites the log along with the checkpoints. Runs trained before structured logging was added have no JSONL file and must be retrained to be plotted.

Plot one or more runs (the default output is `results/training_curves.png`):

```bash
python utils/plot_training.py my_rcrl_run
python utils/plot_training.py my_rcrl_run my_td3_run --smooth 50 \
  --save results/compare_curves.png
```

The figure has eight panels: training reward and episode length, validation success and collision with confidence bands, mean/max multiplier, multiplier at cap/floor, mean predicted safety value, and evaluation mean steps. Sequential runs use their per-episode series; vectorized runs use the episode-window means already recorded in `progress`. `--smooth` applies a trailing mean only to per-episode reward and length (default off; `0` or `1` disables it). Panels that do not apply to a run, such as multiplier diagnostics for plain TD3, are marked "not available". Margin metrics are recorded in the log but are not part of the default figure.

## Evaluation

Evaluate a checkpoint on all test layouts:

```bash
python evaluation_scripts/evaluate.py --name my_rcrl_run
python evaluation_scripts/evaluate.py --name my_rcrl_run --save-json results/my_rcrl_run.json
python evaluation_scripts/evaluate_visual.py --name my_rcrl_run
```

`--save-json PATH` writes the full report — run metadata, aggregate metrics, per-layout metrics, and the per-episode arrays — as JSON. Per-episode arrays use the same seeded start/goal pairs for every checkpoint of a layout set, so reports can be joined by layout and episode index for paired comparisons across runs. The file also embeds the run's `config.json` for provenance.

`evaluation_scripts/evaluate.py` reports per-layout outcomes and the same aggregates as the in-training evaluation: mean reward, mean steps, success rate, collision rate, and margin metrics. Each test layout is run for `--eval-episodes-per-layout` episodes (default 20), each with a freshly sampled feasible start and goal — the same randomized regime as training. Episodes are seeded by index, so evaluation runs are reproducible, and rates are reported with 95% Wilson confidence intervals.

The margin metrics measure constraint satisfaction separately from collision, using the run's `reward_scale` from `config.json` (override with `--reward-scale`) and an evaluation-only clearance threshold. That threshold is set with `--safety-margin` and defaults to the vehicle radius, i.e. the constraint used in training:

- **margin violation rate** — fraction of episodes in which `h(s) > 0` at any step, i.e. the agent came within the threshold clearance at least once;
- **steps in margin** — fraction of all steps spent with `h(s) > 0`;
- **mean clearance** — mean over steps of `min(lidar)`; higher is farther from the boundary;
- **mean min clearance** — mean over episodes of each episode's minimum clearance, the closest approach to the boundary.

When the threshold exceeds the vehicle radius, evaluation also implements it at runtime by shortening the actor's lidar observations by `safety_margin - vehicle_radius` (clamped at zero and scaled with the observation normalization). The policy therefore reacts to obstacles as if they were that much closer, while the simulation, collision detection, termination, rewards, and all reported metrics still use the true lidar. Passing no `--safety-margin` (or the vehicle radius) leaves the observations untouched.

Per-layout output also shows how many episodes had a margin violation and the mean minimum clearance.

Evaluation is batched: all episodes are stepped in parallel through `environment.step_batch`, with one actor forward pass over the stacked observations per timestep instead of one pass per episode. Finished episodes leave the batch immediately, so the loop stops as soon as every episode is done. The sampled starts, goals, and stopping rules match the sequential evaluation. Because the actor runs in a batch, floating-point accumulation differs slightly from single-row forwards, so individual episode rewards can shift marginally while the episode outcomes and step counts remain the same.

The evaluation scripts read the `agent` field in the run's `config.json` to select the agent type (TD3 or RCRL-TD3), so `--agent` is not needed; a run whose `config.json` does not record a valid agent type cannot be evaluated. For RCRL-TD3 the scripts also reconstruct the exact agent (`--multiplier`, `--initial-lambda`, `--max-lambda`, and `--min-lambda`), so a checkpoint loads against the same network structure and multiplier parameterisation it was saved with. The evaluation sampling margin, speed bound, and minimum start-goal distance are also read from the run's `config.json` (the distance can be overridden with `--min-start-goal-distance`).

`evaluation_scripts/evaluate_visual.py` renders a single episode and additionally accepts `--test-layout` (`random` or a zero-based index) to choose the test layout. Both it and `evaluation_scripts/evaluate_gif.py` accept `--safety-margin`, which applies the same runtime observation shift as `evaluation_scripts/evaluate.py` so the rendered episode shows the margin-aware behaviour.

`evaluation_scripts/evaluate_gif.py` renders one episode headlessly and writes only a gif, with no window and no evaluation report:

```bash
python evaluation_scripts/evaluate_gif.py --name my_rcrl_run --test-layout 3
```

The default output is `gifs/<name>_<layout-set>_layout<INDEX>.gif`; use `--output PATH` to choose another path and `--gif-max-width` to bound the frame size. The episode is seeded by layout index, so the same model and layout always reproduce the same start, goal, and animation.

### Reachability-value explorer

`evaluation_scripts/evaluate_value.py` plots the learned RCRL reachability value for a policy on the corridor, at a chosen velocity:

```text
V_h(s) = Q_h(s, pi(s))
```

where `Q_h` is the online safety critic and `pi` is the deterministic actor. The map uses a diverging scale centered on zero: negative values (predicted safe) run blue, positive values (predicted unsafe) run red, zero is neutral gray, and cells whose vehicle footprint overlaps a wall or pillar are left blank. The observation at every map cell is built exactly as `CorridorEnvironment._get_observation` builds it, including the lidar cast at that cell and the observation normalization recorded in the checkpoint.

```bash
python evaluation_scripts/evaluate_value.py --name my_rcrl_run
python evaluation_scripts/evaluate_value.py --name my_rcrl_run --layout-index 3 --speed 1.5 --heading 45
python evaluation_scripts/evaluate_value.py --name my_rcrl_run --save value.png
```

The speed and heading sliders choose the velocity slice: every position on the map is evaluated at that same velocity. The goal sliders move the goal, and the layout slider switches pillar sets (`--layout-type train|test|val`, default `test`). Left-clicking the map pins a state marker, whose value, policy action, and raw constraint `h(s)` are shown in the upper-left corner; hovering reads the value under the cursor. The color scale rescales to each slice unless `--vmax` fixes a symmetric range. `--save` renders one headless figure instead of opening a window.

The script requires a checkpoint whose `config.json` records `"agent": "rcrl_td3"`; plain TD3 runs have no safety critic. A blue or red cell is a prediction of the learned critic, not a runtime safety guarantee (see [Important limits](#important-limits)).

### Learned vs actual largest feasible set

`evaluation_scripts/evaluate_feasibility.py` compares the learned feasible set with the actual largest feasible set on the same positions and velocity slice:

```bash
python evaluation_scripts/evaluate_feasibility.py --name my_rcrl_run
python evaluation_scripts/evaluate_feasibility.py --name my_rcrl_run \
    --layout-indices 0 1 --slices 0,0 1,0 0,1 --goals-per-layout 8 \
    --save-json results/feasibility.json --save-png results/feasibility.png
```

- **Learned feasible set:** `{s : V_h(s) <= 0}`, the safety value used by the policy (same quantity as the [reachability-value explorer](#reachability-value-explorer)).
- **Actual largest feasible set:** the viability kernel — the largest set of states from which *some* control sequence keeps the vehicle collision-free forever, computed from the environment's exact double-integrator dynamics and collision geometry, with no policy involved. Because the zero action is admissible on a static map, it includes every state that can brake to a safe standstill; it also keeps states that stay safe by moving, so stopping is not assumed.

The kernel is the fixed point of `K_0 = collision-free states`, `K_{k+1} = K_k ∩ {s : ∃u, f(s,u) ∈ K_k}` on an `(x, y, vx, vy)` grid (`evaluation_scripts/feasibility.py`). The default velocity spacing is one full-braking step, `dt * max_acceleration` (0.2 m/s); a coarser grid rounds deceleration back to the same node and falsely declares high-speed states infeasible. Positions where the vehicle footprint overlaps an obstacle are excluded from both sets.

Because observations are goal-relative, `V_h` depends on the goal even though feasibility does not, so the learned maps and metrics are averaged over `--goals-per-layout` goals sampled from the training distribution. The reported metrics, restricted to collision-free grid positions, are:

- `IoU` — intersection over union of the two feasible sets;
- `false-safe` — share of actually infeasible states the learned value calls safe (the dangerous error);
- `false-unsafe` — share of actually feasible states it calls unsafe (the over-cautious error);
- `true feasible` / `learned feasible` — share of free positions each set claims.

With no `--save-png`, only the table prints. The figure has one row per velocity slice and three panels: learned set, viability kernel, and disagreement (learned-feasible-but-infeasible in red, learned-infeasible-but-feasible in blue). The map drawn is the first sampled goal; the row metrics are averaged over all sampled goals. `--position-spacing` (default 0.2 m), `--velocity-spacing`, `--action-levels` (default 3, i.e. `{-a, 0, +a}` per axis), and `--max-iterations` control solver accuracy; `--layout-type`, `--layout-indices`, and `--seed` select layouts and goals. A layout with a 148×32×31×31 grid converges in about 15 iterations and takes roughly 30 s on CPU. Only RCRL-TD3 checkpoints have a safety value.

Note the asymmetry: the kernel is the maximal set for *some* controller, while `V_h` is conditioned on the trained policy. The safety critic can only learn from states the policy visits, so an untrained or over-cautious policy shows mostly `false-unsafe` error, and a state outside the kernel is only truly unsolvable if no control sequence can save it — the policy may still fail inside the kernel.

## Manual control and tests

```bash
python -m environment.manual_control   # drive manually (needs a display)
python utils/test.py                              # check_env, random-state margins, goal termination/reward
```

`utils/test.py` is an exploratory script (not a pytest suite). Re-run it when changing environment behavior.

## Constraint convention

The environment defines the state constraint as:

```text
h(s) = vehicle_radius - min(lidar_distances)
```

Therefore `h(s) <= 0` is safe and `h(s) > 0` means the vehicle footprint overlaps a wall or pillar — the same condition that ends the episode in collision. Training always uses this boundary: there is no separate safety margin in the environment or the trainers.

Evaluation can optionally impose a wider runtime clearance threshold with `--safety-margin`, measured from the vehicle centre to the obstacle surface (the same quantity as `min(lidar_distances)`) and therefore **including** the vehicle radius. It must lie between the vehicle radius and the 5 m lidar range; the default is the vehicle radius itself, so without the flag nothing changes.

The threshold does two things at evaluation time. First, the margin metrics count a violation when `safety_margin - min(lidar) > 0`. Second, the actor's lidar observations are shortened by `safety_margin - vehicle_radius`, which is observation-space obstacle inflation: the policy perceives obstacles as closer and reacts accordingly, but the environment, collision checks, termination, rewards, and the metrics continue to use the true lidar. Choosing `safety_margin = m` therefore asks the policy to keep `m` metres from the surface, starting from the contact boundary `vehicle_radius` it was trained on. The threshold is an evaluation-time choice and is not stored in `config.json`.

The replay buffer stores both `h(s)` before the action and `h(s')` after it. The current-state value is essential: reachability is about the worst violation from the current state onward, not only the next collision signal.

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

`MultiplierNetwork` produces `lambda(s)`, rather than one global penalty. It outputs `min_lambda + (max_lambda - min_lambda) * sigmoid(...)`, so the bounds are built into the parameterisation instead of being applied afterwards. A plain clamp freezes the dual at the cap because it zeroes the gradient of every saturated state; a straight-through clamp keeps the gradient but lets the pre-clamp value drift arbitrarily far above the cap, which the dual cannot undo within a run. Sigmoid keeps the value inside `[min_lambda, max_lambda]` and keeps a usable gradient at both ends. The multiplier is updated by gradient ascent on `lambda(s) Q_h(s, pi(s))`, implemented as minimization of its negative with the actor and safety critic detached. Thus it grows at states predicted to be unsafe and the training log reports `mean_lambda`, `max_lambda`, `lambda_at_cap`, `lambda_at_floor`, and `mean_h_pred` so saturation is visible immediately.

`--max-lambda` must be scaled with the reward divisor. With the default `--reward-scale 1.0`, the paper's unscaled bound of 100 is the default. At `--reward-scale 50`, use `2.0` instead: keeping 100 there would let the dual apply a penalty 5000× the reward scale, and the actor would avoid the goal to maximize clearance.

`--min-lambda` (default `0.0`) is the floor of that range. The pure dual (`0.0`) can drive `lambda(s)` to zero: the dual only sees states that reach the replay buffer, those are overwhelmingly safe once the actor starts avoiding obstacles (a few percent of steps have `h(s) > 0`), and one shared network cannot preserve `lambda` on the rare risky states while it is pushed down by the rest. The actor then lowers its predicted `h` and the constraint switches off — evaluated policies showed `lambda` below `1e-3` and collision rates above 60%. A floor such as `0.5` keeps a minimum safety pressure everywhere while the network still adapts upward on risky states. The floor does not prevent `lambda` from being pinned at `--max-lambda` on the early collision-heavy phase, so the two bounds should be chosen with the reward scale in mind; a run whose log shows `lambda_at_floor` near 100% has no statewise adaptation left and is behaving like `--multiplier scalar`.

`--multiplier scalar` replaces the network with a single fixed Lagrange multiplier shared by all states. It is set by `--initial-lambda` and never updated during training (so `--multiplier-lr` and `--multiplier-freq` have no effect in this mode), and the value is recorded in `config.json` like any other hyperparameter. This is a fixed penalty, not the paper's dual update; it is opt-in for comparison runs.

### 4. Separate update time scales

The reward critic and safety critic update every gradient step. The actor updates every `--policy-freq` steps (default 2), and the statewise multiplier every `--multiplier-freq` steps (default 6, network mode only). Every gradient step clips the global gradient norm of its network to `--grad-clip` (default `1.0`; `0` disables clipping). The default learning rates also make critic/safety updates faster than actor updates, and multiplier updates slowest:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--critic-lr` | `1e-4` | TD3 reward-critic learning rate |
| `--safety-lr` | `1e-4` | reachability safety-critic learning rate |
| `--actor-lr` | `2e-5` | policy learning rate |
| `--grad-clip` | `1.0` | max global gradient norm for all networks (`0` disables) |
| `--multiplier-lr` | `1e-5` | statewise dual-ascent learning rate |
| `--policy-freq` | `2` | actor/target update interval |
| `--multiplier-freq` | `6` | multiplier update interval |
| `--max-lambda` | `100.0` | upper multiplier bound (paper's unscaled bound at `--reward-scale 1.0`; use `2.0` at scale 50) |
| `--min-lambda` | `0.0` | lower multiplier bound; positive values keep the constraint from switching off (`0.0` reproduces the pure dual) |
| `--initial-lambda` | `10.0` | initial multiplier output (network mode); the fixed value in `--multiplier scalar` mode |
| `--multiplier` | `network` | `network` for the statewise `lambda(s)` network, `scalar` for one fixed multiplier |

### 5. Preserve stable TD3 reward learning

The reward critic remains TD3: twin Q-functions, clipped target-policy noise, delayed actor updates, and Polyak target averaging:

```text
target = tau * online + (1 - tau) * target
```

The actor, reward critic, safety critic, and each optimizer are saved in every RCRL checkpoint; in `--multiplier network` mode the multiplier network and its optimizer are saved as well (the fixed scalar needs no checkpoint file, it lives in `config.json`).

## Reward scaling

The RCRL actor loss adds a reward term and a safety term:

```text
mean(-Q(s, pi(s)) + lambda(s) Q_h(s, pi(s)))
```

The two must be on comparable scales, otherwise one term dominates for reasons that have nothing to do with the constraint. Raw corridor rewards are large relative to the constraint value: a successful episode returns roughly 50–100, while `Q_h` is bounded by the constraint range (a few metres at most). Evaluated at the same scale, a multiplier near 0.1 carries no weight against a `Q` near 100, so the dual can never make the constraint bite.

The environment therefore divides every reward by a fixed `reward_scale`, recorded in `config.json`:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--reward-scale` | `1.0` | fixed divisor applied to every step reward and the goal bonus |
| `--safety-margin` | vehicle radius (`0.2`) | evaluation-only runtime clearance threshold in metres; also shortens the actor's lidar by the excess over the vehicle radius; see [Constraint convention](#constraint-convention) |

This is deliberately not a running mean/std normalization. A fixed divisor keeps the MDP stationary — the same state-action pair always yields the same reward — which TD3's targets and the safety critic's fixed point assume. Running statistics would make the reward non-stationary and leak evaluation-time information into the scale.

Reward scale and multiplier must be tuned together. Dividing rewards by 50 divides `Q` by 50, so a multiplier of `lambda = 0.1` produces the same safety pressure as `lambda = 5` did at unscaled rewards (multiplying the whole actor loss by 50 gives `-Q + 50 * lambda * Q_h`). The defaults (`--reward-scale 1.0`, `--initial-lambda 10`, `--max-lambda 100`) start the dual at the paper's unscaled pressure and cap it at the paper's unscaled bound of 100; at `--reward-scale 50`, the equivalent settings are `0.1` and `2.0`. The dual learning rate is `1e-5`. The multiplier network's hidden activations amplify these logit steps, so `lambda(s)` can still reach the cap during the early collision-heavy phase and then fall back once the safety critic stops flagging the visited states. Watch `mean_lambda`, `lambda_at_cap`, `lambda_at_floor`, and `mean_h_pred` in the training log: a sustained cap fraction means the bound is acting as a fixed penalty, while `lambda` close to the floor means the constraint has effectively switched off (unless the floor is set high on purpose). Evaluation falls back to `reward_scale = 1.0` for checkpoints trained before the field existed, so old models still evaluate with their original reward, and `--reward-scale` overrides the recorded value when needed.

## Randomized episodes

RCRL imposes its constraint on the states in the support of the initial-state distribution (Prop. 4.1 of the paper), and the dual update only learns `lambda(s)` for states that appear in the replay buffer. Every episode therefore starts from a randomly sampled state, so the policy and the safety value are trained over the feasible set as a whole instead of from a single fixed start.

One `env.reset(...)` call does all of the following, in this order:

1. select a **layout** uniformly from the training layouts,
2. sample a **start state** via `CorridorEnvironment.sample_random_state` on that layout — a position with `vehicle_radius + margin` clearance from the walls and `pillar_radius + vehicle_radius + margin` clearance from every pillar, and a velocity drawn uniformly from a disk of radius `--random-reset-max-speed`, and
3. sample a **goal** the same way (with zero velocity), resampled until it is at least `--min-start-goal-distance` away from the start (default `2 * goal_radius`, 1.0 m).

Sampling after the layout is selected guarantees the start and goal are feasible for the layout the episode actually runs on. With a large minimum distance, some layouts admit few or no valid goal positions for a given start; the goal resampling is bounded and `reset` raises a `RuntimeError` instead of looping forever when no valid goal is found within 1000 attempts. The resolved distance is recorded in `config.json`, and evaluation uses it unless `evaluation_scripts/evaluate.py --min-start-goal-distance` overrides it; checkpoints without the field fall back to two goal radii.

Full RCRL training runs from step 0 — there is no separate exploration or warm-up phase; exploration comes from the per-step action noise (`--expl-noise`).

| Option | Default | Purpose |
| --- | ---: | --- |
| `--random-reset-margin` | `0.3` | extra clearance (m) beyond the collision boundary for sampled start and goal states |
| `--random-reset-max-speed` | `1.0` | maximum speed (m/s) of the sampled start velocity |
| `--min-start-goal-distance` | `2 * goal_radius` | minimum Euclidean start-goal distance (m) for sampled episodes; infeasible values raise an error at reset |

Evaluation follows the same regime: `evaluation_scripts/evaluate.py` runs `--eval-episodes-per-layout` episodes (default 20) per test layout, each with a freshly sampled start and goal and seeded by episode index. `evaluation_scripts/evaluate_visual.py` and `evaluation_scripts/evaluate_gif.py` render one episode per layout, seeded by layout index.

## Important limits

RCRL's formal guarantees assume a deterministic Markov decision process, exact multi-time-scale updates, bounded/projection-compatible parameterizations, and an initial-state distribution that covers the largest feasible set (the randomized episode starts in [Randomized episodes](#randomized-episodes) widen this distribution). This corridor experiment uses neural networks, finite replay training, and observations made from velocity plus lidar rather than the full physical state. Treat success and collision rates as empirical evidence, not a formal persistent-safety guarantee.

The actor has no runtime safety filter. It returns its learned action directly, so safety depends on the learned safety critic and the data encountered during training.

## Acknowledgements

The TD3 implementation in `agents/td3.py`, its training loop, and the `ReplayBuffer` in `utils/replay_buffer.py` are adapted from Scott Fujimoto's TD3 reference implementation (https://github.com/sfujim/TD3), which is distributed under the MIT License. `agents/rcrl_td3.py` builds on that implementation; the RCRL objective and statewise multiplier follow the paper cited below.

## Reference

Yu, D., Ma, H., Li, S. E., & Chen, J. (2022). *Reachability Constrained Reinforcement Learning*. ICML 2022. https://proceedings.mlr.press/v162/yu22d.html
