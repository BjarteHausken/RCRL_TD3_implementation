# RCRL-TD3

This project adapts Reachability Constrained Reinforcement Learning (RCRL) to the existing TD3 corridor-navigation agent. It retains TD3 for reward learning and adds RCRL's reachability safety value, statewise Lagrange multiplier, and separated update time scales.

The implementation is an adaptation, not a reproduction of the paper's experiments: the paper's Algorithm 1 is a deterministic actor-critic template, which the paper evaluates in both on-policy and off-policy variants. This repository uses TD3 as its off-policy base. The RCRL safety objective, statewise Lagrangian, and dual update are preserved.

## Run training

From the repository root:

```bash
python train_rcrl_td3.py --name my_rcrl_run
```

For a small end-to-end check:

```bash
python train_rcrl_td3.py --name rcrl_smoke --max-timesteps 8 \
  --start-timesteps 0 --batch-size 2 --replay-size 32 \
  --eval-freq 0 --log-freq 1 --output-dir /tmp/rcrl_runs
```

Checkpoints are written under `models/<name>/` with the `rcrl_td3` prefix. Use them with:

```bash
python evaluate_td3.py --name my_rcrl_run --agent rcrl_td3
python evaluate_td3_visual.py --name my_rcrl_run --agent rcrl_td3
```

Pillar layouts live in `layouts/` as `<set>_{train,test,validation}.npy` (see `double_integrator_sim/layouts/generate_layouts.py` for generation). All training and evaluation scripts select the set with `--layout-set` (default `one`); the set name is recorded in each run's `config.json`, so a checkpoint should be evaluated with the same `--layout-set` it was trained on.

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

### 4. Separate update time scales

The reward critic and safety critic update every gradient step. The actor updates every `--policy-freq` steps (default 2), and the multiplier every `--multiplier-freq` steps (default 6). The default learning rates also make critic/safety updates faster than actor updates, and multiplier updates slowest:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--critic-lr` | `1e-4` | TD3 reward-critic learning rate |
| `--safety-lr` | `1e-4` | reachability safety-critic learning rate |
| `--actor-lr` | `2e-5` | policy learning rate |
| `--multiplier-lr` | `1e-6` | statewise dual-ascent learning rate |
| `--policy-freq` | `2` | actor/target update interval |
| `--multiplier-freq` | `6` | multiplier update interval |
| `--max-lambda` | `100` | upper multiplier bound |
| `--initial-lambda` | `0.1` | initial multiplier output |

### 5. Preserve stable TD3 reward learning

The reward critic remains TD3: twin Q-functions, clipped target-policy noise, delayed actor updates, and Polyak target averaging:

```text
target = tau * online + (1 - tau) * target
```

The actor, reward critic, safety critic, multiplier, and each optimizer are saved in every RCRL checkpoint. Older actor/reward-only checkpoints can still be loaded for evaluation, but they cannot faithfully resume RCRL training.

## Feasible-set coverage (random resets and critic warmup)

RCRL imposes its constraint on the states in the support of the initial-state distribution (Prop. 4.1 of the paper), and the dual update only learns `lambda(s)` for states that appear in the replay buffer. With only the fixed start, that support is a single point. On top of that, the time-based reward pushes the agent toward the fastest trajectories, which hug the feasible-set boundary, so the interior of the feasible set (where `V_h` should stay `<= 0`) is underrepresented.

Training is therefore split into two phases:

1. **Random-exploration phase** (`t < --start-timesteps`, default 10000 steps): random actions, and every episode starts from a random safe state. During this phase the reward critic and safety critic are trained (`RCRLTD3.train_warmup`) while the actor and multiplier stay frozen, so `Q_h` is learned over a broad slice of the feasible set before the reward-driven policy starts, at no extra time cost.
2. **Learning phase** (`t >= --start-timesteps`): the policy with exploration noise runs from the standard start (the deployment scenario) with full RCRL training.

| Option | Default | Purpose |
| --- | ---: | --- |
| `--start-timesteps` | `10_000` | length of the random-exploration (critic warmup) phase |
| `--reset-mode` | `fixed` | learning-phase episode starts: `fixed` keeps the standard start; `mixed` starts each episode from a random state with probability `--random-reset-frac` |
| `--random-reset-frac` | `0.15` | probability of a random start in `mixed` mode |
| `--random-reset-margin` | `0.3` | extra clearance (m) beyond the collision boundary for random start states |
| `--random-reset-max-speed` | `1.0` | maximum initial speed (m/s) for random start states |

Random start states are sampled by `CorridorEnvironment.sample_random_state`, which draws positions uniformly from the corridor while keeping `vehicle_radius + margin` clearance from the walls and `pillar_radius + vehicle_radius + margin` clearance from every pillar, and draws the velocity uniformly from a disk of radius `--random-reset-max-speed`.

The defaults already include the coverage mechanism, so the fastest-to-goal objective is covered by the plain command `python train_rcrl_td3.py --name my_rcrl_run`. Use `--reset-mode mixed` to keep injecting random starts during the learning phase (e.g. to maintain constraint data over the wider state set once the base policy has converged). Evaluation always uses the fixed deployment start.

## Important limits

RCRL's formal guarantees assume a deterministic Markov decision process, exact multi-time-scale updates, bounded/projection-compatible parameterizations, and an initial-state distribution that covers the largest feasible set (see [Feasible-set coverage](#feasible-set-coverage-random-resets-and-critic-warmup) for the training options that widen this distribution). This corridor experiment uses neural networks, finite replay training, and observations made from velocity plus lidar rather than the full physical state. Treat success and collision rates as empirical evidence, not a formal persistent-safety guarantee.

The actor has no runtime safety filter. It returns its learned action directly, so safety depends on the learned safety critic and the data encountered during training.

## Reference

Yu, D., Ma, H., Li, S. E., & Chen, J. (2022). *Reachability Constrained Reinforcement Learning*. ICML 2022. https://proceedings.mlr.press/v162/yu22d.html
