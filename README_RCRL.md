# RCRL-TD3 additions

This note describes the reachability-constrained reinforcement learning (RCRL) pieces added alongside the existing TD3 implementation. The RCRL agent keeps TD3's reward-learning setup and adds a second critic that estimates future constraint values.

## What was added

### `agents/rcrl_td3.py`

`RCRL_TD3` is the agent implementation. It contains:

- **Actor:** the same deterministic policy structure used by TD3. It maps the current observation to an action.
- **Reward critic:** TD3's twin Q networks, trained to estimate expected reward.
- **Safety critic:** `SafetyCritic`, a separate network that takes an observation and action and estimates the largest constraint value encountered along the policy's future trajectory.
- **Target networks:** slowly updated copies used to form stable reward and safety learning targets.

The environment's constraint is `vehicle_radius - min(lidar_distances)`. A value at or below zero means the nearest sensed surface is at least one vehicle radius away; a positive value means the vehicle's circular footprint would overlap that surface. The safety critic uses the same sign convention: predicted value `<= 0` means safe according to its estimate, and `> 0` means predicted violation.

For a nonterminal transition, its training target is the maximum of the observed next-state constraint and the target safety critic's estimate at the next state under the target actor. For terminal transitions, it uses the observed next-state constraint without bootstrapping. This teaches the critic to represent the largest predicted constraint along the policy continuation.

The actor objective currently combines reward and a fixed safety penalty:

```text
actor_loss = -mean(reward_Q) + mean(relu(safety_Q))
```

The ReLU penalizes predicted positive constraint values. Its weight is currently `1.0`; a Lagrange multiplier has not been implemented. During this update, critic parameters are frozen while gradients still pass through the critics to the action, so the actor can learn from their outputs without adding actor-loss gradients to critic parameters.

### `ReplayBufferRCRLTD3` in `utils.py`

The RCRL buffer stores the usual transition data—observation, action, next observation, reward, and terminal flag—plus the **next-state constraint**. `RCRL_TD3.train()` samples that constraint to update the safety critic.

In the RCRL training loop, true terminal states (`terminated`, such as collision or reaching the goal) stop bootstrapping. Time-limit truncations do not stop bootstrapping.

### `train_rcrl_td3.py`

This script trains the RCRL agent in the corridor environment, stores the constraint from `info["constraint"]` on every transition, periodically evaluates on validation layouts, and saves a final checkpoint. It retains the environment's existing observation: velocity and lidar readings, without absolute position. This supports training across layouts without directly providing global coordinates.

Example full run:

```bash
python train_rcrl_td3.py --name my_rcrl_run
```

Useful options:

```text
--max-timesteps   Training steps (default: 30000)
--start-timesteps Random-action exploration steps (default: 10000)
--batch-size      Minibatch size (default: 256)
--seed            Random seed (default: 0)
--eval-freq       Evaluation interval; use 0 to disable (default: 5000)
--output-dir      Checkpoint root (default: models)
```

For a short run to check the training path:

```bash
python train_rcrl_td3.py --name rcrl_smoke --max-timesteps 8 \
  --start-timesteps 0 --batch-size 2 --replay-size 32 \
  --eval-freq 0 --log-freq 1 --output-dir /tmp/rcrl_runs
```

### Evaluation scripts

Both `evaluate_td3.py` and `evaluate_td3_visual.py` accept `--agent td3` or `--agent rcrl_td3`. The default is `td3`, so existing TD3 commands continue to work. The selected agent determines which checkpoint prefix is loaded from `models/<name>/`.

Evaluate all test layouts:

```bash
python evaluate_td3.py --name my_rcrl_run --agent rcrl_td3
```

Render an episode:

```bash
python evaluate_td3_visual.py --name my_rcrl_run --agent rcrl_td3
```

### Checkpoints

RCRL training writes files under `models/<name>/` using the `rcrl_td3` prefix. Besides actor and reward-critic weights and optimizer states, the checkpoint includes safety-critic weights and its optimizer state. Loading reconstructs target networks from the loaded online networks.

## Current limits

The safety critic is a learned estimate and the actor uses a soft penalty. `select_action()` returns the actor's action directly; there is no runtime action filter or formal safety guarantee. The estimate also depends on the training data, the constraint signal, and the policy used for bootstrapping. Treat collision and success rates on held-out layouts as empirical results, not proof of safety.

The constraint is based on sampled lidar rays and the environment checks collision at the end of a simulation step. It can therefore miss geometric or between-step collision details. The fixed safety-penalty weight is intended for an initial runnable implementation and should be tuned or replaced by a constrained optimization method for further experiments.
