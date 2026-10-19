"""Train RCRL-TD3 with vectorized environments and a device-resident buffer.

Runs ``--num-envs`` independent corridor environments. Each vectorized step
collects one transition per environment and then performs one gradient
update per collected transition, so the update-to-data ratio and every
per-transition semantic match the sequential trainer
``train_scripts/train_rcrl_td3.py``. Only the execution schedule differs:

- the actor runs once per vectorized step on the stacked observations
  instead of once per environment step;
- the replay buffer lives on the training device, so sampled batches never
  round-trip through the host;
- the per-step gradient updates run back-to-back, letting the GPU overlap
  kernel execution with launch overhead.

This amortization is what makes GPU training worthwhile for the small
(3x256) networks; the sequential trainer pays the full per-step overhead
once per environment step.

Examples:
  python -m train_scripts.train_rcrl_td3_vec --name rcrl_vec --num-envs 32
  ./train.sh rcrl-vec --name quick --max-timesteps 8 --eval-freq 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from agents.rcrl_td3 import RCRLTD3, device
from environment import create_env
from train_scripts.train_rcrl_td3 import (
    episode_reset,
    evaluate_policy,
    set_seed,
)
from utils import ReplayBufferRCRLTD3Torch


def main(args):
    set_seed(args.seed)

    envs = [
        create_env(args.layout_set, normalize_obs=args.normalize_obs)
        for _ in range(args.num_envs)
    ]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = float(envs[0].action_space.high[0])

    policy = RCRLTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        safety_lr=args.safety_lr,
        grad_clip=args.grad_clip,
        multiplier_lr=args.multiplier_lr,
        multiplier_freq=args.multiplier_freq,
        max_lambda=args.max_lambda,
        initial_lambda=args.initial_lambda,
        multiplier_type=args.multiplier,
    )
    if args.random_reset_max_speed > envs[0].dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    replay_buffer = ReplayBufferRCRLTD3Torch(
        state_dim,
        action_dim,
        max_size=args.replay_size,
        device=device,
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "rcrl_td3")
    config = vars(args) | {
        "agent": "rcrl_td3",
        "trainer": "vectorized",
        "layout_set": args.layout_set,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": str(device),
        "dt": 0.1,
        "max_velocity": 3.0,
        "corridor_length": 30.0,
        "corridor_width": 5.0,
        "lidar_range": 5.0,
        "num_lidar_rays": 64,
        "goal_radius": envs[0].goal_radius,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Device: {device}")
    print(f"Saving checkpoints to: {model_path}")

    # Per-environment RNGs are offset by index so the environments explore
    # decorrelated action/layout sequences.
    observations = np.empty((args.num_envs, state_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        env.action_space.seed(args.seed + i)
        # The first reset seeds the environment RNG; episode_reset then
        # picks the layout and the random start and goal states.
        env.reset(layout_type="train", seed=args.seed + i)
        observation, _ = episode_reset(
            env, args.random_reset_margin, args.random_reset_max_speed
        )
        observations[i] = observation

    # Per-step batched transition storage, filled by the stepping loop.
    states = np.empty((args.num_envs, state_dim), dtype=np.float32)
    actions = np.empty((args.num_envs, action_dim), dtype=np.float32)
    next_states = np.empty((args.num_envs, state_dim), dtype=np.float32)
    rewards = np.empty((args.num_envs, 1), dtype=np.float32)
    constraints = np.empty((args.num_envs, 1), dtype=np.float32)
    next_constraints = np.empty((args.num_envs, 1), dtype=np.float32)
    dones = np.empty((args.num_envs, 1), dtype=np.float32)

    # Per-environment episode tracking, plus aggregates since the last log.
    episode_reward = np.zeros(args.num_envs)
    episode_steps = np.zeros(args.num_envs, dtype=int)
    window_rewards = []
    window_lengths = []
    window_collisions = 0
    window_goals = 0

    env_steps_done = 0
    next_log = args.log_freq if args.log_freq > 0 else None
    next_eval = args.eval_freq if args.eval_freq > 0 else None
    start_time = time.perf_counter()

    meta_steps = -(-args.max_timesteps // args.num_envs)  # ceil division
    for _ in range(meta_steps):
        # One actor forward pass for all environments.
        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            policy_actions = policy.actor(obs_tensor).cpu().numpy()
        noise = np.random.normal(
            loc=0.0,
            scale=max_action * args.expl_noise,
            size=(args.num_envs, action_dim),
        )
        actions[:] = np.clip(policy_actions + noise, -max_action, max_action)

        for i, env in enumerate(envs):
            next_state, reward, terminated, truncated, info = env.step(actions[i])

            states[i] = observations[i]
            next_states[i] = next_state
            rewards[i, 0] = reward
            constraints[i, 0] = info["constraint"]
            next_constraints[i, 0] = info["next_constraint"]
            # Collisions and goals terminate; time-limit truncations keep
            # bootstrap targets, matching the sequential trainer.
            dones[i, 0] = float(terminated)

            episode_reward[i] += reward
            episode_steps[i] += 1

            if terminated or truncated:
                window_rewards.append(episode_reward[i])
                window_lengths.append(episode_steps[i])
                window_collisions += int(info["collision"])
                window_goals += int(info["goal_reached"])
                observation, _ = episode_reset(
                    env, args.random_reset_margin, args.random_reset_max_speed
                )
                observations[i] = observation
                episode_reward[i] = 0.0
                episode_steps[i] = 0
            else:
                observations[i] = next_state

        replay_buffer.add_batch(
            states,
            actions,
            next_states,
            rewards,
            constraints,
            next_constraints,
            dones,
        )
        env_steps_done += args.num_envs

        # One update per collected transition, back-to-back so the GPU can
        # overlap kernel execution with launch overhead.
        if replay_buffer.size >= args.batch_size:
            for _ in range(args.num_envs):
                policy.train(replay_buffer, batch_size=args.batch_size)

        if next_log is not None and env_steps_done >= next_log:
            next_log += args.log_freq
            elapsed = time.perf_counter() - start_time
            with torch.no_grad():
                mean_lambda = policy.mean_lambda(observations)
            message = (
                f"Step {env_steps_done}: replay_size={replay_buffer.size}, "
                f"mean_lambda={mean_lambda:.4f}, "
                f"{env_steps_done / elapsed:.0f} steps/s"
            )
            if window_rewards:
                message += (
                    f" | episodes={len(window_rewards)}, "
                    f"mean_reward={np.mean(window_rewards):.3f}, "
                    f"mean_length={np.mean(window_lengths):.1f}, "
                    f"collision_rate={window_collisions / len(window_rewards):.1%}, "
                    f"goal_rate={window_goals / len(window_rewards):.1%}"
                )
            print(message)
            window_rewards.clear()
            window_lengths.clear()
            window_collisions = 0
            window_goals = 0

        if next_eval is not None and env_steps_done >= next_eval:
            next_eval += args.eval_freq
            results = evaluate_policy(
                policy, args.layout_set, args.normalize_obs
            )
            print(
                f"Evaluation at {env_steps_done}: "
                f"mean_reward={results['mean_reward']:.3f}, "
                f"mean_steps={results['mean_steps']:.1f}, "
                f"success_rate={results['success_rate']:.1%}, "
                f"collision_rate={results['collision_rate']:.1%}"
            )
            policy.save(model_path)

    policy.save(model_path)
    print(f"Training complete. Model saved to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument("--output-dir", default="models", help="Checkpoint root")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Number of parallel environments; each vectorized step "
             "collects one transition and runs one update per environment",
    )
    parser.add_argument(
        "--layout-set",
        default="one",
        help="Layout set name; loads layouts/<name>_{train,test,validation}.npy",
    )
    parser.add_argument(
        "--normalize-obs",
        dest="normalize_obs",
        action="store_true",
        default=True,
        help="Scale observation components by their known ranges (default); "
             "checkpoints trained with --no-normalize-obs must be evaluated "
             "with --no-normalize-obs",
    )
    parser.add_argument(
        "--no-normalize-obs",
        dest="normalize_obs",
        action="store_false",
        help="Disable observation scaling",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--expl-noise", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=2)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--safety-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global gradient norm for all networks; 0 disables clipping")
    parser.add_argument("--multiplier-lr", type=float, default=1e-6)
    parser.add_argument("--multiplier-freq", type=int, default=6)
    parser.add_argument(
        "--multiplier",
        choices=["network", "scalar"],
        default="scalar",
        help="'network' uses a statewise lambda(s) network; 'scalar' uses a "
             "single fixed Lagrange multiplier for all states, set by "
             "--initial-lambda and not updated during training",
    )
    parser.add_argument("--max-lambda", type=float, default=100.0)
    parser.add_argument("--initial-lambda", type=float, default=0.1)
    parser.add_argument("--random-reset-margin", type=float, default=0.3,
                        help="Extra clearance (m) beyond the collision "
                             "boundary for sampled start and goal states")
    parser.add_argument("--random-reset-max-speed", type=float, default=1.0,
                        help="Maximum speed (m/s) of the sampled start velocity")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Set to 0 to disable evaluation")
    parser.add_argument("--log-freq", type=int, default=1_000,
                        help="Set to 0 to disable progress logging")
    args = parser.parse_args()

    if args.max_timesteps < 1:
        parser.error("--max-timesteps must be positive")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.batch_size < 1 or args.replay_size < args.batch_size:
        parser.error("--replay-size must be at least --batch-size, both positive")
    if args.policy_freq < 1:
        parser.error("--policy-freq must be positive")
    if args.multiplier_freq < args.policy_freq:
        parser.error("--multiplier-freq must be at least --policy-freq")
    if args.max_lambda <= 0 or not 0 < args.initial_lambda <= args.max_lambda:
        parser.error("--initial-lambda must be in (0, --max-lambda]")
    if args.random_reset_margin < 0:
        parser.error("--random-reset-margin must be nonnegative")
    if args.random_reset_max_speed <= 0:
        parser.error("--random-reset-max-speed must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
