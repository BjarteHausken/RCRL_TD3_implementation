"""Train plain TD3 with vectorized environments and a device-resident buffer.

Runs ``--num-envs`` independent corridor environments. Each vectorized step
collects one transition per environment and then performs one gradient
update per collected transition, so the update-to-data ratio and every
per-transition semantic match the sequential trainer
``train_scripts/train_td3.py``. Only the execution schedule differs:

- the actor runs once per vectorized step on the stacked observations
  instead of once per environment step;
- the replay buffer lives on the training device, so sampled batches never
  round-trip through the host;
- the per-step gradient updates run back-to-back, letting the GPU overlap
  kernel execution with launch overhead.

The episode regime is the same randomized start+goal sampling used by the
sequential trainer: no separate warm-up phase, exploration comes from the
per-step action noise.

Examples:
  python -m train_scripts.train_td3_vec --name td3_vec --num-envs 32
  ./train.sh td3-vec --name quick --max-timesteps 8 --eval-freq 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from agents.td3 import TD3, device
from environment import create_env, step_batch
from train_scripts.train_td3 import (
    episode_reset,
    evaluate_policy,
    set_seed,
)
from utils.training_log import TrainingLogger, evaluation_metrics
from utils import ReplayBufferTD3Torch


def main(args):
    set_seed(args.seed)

    envs = [
        create_env(
            args.layout_set,
            normalize_obs=args.normalize_obs,
            reward_scale=args.reward_scale,
            max_steps=args.max_steps,
        )
        for _ in range(args.num_envs)
    ]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = float(envs[0].action_space.high[0])
    if args.min_start_goal_distance is None:
        args.min_start_goal_distance = 2.0 * envs[0].goal_radius

    policy = TD3(
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
        grad_clip=args.grad_clip,
        critic_lr_final=args.critic_lr_final,
        actor_lr_final=args.actor_lr_final,
        # One update per collected transition, so the anneal spans the
        # same number of updates as env steps.
        anneal_steps=(
            args.max_timesteps if args.lr_decay == "linear" else None
        ),
    )
    if args.random_reset_max_speed > envs[0].dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    replay_buffer = ReplayBufferTD3Torch(
        state_dim,
        action_dim,
        max_size=args.replay_size,
        device=device,
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "td3")
    config = vars(args) | {
        "agent": "td3",
        "trainer": "vectorized",
        "layout_set": args.layout_set,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": str(device),
        "dt": envs[0].dynamics.dt,
        "max_velocity": envs[0].dynamics.max_velocity,
        "corridor_length": envs[0].length,
        "corridor_width": envs[0].width,
        "vehicle_radius": envs[0].vehicle_radius,
        "lidar_range": envs[0].lidar_range,
        "num_lidar_rays": envs[0].num_lidar_rays,
        "max_acceleration": envs[0].max_acceleration,
        "goal_radius": envs[0].goal_radius,
        "max_steps": envs[0].max_steps,
        "reward_scale": envs[0].reward_scale,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    training_log = TrainingLogger(run_dir)
    training_log.log_run(
        config, trainer="vectorized", num_envs=args.num_envs
    )

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
        # The first episode reset also seeds the environment RNG.
        observation, _ = episode_reset(
            env,
            args.random_reset_margin,
            args.random_reset_max_speed,
            min_start_goal_distance=args.min_start_goal_distance,
            seed=args.seed + i,
        )
        observations[i] = observation

    # Per-step batched transition storage, filled by the stepping loop.
    states = np.empty((args.num_envs, state_dim), dtype=np.float32)
    actions = np.empty((args.num_envs, action_dim), dtype=np.float32)
    rewards = np.empty((args.num_envs, 1), dtype=np.float32)
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
    next_save = args.save_freq if args.save_freq > 0 else None
    start_time = time.perf_counter()

    meta_steps = -(-args.max_timesteps // args.num_envs)  # ceil division
    try:
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

            states[:] = observations
            next_states, step_rewards, terminated, truncated, info = step_batch(
                envs, actions
            )
            rewards[:, 0] = step_rewards
            # Collisions and goals terminate; time-limit truncations keep
            # bootstrap targets, matching the sequential trainer.
            dones[:, 0] = terminated

            # Episode bookkeeping and resets stay per environment; only the
            # environment stepping itself is batched.
            for i, env in enumerate(envs):
                episode_reward[i] += step_rewards[i]
                episode_steps[i] += 1

                if terminated[i] or truncated[i]:
                    window_rewards.append(episode_reward[i])
                    window_lengths.append(episode_steps[i])
                    window_collisions += int(info["collision"][i])
                    window_goals += int(info["goal_reached"][i])
                    observation, _ = episode_reset(
                        env,
                        args.random_reset_margin,
                        args.random_reset_max_speed,
                        min_start_goal_distance=args.min_start_goal_distance,
                    )
                    observations[i] = observation
                    episode_reward[i] = 0.0
                    episode_steps[i] = 0
                else:
                    observations[i] = next_states[i]

            replay_buffer.add_batch(
                states,
                actions,
                next_states,
                rewards,
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
                window_count = len(window_rewards)
                message = (
                    f"Step {env_steps_done}: replay_size={replay_buffer.size}, "
                    f"{env_steps_done / elapsed:.0f} steps/s"
                )
                if window_count:
                    message += (
                        f" | episodes={window_count}, "
                        f"mean_reward={np.mean(window_rewards):.3f}, "
                        f"mean_length={np.mean(window_lengths):.1f}, "
                        f"collision_rate={window_collisions / window_count:.1%}, "
                        f"goal_rate={window_goals / window_count:.1%}"
                    )
                print(message)
                log_metrics = {
                    "replay_size": replay_buffer.size,
                    "steps_per_second": env_steps_done / elapsed,
                }
                if window_count:
                    log_metrics.update(
                        episodes=window_count,
                        mean_reward=float(np.mean(window_rewards)),
                        mean_length=float(np.mean(window_lengths)),
                        collision_rate=window_collisions / window_count,
                        goal_rate=window_goals / window_count,
                    )
                training_log.log("progress", env_steps_done, **log_metrics)
                window_rewards.clear()
                window_lengths.clear()
                window_collisions = 0
                window_goals = 0

            if next_save is not None and env_steps_done >= next_save:
                next_save += args.save_freq
                policy.save(model_path)
                print(f"Step {env_steps_done}: checkpoint saved to {model_path}")

            if next_eval is not None and env_steps_done >= next_eval:
                next_eval += args.eval_freq
                results = evaluate_policy(
                    policy,
                    args.layout_set,
                    args.normalize_obs,
                    episodes_per_layout=args.eval_episodes_per_layout,
                    random_reset_margin=args.random_reset_margin,
                    random_reset_max_speed=args.random_reset_max_speed,
                    min_start_goal_distance=args.min_start_goal_distance,
                    reward_scale=args.reward_scale,
                )
                print(
                    f"Evaluation at {env_steps_done}: "
                    f"mean_reward={results['mean_reward']:.3f}, "
                    f"mean_steps={results['mean_steps']:.1f}, "
                    f"success_rate={results['success_rate']:.1%} "
                    f"(95% CI {results['success_rate_ci'][0]:.1%}-"
                    f"{results['success_rate_ci'][1]:.1%}), "
                    f"collision_rate={results['collision_rate']:.1%}, "
                    f"margin_violation_rate={results['margin_violation_rate']:.1%}, "
                    f"margin_step_fraction={results['margin_step_fraction']:.2%}, "
                    f"mean_min_clearance={results['mean_min_clearance']:.3f} m"
                )
                training_log.log(
                    "eval", env_steps_done, **evaluation_metrics(results)
                )
                policy.save(model_path)
    except KeyboardInterrupt:
        policy.save(model_path)
        print(f"Interrupted. Checkpoint saved to: {model_path}")
        return
    finally:
        training_log.close()

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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--expl-noise", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.1)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument(
        "--critic-lr-final",
        type=float,
        default=1e-6,
        help="Final critic learning rate of the linear anneal (paper: 1e-6)",
    )
    parser.add_argument(
        "--actor-lr-final",
        type=float,
        default=1e-6,
        help="Final actor learning rate of the linear anneal (paper: 1e-6)",
    )
    parser.add_argument(
        "--lr-decay",
        choices=["linear", "none"],
        default="linear",
        help="'linear' anneals every learning rate from its initial to its "
             "final value over --max-timesteps, as in the paper (Table 1); "
             "'none' keeps the initial rates constant",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global gradient norm for all networks; 0 disables clipping")
    parser.add_argument("--random-reset-margin", type=float, default=0.3,
                        help="Extra clearance (m) beyond the collision "
                             "boundary for sampled start and goal states")
    parser.add_argument("--random-reset-max-speed", type=float, default=1.0,
                        help="Maximum speed (m/s) of the sampled start velocity")
    parser.add_argument("--min-start-goal-distance", type=float, default=None,
                        help="Minimum start-goal distance (m) for sampled "
                             "episodes; default is 2 * goal_radius (1.0 m). "
                             "Distances close to the corridor length may be "
                             "infeasible on some layouts, in which case reset "
                             "raises an error")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Fixed reward divisor used by the environment; "
                             "1.0 uses the unscaled reward. Match it across "
                             "agents so rewards are comparable")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="Maximum episode length N in environment steps "
                             "(paper: 360); episodes are truncated at this "
                             "length")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Set to 0 to disable evaluation")
    parser.add_argument("--save-freq", type=int, default=5_000,
                        help="Save a checkpoint every this many steps, "
                             "independent of evaluation; 0 disables "
                             "intermediate saves (default: 5000)")
    parser.add_argument("--eval-episodes-per-layout", type=int, default=20,
                        help="Episodes with sampled start/goal pairs to run "
                             "on each validation layout during evaluation")
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
    for name in ("critic_lr", "actor_lr", "critic_lr_final",
                 "actor_lr_final"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.random_reset_margin < 0:
        parser.error("--random-reset-margin must be nonnegative")
    if args.random_reset_max_speed <= 0:
        parser.error("--random-reset-max-speed must be positive")
    if (args.min_start_goal_distance is not None
            and args.min_start_goal_distance < 0):
        parser.error("--min-start-goal-distance must be nonnegative")
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if args.eval_episodes_per_layout < 1:
        parser.error("--eval-episodes-per-layout must be positive")
    if args.save_freq < 0:
        parser.error("--save-freq must be nonnegative")
    return args


if __name__ == "__main__":
    main(parse_args())
