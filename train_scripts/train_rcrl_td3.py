"""Train the reachability-constrained TD3 agent in the corridor environment."""

import argparse
import json
import os
import random

import numpy as np
import torch

from agents.rcrl_td3 import RCRLTD3
from environment import create_env
from evaluation import evaluate_policy
from utils import ReplayBufferRCRLTD3


def episode_reset(env, random_reset_margin, random_reset_max_speed, seed=None):
    """Reset into a new randomized training episode in one call.

    ``reset`` selects the layout first and then samples a feasible start and
    goal on that layout, so both always satisfy the clearance margins for the
    layout the episode actually runs on. The goal is resampled until it is at
    least two goal radii from the start.
    """
    return env.reset(
        layout_type="train",
        seed=seed,
        sample_start=True,
        start_margin=random_reset_margin,
        start_max_speed=random_reset_max_speed,
        goal_margin=random_reset_margin,
        min_start_goal_distance=2 * env.goal_radius,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    env = create_env(args.layout_set, normalize_obs=args.normalize_obs)
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

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
    if args.random_reset_max_speed > env.dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    replay_buffer = ReplayBufferRCRLTD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args.replay_size,
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "rcrl_td3")
    config = vars(args) | {
        "agent": "rcrl_td3",
        "layout_set": args.layout_set,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "dt": env.dynamics.dt,
        "max_velocity": env.dynamics.max_velocity,
        "corridor_length": env.length,
        "corridor_width": env.width,
        "vehicle_radius": env.vehicle_radius,
        "lidar_range": env.lidar_range,
        "num_lidar_rays": env.num_lidar_rays,
        "max_acceleration": env.max_acceleration,
        "goal_radius": env.goal_radius,
        "max_steps": env.max_steps,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Saving checkpoints to: {model_path}")

    # The first episode reset also seeds the environment RNG.
    state, _ = episode_reset(
        env, args.random_reset_margin, args.random_reset_max_speed, seed=args.seed
    )
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0

    for timestep in range(args.max_timesteps):
        episode_steps += 1
        action = policy.select_action(state)
        noise = np.random.normal(
            loc=0.0,
            scale=max_action * args.expl_noise,
            size=action_dim,
        )
        action = np.clip(action + noise, -max_action, max_action)

        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # info["constraint"] is h(s) and info["next_constraint"] is h(s').
        # Time-limit truncations retain bootstrap targets; collisions and goals do not.
        replay_buffer.add(
            state,
            action,
            next_state,
            reward,
            info["constraint"],
            info["next_constraint"],
            float(terminated),
        )
        state = next_state

        if replay_buffer.size >= args.batch_size:
            policy.train(replay_buffer, batch_size=args.batch_size)

        if terminated or truncated:
            if episode_num % 20 == 0:
                print(
                    f"Episode {episode_num:4d} | step {timestep + 1:7d} | "
                    f"length {episode_steps:4d} | reward {episode_reward:8.3f} | "
                    f"collision {info['collision']} | goal {info['goal_reached']}"
                )
            state, _ = episode_reset(
                env, args.random_reset_margin, args.random_reset_max_speed
            )
            episode_reward = 0.0
            episode_steps = 0
            episode_num += 1

        if args.log_freq > 0 and (timestep + 1) % args.log_freq == 0:
            with torch.no_grad():
                mean_lambda = policy.mean_lambda(state)
            print(
                f"Step {timestep + 1}: replay_size={replay_buffer.size}, "
                f"mean_lambda={mean_lambda:.4f}"
            )

        if args.eval_freq > 0 and (timestep + 1) % args.eval_freq == 0:
            results = evaluate_policy(
                policy,
                args.layout_set,
                args.normalize_obs,
                episodes_per_layout=args.eval_episodes_per_layout,
                random_reset_margin=args.random_reset_margin,
                random_reset_max_speed=args.random_reset_max_speed,
            )
            print(
                f"Evaluation at {timestep + 1}: "
                f"mean_reward={results['mean_reward']:.3f}, "
                f"mean_steps={results['mean_steps']:.1f}, "
                f"success_rate={results['success_rate']:.1%} "
                f"(95% CI {results['success_rate_ci'][0]:.1%}-"
                f"{results['success_rate_ci'][1]:.1%}), "
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
    parser.add_argument("--policy-noise", type=float, default=0.1)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=2)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--safety-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global gradient norm for all networks; 0 disables clipping")
    parser.add_argument("--multiplier-lr", type=float, default=1e-6)
    parser.add_argument("--multiplier-freq", type=int, default=6)
    parser.add_argument("--max-lambda", type=float, default=100.0)
    parser.add_argument("--initial-lambda", type=float, default=10.0)
    parser.add_argument(
        "--multiplier",
        choices=["network", "scalar"],
        default="scalar",
        help="'network' uses a statewise lambda(s) network; 'scalar' uses a "
             "single fixed Lagrange multiplier for all states, set by "
             "--initial-lambda and not updated during training",
    )
    parser.add_argument("--random-reset-margin", type=float, default=0.3,
                        help="Extra clearance (m) beyond the collision "
                             "boundary for sampled start and goal states")
    parser.add_argument("--random-reset-max-speed", type=float, default=1.0,
                        help="Maximum speed (m/s) of the sampled start velocity")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Set to 0 to disable evaluation")
    parser.add_argument("--eval-episodes-per-layout", type=int, default=20,
                        help="Episodes with sampled start/goal pairs to run "
                             "on each validation layout during evaluation")
    parser.add_argument("--log-freq", type=int, default=1_000,
                        help="Set to 0 to disable progress logging")
    args = parser.parse_args()

    if args.max_timesteps < 1:
        parser.error("--max-timesteps must be positive")
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
    if args.eval_episodes_per_layout < 1:
        parser.error("--eval-episodes-per-layout must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
