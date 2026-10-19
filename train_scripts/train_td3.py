"""Train the plain TD3 agent in the corridor environment.

Same randomized start+goal episode regime as train_rcrl_td3.py, but
without the safety critic and multiplier.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from agents.td3 import TD3
from environment import create_env
from utils import ReplayBuffer


def evaluate_policy(policy, layout_set, normalize_obs=True):
    """Evaluate a deterministic policy once on every validation layout.

    Each layout gets one sampled feasible start and one sampled feasible
    goal, so evaluation follows the same randomized regime as training.
    Layouts are seeded by index, which makes the runs reproducible.
    """
    env = create_env(layout_set, normalize_obs=normalize_obs)
    total_reward = 0.0
    total_steps = 0
    successes = 0
    collisions = 0

    for layout_index in range(len(env.val_layouts)):
        env.reset(layout_type="val", layout_index=layout_index, seed=layout_index)
        start = env.sample_random_state(margin=0.3, max_speed=1.0)
        goal = env.sample_random_state(margin=0.3, max_speed=0.0)
        while np.hypot(start[0] - goal[0], start[1] - goal[1]) <= 2 * env.goal_radius:
            goal = env.sample_random_state(margin=0.3, max_speed=0.0)
        state, _ = env.reset(layout_type="fixed", state=start, goal=goal)

        episode_reward = 0.0
        episode_steps = 0

        while True:
            action = policy.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            if terminated or truncated:
                break

        total_reward += episode_reward
        total_steps += episode_steps
        successes += int(info["goal_reached"])
        collisions += int(info["collision"])

    num_layouts = len(env.val_layouts)
    return {
        "mean_reward": total_reward / num_layouts,
        "mean_steps": total_steps / num_layouts,
        "success_rate": successes / num_layouts,
        "collision_rate": collisions / num_layouts,
    }


def episode_reset(env, random_reset_margin, random_reset_max_speed):
    """Reset for a new episode: sample a layout, a random feasible start,
    and a random feasible goal separated from the start."""
    start = env.sample_random_state(
        margin=random_reset_margin, max_speed=random_reset_max_speed
    )
    goal = env.sample_random_state(margin=random_reset_margin, max_speed=0.0)
    while np.hypot(start[0] - goal[0], start[1] - goal[1]) <= 2 * env.goal_radius:
        goal = env.sample_random_state(margin=random_reset_margin, max_speed=0.0)
    return env.reset(layout_type="train", state=start, goal=goal)


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

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
    )
    if args.random_reset_max_speed > env.dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args.replay_size,
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "td3")
    config = vars(args) | {
        "agent": "td3",
        "layout_set": args.layout_set,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "dt": 0.1,
        "max_velocity": 3.0,
        "corridor_length": 30.0,
        "corridor_width": 5.0,
        "lidar_range": 5.0,
        "num_lidar_rays": 64,
        "goal_radius": env.goal_radius,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Saving checkpoints to: {model_path}")

    # The first reset seeds the environment RNG; episode_reset then picks
    # the layout and the random start and goal states.
    env.reset(layout_type="train", seed=args.seed)
    state, _ = episode_reset(
        env, args.random_reset_margin, args.random_reset_max_speed
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

        # Time-limit truncations retain bootstrap targets; collisions and goals do not.
        replay_buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state

        if replay_buffer.size >= args.batch_size:
            policy.train(replay_buffer, batch_size=args.batch_size)

        if terminated or truncated:
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
            print(f"Step {timestep + 1}: replay_size={replay_buffer.size}")

        if args.eval_freq > 0 and (timestep + 1) % args.eval_freq == 0:
            results = evaluate_policy(
                policy, args.layout_set, args.normalize_obs
            )
            print(
                f"Evaluation at {timestep + 1}: "
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
    if args.batch_size < 1 or args.replay_size < args.batch_size:
        parser.error("--replay-size must be at least --batch-size, both positive")
    if args.policy_freq < 1:
        parser.error("--policy-freq must be positive")
    if args.random_reset_margin < 0:
        parser.error("--random-reset-margin must be nonnegative")
    if args.random_reset_max_speed <= 0:
        parser.error("--random-reset-max-speed must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
