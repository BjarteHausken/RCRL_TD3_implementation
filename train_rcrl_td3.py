"""Train TD3 with an action-conditioned reachability safety critic."""

import argparse
import json
import os
import random

import numpy as np
import torch

from agents.rcrl_td3 import RCRL_TD3
from double_integrator_sim.dynamics import DoubleIntegratorExactDiscretization
from double_integrator_sim.environment import CorridorEnvironment
from utils import ReplayBufferRCRLTD3


def create_env():
    dynamics = DoubleIntegratorExactDiscretization(dt=0.1, max_velocity=3.0)
    return CorridorEnvironment(
        dynamics=dynamics,
        length=30.0,
        width=5.0,
        max_acceleration=2.0,
        lidar_range=5.0,
        num_lidar_rays=64,
    )


def evaluate_policy(policy, env):
    total_reward = 0.0
    successes = 0
    collisions = 0
    num_layouts = len(env.val_layouts)

    for layout_idx in range(num_layouts):
        env.set_val_layout(layout_idx)
        state, _ = env.reset(layout_type="fixed")
        episode_reward = 0.0

        while True:
            action = policy.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        total_reward += episode_reward
        successes += int(info["goal_reached"])
        collisions += int(info["collision"])

    return {
        "mean_reward": total_reward / num_layouts,
        "success_rate": successes / num_layouts,
        "collision_rate": collisions / num_layouts,
    }


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = create_env()
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = RCRL_TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
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
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "dt": 0.1,
        "max_velocity": 3.0,
        "corridor_length": 30.0,
        "corridor_width": 5.0,
        "lidar_range": 5.0,
        "num_lidar_rays": 64,
        "safety_penalty_weight": 1.0,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)

    state, _ = env.reset(layout_type="train", seed=args.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0

    for timestep in range(args.max_timesteps):
        episode_steps += 1
        if timestep < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state)
            action += np.random.normal(0.0, max_action * args.expl_noise, size=action_dim)
            action = np.clip(action, -max_action, max_action)

        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Bootstrap across time-limit truncations, but not collision/goal terminals.
        replay_buffer.add(
            state,
            action,
            next_state,
            reward,
            info["constraint"],
            float(terminated),
        )
        state = next_state

        if timestep >= args.start_timesteps and replay_buffer.size >= args.batch_size:
            metrics = policy.train(replay_buffer, batch_size=args.batch_size)
            if (timestep + 1) % args.log_freq == 0:
                print(
                    f"Step {timestep + 1}: critic_loss={metrics['critic_loss']:.4f}, "
                    f"safety_loss={metrics['safety_loss']:.4f}, "
                    f"mean_constraint={metrics['mean_constraint']:.4f}"
                )

        if terminated or truncated:
            print(
                f"Episode {episode_num}: step={timestep + 1}, steps={episode_steps}, "
                f"reward={episode_reward:.3f}, collision={info['collision']}, "
                f"goal={info['goal_reached']}"
            )
            state, _ = env.reset(layout_type="train")
            episode_reward = 0.0
            episode_steps = 0
            episode_num += 1

        if args.eval_freq > 0 and (timestep + 1) % args.eval_freq == 0:
            results = evaluate_policy(policy, create_env())
            print(
                f"Evaluation at {timestep + 1}: mean_reward={results['mean_reward']:.3f}, "
                f"success_rate={results['success_rate']:.3f}, "
                f"collision_rate={results['collision_rate']:.3f}"
            )
            policy.save(model_path)

    policy.save(model_path)
    print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Name of this run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=30_000)
    parser.add_argument("--start-timesteps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--expl-noise", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Set to 0 to disable evaluation")
    parser.add_argument("--log-freq", type=int, default=1_000)
    parser.add_argument("--output-dir", default="models")
    main(parser.parse_args())
