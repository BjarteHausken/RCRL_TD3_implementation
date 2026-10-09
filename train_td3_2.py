import os
import random
import argparse
import json

import numpy as np
import torch

from agents.td3 import TD3
from utils import ReplayBuffer

from double_integrator_sim.dynamics import (
    DoubleIntegratorExactDiscretization
)
from double_integrator_sim.environment import (
    CorridorEnvironment
)


# ============================================================
# Configuration
# ============================================================

SEED = 0

MAX_TIMESTEPS = 30_000

# Use completely random actions before training TD3
START_TIMESTEPS = 10_000

BATCH_SIZE = 256

# Exploration noise added to actor actions during training
EXPL_NOISE = 0.1

# Evaluate the deterministic policy this often
EVAL_FREQ = 5_000
EVAL_EPISODES = 10

BASE_SAVE_DIR = "models"


# ============================================================
# Environment
# ============================================================

def create_env():
    dynamics = DoubleIntegratorExactDiscretization(
        dt=0.1,
        max_velocity=3.0
    )

    env = CorridorEnvironment(
        dynamics=dynamics,
        length=30.0,
        width=5.0,
        max_acceleration=2.0,
        lidar_range=5.0,
        num_lidar_rays=64,
    )

    return env


# ============================================================
# Evaluation
# ============================================================

def evaluate_policy(policy, episodes=10):
    """
    Evaluate the deterministic TD3 policy without exploration noise.
    """

    eval_env = create_env()

    total_reward = 0.0
    total_steps = 0

    successes = 0
    collisions = 0
    
    num_layouts = len(eval_env.val_layouts)

    for layout_idx in range(num_layouts):
        eval_env.set_val_layout(layout_idx)

        state, info = eval_env.reset(layout_type="fixed")

        episode_reward = 0.0
        episode_steps = 0

        while True:

            # Deterministic action, no exploration noise
            action = policy.select_action(state)

            next_state, reward, terminated, truncated, info = eval_env.step(action)

            episode_reward += reward
            episode_steps += 1

            state = next_state

            if terminated or truncated:
                break

        total_reward += episode_reward
        total_steps += episode_steps

        if info["goal_reached"]:
            successes += 1

        if info["collision"]:
            collisions += 1

    mean_reward = total_reward /num_layouts 
    mean_steps = total_steps / num_layouts

    success_rate = successes / num_layouts
    collision_rate = collisions / num_layouts

    return {
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
    }


# ============================================================
# Main training
# ============================================================

def main(run_name):

    # --------------------------------------------------------
    # Random seeds
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = create_env()

    env.action_space.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Your normalized action space is [-1, 1]
    max_action = float(env.action_space.high[0])

    print(f"State dimension:  {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action:       {max_action}")
    print()

    # --------------------------------------------------------
    # TD3
    # --------------------------------------------------------

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    )

    # --------------------------------------------------------
    # Replay buffer
    # --------------------------------------------------------

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=int(1e6),
    )

    # --------------------------------------------------------
    # Initial episode
    # --------------------------------------------------------

    state, info = env.reset(layout_type="train", seed=SEED)

    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0

    # --------------------------------------------------------
    # Model directory
    # --------------------------------------------------------
    
    run_dir = os.path.join(BASE_SAVE_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "td3")

    config = {
            "run_name": run_name,
            "seed": SEED,
            "max_timesteps": MAX_TIMESTEPS,
            "start_timesteps": START_TIMESTEPS,
            "batch_size": BATCH_SIZE,
            "expl_noise": EXPL_NOISE,
            "eval_freq": EVAL_FREQ,
            "eval_episodes": EVAL_EPISODES,

            "dt": 0.1,
            "max_velocity": 3.0,
            "corridor_length": 30.0,
            "corridor_width": 5.0,
            "lidar_range": 5.0,
            "num_lidar_rays": 64,

            "discount": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f: json.dump(config, f, indent=4)

    # ========================================================
    # Training loop
    # ========================================================

    for t in range(MAX_TIMESTEPS):

        episode_steps += 1

        # ====================================================
        # Select action
        # ====================================================

        if t < START_TIMESTEPS:

            # Pure random exploration before TD3 training starts
            action = env.action_space.sample()

        else:

            # Actor action
            action = policy.select_action(state)

            # Add exploration noise
            noise = np.random.normal(
                loc=0.0,
                scale=max_action * EXPL_NOISE,
                size=action_dim
            )

            action = action + noise

            action = np.clip(
                action,
                -max_action,
                max_action
            )

        # ====================================================
        # Step environment
        # ====================================================

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        episode_reward += reward

        # ====================================================
        # Store transition
        # ====================================================

        # Only true terminal states stop TD bootstrapping.
        #
        # terminated:
        #     collision or goal
        #
        # truncated:
        #     artificial episode time limit
        #
        # We therefore store terminated rather than
        # terminated OR truncated in the replay buffer.

        done_for_buffer = float(terminated)

        replay_buffer.add(
            state,
            action,
            next_state,
            reward,
            done_for_buffer
        )

        state = next_state

        # ====================================================
        # Train TD3
        # ====================================================

        if (
            t >= START_TIMESTEPS
            and replay_buffer.size >= BATCH_SIZE
        ):
            policy.train(
                replay_buffer,
                batch_size=BATCH_SIZE
            )

        # ====================================================
        # Episode finished
        # ====================================================

        if terminated or truncated:

            print(
                f"Episode {episode_num:4d} | "
                f"Timestep {t + 1:7d} | "
                f"Steps {episode_steps:4d} | "
                f"Reward {episode_reward:8.3f} | "
                f"Collision {info['collision']} | "
                f"Goal {info['goal_reached']}"
            )

            state, info = env.reset(layout_type="train")

            episode_reward = 0.0
            episode_steps = 0
            episode_num += 1

        # ====================================================
        # Evaluation
        # ====================================================

        if (t + 1) % EVAL_FREQ == 0:

            results = evaluate_policy(
                policy,
                episodes=EVAL_EPISODES
            )

            print()
            print("=" * 60)
            print(
                f"Evaluation after {t + 1} timesteps"
            )
            print(
                f"Mean reward:    "
                f"{results['mean_reward']:.3f}"
            )
            print(
                f"Mean steps:     "
                f"{results['mean_steps']:.1f}"
            )
            print(
                f"Success rate:   "
                f"{100 * results['success_rate']:.1f}%"
            )
            print(
                f"Collision rate: "
                f"{100 * results['collision_rate']:.1f}%"
            )
            print("=" * 60)
            print()

            # Save current model

            policy.save(model_path)

    # ========================================================
    # Final save
    # ========================================================

    policy.save(model_path)

    print()
    print("Training complete.")
    print(f"Model saved to: {model_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the training run"
    )

    args = parser.parse_args()

    main(args.name)
