import argparse
import json
import os

import numpy as np

from agents.td3 import TD3
from agents.rcrl_td3 import RCRLTD3
from environment import create_env


def main(MODEL_PATH, agent_type="td3", layout_set="one", normalize_obs=True):
    env = create_env(layout_set, normalize_obs=normalize_obs)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if agent_type == "td3":
        policy = TD3(state_dim, action_dim, max_action)
    else:
        # The multiplier type is a property of the checkpoint, not of the
        # evaluation: reconstruct the agent the run was trained with.
        with open(os.path.join(os.path.dirname(MODEL_PATH), "config.json")) as f:
            multiplier_type = json.load(f)["multiplier"]
        policy = RCRLTD3(state_dim, action_dim, max_action, multiplier_type=multiplier_type)
    policy.load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    # Same metrics as the in-training evaluation.
    successes = 0
    collisions = 0
    total_reward = 0.0
    total_steps = 0

    for layout_idx in range(len(env.test_layouts)):
        # Same randomized regime as training: one sampled feasible start
        # and one sampled feasible goal per layout, seeded by layout index.
        env.reset(layout_type="test", layout_index=layout_idx, seed=layout_idx)
        start = env.sample_random_state(margin=0.3, max_speed=1.0)
        goal = env.sample_random_state(margin=0.3, max_speed=0.0)
        while np.hypot(start[0] - goal[0], start[1] - goal[1]) <= 2 * env.goal_radius:
            goal = env.sample_random_state(margin=0.3, max_speed=0.0)
        state, info = env.reset(layout_type="fixed", state=start, goal=goal)

        episode_reward = 0.0
        episode_steps = 0

        while True:
            # IMPORTANT: No exploration noise during evaluation
            action = policy.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                break

        successes += int(info["goal_reached"])
        collisions += int(info["collision"])
        total_reward += episode_reward
        total_steps += episode_steps

        print(
            f"Layout {layout_idx}: "
            f"goal={info['goal_reached']} "
            f"collision={info['collision']} "
            f"reward={episode_reward:.1f} "
            f"steps={episode_steps}"
        )

    num_layouts = len(env.test_layouts)
    print()
    print(
        f"Success rate: {successes}/{num_layouts} | "
        f"Mean reward: {total_reward / num_layouts:.2f} | "
        f"Mean steps: {total_steps / num_layouts:.1f} | "
        f"Collision rate: {collisions / num_layouts:.1%}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a TD3 or RCRL-TD3 agent")

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run directory under models/"
    )
    parser.add_argument(
        "--agent",
        choices=("td3", "rcrl_td3"),
        default="td3",
        help="Agent type and checkpoint format (default: td3)",
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

    args = parser.parse_args()

    MODEL_PATH = os.path.join("models", args.name, args.agent)

    main(MODEL_PATH, args.agent, args.layout_set, args.normalize_obs)

