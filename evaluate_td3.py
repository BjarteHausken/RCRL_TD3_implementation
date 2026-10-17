import os
import argparse

from agents.td3 import TD3
from agents.rcrl_td3 import RCRL_TD3
from environment import create_env


def main(MODEL_PATH, agent_type="td3", layout_set="one", obs_type="lidar"):
    env = create_env(layout_set, obs_type)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent_class = {"td3": TD3, "rcrl_td3": RCRL_TD3}[agent_type]
    policy = agent_class(state_dim, action_dim, max_action)
    policy.load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    # Same metrics as the in-training evaluation.
    successes = 0
    collisions = 0
    total_reward = 0.0
    total_steps = 0

    for layout_idx in range(len(env.test_layouts)):
        state, info = env.reset(layout_type="test", layout_index=layout_idx)

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
        "--obs",
        choices=("lidar", "state"),
        default="lidar",
        help="Observation variant the checkpoint was trained with",
    )

    args = parser.parse_args()

    MODEL_PATH = os.path.join("models", args.name, args.agent)

    main(MODEL_PATH, args.agent, args.layout_set, args.obs)

