import time
import os
import argparse

from agents.td3 import TD3
from agents.rcrl_td3 import RCRL_TD3
from double_integrator_sim.dynamics import DoubleIntegratorExactDiscretization
from double_integrator_sim.environment import CorridorEnvironment


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

def main(MODEL_PATH, agent_type="td3"):
    env = create_env()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent_class = {"td3": TD3, "rcrl_td3": RCRL_TD3}[agent_type]
    policy = agent_class(state_dim, action_dim, max_action)
    policy.load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    state, info = env.reset(layout_type="test")
    
    episode_reward = 0
    step_count = 0

    while True:
        # IMPORTANT: No exploration noise during evaluation
        action = policy.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        state = next_state
        # Render current state
        env.render()

        # Slow simulation to approximately real time
        time.sleep(env.dynamics.dt)

        if terminated or truncated:
            break

    print()
    print("Episode finished")
    print("----------------")
    print("Steps:", step_count)
    print("Reward:", episode_reward)
    print("Collision:", info["collision"])
    print("Goal reached:", info["goal_reached"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visually evaluate a TD3 or RCRL-TD3 agent")

    parser.add_argument(
        "--name",
        type=str,
        help="Run directory under models/"
    )
    parser.add_argument(
        "--agent",
        choices=("td3", "rcrl_td3"),
        default="td3",
        help="Agent type and checkpoint format (default: td3)",
    )

    args = parser.parse_args()

    MODEL_PATH = os.path.join("models", args.name, args.agent)

    main(MODEL_PATH, args.agent)
