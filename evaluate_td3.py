import time
import os
import argparse

from agents.td3 import TD3
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

def main(MODEL_PATH):
    env = create_env()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create TD3 agent
    policy = TD3(state_dim, action_dim, max_action)
    policy.load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    successes = 0

    for layout_idx in range(len(env.test_layouts)):
        env.set_test_layout(layout_idx)
        
        state, info = env.reset(layout_type="fixed")

        while True:
            # IMPORTANT: No exploration noise during evaluation
            action = policy.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            state = next_state

            if terminated or truncated:
                break

        print(
            f"Layout {layout_idx}: "
            f"goal={info['goal_reached']} "
            f"collision={info['collision']}"
        )

        if info["goal_reached"]:
            successes += 1

    print()
    print(
        f"Success rate: "
        f"{successes}/{len(env.test_layouts)}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TD3 on Double Integrator Environment")

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Path to the trained TD3 model"
    )

    args = parser.parse_args()

    MODEL_PATH = os.path.join("models", args.name, "td3")

    main(MODEL_PATH)

