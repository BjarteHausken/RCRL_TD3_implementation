import time
import os
import argparse

import matplotlib
from PIL import Image

from agents.td3 import TD3
from agents.rcrl_td3 import RCRL_TD3
from environment import create_env


def main(MODEL_PATH, agent_type="td3", test_layout="random", layout_set="one", obs_type="lidar",
         save_gif=None, gif_max_width=900):
    if save_gif is not None:
        # Headless capture: no interactive window, frames come from the canvas buffer.
        matplotlib.use("Agg")

    env = create_env(layout_set, obs_type)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent_class = {"td3": TD3, "rcrl_td3": RCRL_TD3}[agent_type]
    policy = agent_class(state_dim, action_dim, max_action)
    policy.load(MODEL_PATH)
    print("Loaded model:", MODEL_PATH)

    if test_layout == "random":
        layout_idx = int(env.np_random.integers(len(env.test_layouts)))
    else:
        layout_idx = test_layout
        if not 0 <= layout_idx < len(env.test_layouts):
            raise ValueError(
                f"Test layout index must be between 0 and {len(env.test_layouts) - 1}"
            )

    state, info = env.reset(layout_type="test", layout_index=layout_idx)
    print(f"Test layout: {layout_idx}" + (" (random)" if test_layout == "random" else ""))
    
    episode_reward = 0
    step_count = 0
    frames = []

    while True:
        # IMPORTANT: No exploration noise during evaluation
        action = policy.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        state = next_state
        # Render current state
        env.render()

        if save_gif is not None:
            env.fig.canvas.draw()
            width, height = env.fig.canvas.get_width_height()
            frame = Image.frombuffer(
                "RGBA", (width, height), env.fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1
            )
            frame = frame.convert("RGB")
            if width > gif_max_width:
                # Downscale to bound the gif's size and memory use.
                new_width = gif_max_width
                new_height = max(1, round(height * new_width / width))
                frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
            frames.append(frame)
        else:
            # Slow simulation to approximately real time
            time.sleep(env.dynamics.dt)

        if terminated or truncated:
            break

    if save_gif is not None:
        frames[0].save(
            save_gif,
            save_all=True,
            append_images=frames[1:],
            duration=int(round(1000.0 * env.dynamics.dt)),
            loop=0,
        )
        print(f"Gif saved: {save_gif} ({len(frames)} frames)")

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
    parser.add_argument(
        "--test-layout",
        default="random",
        metavar="random|INDEX",
        help="Test layout to run: random (default) or a zero-based index",
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
    parser.add_argument(
        "--save-gif",
        metavar="PATH",
        default=None,
        help="Save the episode as a looping gif to PATH (runs headless, no window)",
    )
    parser.add_argument(
        "--gif-max-width",
        type=int,
        default=900,
        metavar="PIXELS",
        help="Maximum gif frame width in pixels; wider frames are downscaled (default: 900)",
    )

    args = parser.parse_args()

    if args.test_layout != "random":
        try:
            args.test_layout = int(args.test_layout)
        except ValueError as error:
            parser.error("--test-layout must be 'random' or a zero-based integer index")

    MODEL_PATH = os.path.join("models", args.name, args.agent)

    main(
        MODEL_PATH, args.agent, args.test_layout, args.layout_set, args.obs,
        args.save_gif, args.gif_max_width,
    )
