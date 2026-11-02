import argparse
import time
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
from PIL import Image


def main(run_dir, test_layout="random", layout_set="one",
         normalize_obs=True, save_gif=None, gif_max_width=900,
         safety_margin=None):
    if save_gif is not None:
        # Headless capture: no interactive window, frames come from the canvas
        # buffer. This must happen before importing environment (which imports
        # matplotlib.pyplot): once pyplot has selected a GUI backend, switching
        # to Agg can be too late and the window still opens.
        matplotlib.use("Agg")

    from agents.td3 import TD3
    from agents.rcrl_td3 import RCRLTD3
    from environment import create_env
    from evaluation_scripts.evaluate import agent_type_from_config, load_config
    from evaluation_scripts.evaluation import _inflate_lidar

    config = load_config(run_dir)
    agent_type = agent_type_from_config(config, run_dir)
    model_path = os.path.join(run_dir, agent_type)

    env = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        reward_scale=config.get("reward_scale", 1.0),
    )
    if safety_margin is None:
        safety_margin = env.vehicle_radius
    elif not env.vehicle_radius <= safety_margin <= env.lidar_range:
        raise ValueError(
            f"--safety-margin must be between the vehicle radius "
            f"({env.vehicle_radius} m) and the lidar range "
            f"({env.lidar_range} m)"
        )
    lidar_shift = safety_margin - env.vehicle_radius
    if lidar_shift > 0:
        print(
            f"Runtime safety margin: actor lidar shortened by "
            f"{lidar_shift:.3f} m"
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if agent_type == "td3":
        policy = TD3(state_dim, action_dim, max_action)
    else:
        # Agent hyperparameters are a property of the checkpoint, not of the
        # evaluation: reconstruct the agent the run was trained with.
        policy = RCRLTD3(
            state_dim,
            action_dim,
            max_action,
            multiplier_type=config.get("multiplier", "scalar"),
            max_lambda=config.get("max_lambda", 100.0),
            initial_lambda=config.get("initial_lambda", 10.0),
            min_lambda=config.get("min_lambda", 0.0),
        )
    policy.load(model_path)
    print("Loaded model:", model_path)

    if test_layout == "random":
        layout_idx = int(env.np_random.integers(len(env.test_layouts)))
    else:
        layout_idx = test_layout
        if not 0 <= layout_idx < len(env.test_layouts):
            raise ValueError(
                f"Test layout index must be between 0 and {len(env.test_layouts) - 1}"
            )

    # Same randomized regime as training: one sampled feasible start and one
    # sampled feasible goal on the selected layout, seeded by layout index
    # for reproducibility.
    random_reset_margin = config.get("random_reset_margin", 0.3)
    random_reset_max_speed = config.get("random_reset_max_speed", 1.0)
    # Match the sampled start-goal distribution used in training; older
    # checkpoints fall back to the previous two-goal-radius minimum.
    min_start_goal_distance = config.get("min_start_goal_distance")
    if min_start_goal_distance is None:
        min_start_goal_distance = 2 * env.goal_radius
    state, info = env.reset(
        layout_type="test",
        layout_index=layout_idx,
        seed=layout_idx,
        sample_start=True,
        start_margin=random_reset_margin,
        start_max_speed=random_reset_max_speed,
        goal_margin=random_reset_margin,
        min_start_goal_distance=min_start_goal_distance,
    )
    print(f"Test layout: {layout_idx}" + (" (random)" if test_layout == "random" else ""))
    
    episode_reward = 0
    step_count = 0
    frames = []

    while True:
        # IMPORTANT: No exploration noise during evaluation. The optional
        # safety margin is applied as observation-space obstacle inflation.
        action = policy.select_action(_inflate_lidar(state, lidar_shift, env))
        next_state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        state = next_state
        # Render current state, annotated with the episode's running outcome.
        env.render(
            status=(
                f"Layout {layout_idx} | Reward: {episode_reward:.2f} | "
                f"Collision: {'yes' if info['collision'] else 'no'} | "
                f"Goal: {'yes' if info['goal_reached'] else 'no'}"
            )
        )

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
        required=True,
        help="Run directory under models/; its config.json records the agent type"
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
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=None,
        help="Optional runtime clearance threshold (m from vehicle centre to "
             "obstacle surface); the actor's lidar is shortened by the excess "
             "over the vehicle radius. Default: the vehicle radius, i.e. no "
             "observation shift",
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

    run_dir = os.path.join("models", args.name)

    main(
        run_dir, args.test_layout, args.layout_set,
        args.normalize_obs, args.save_gif, args.gif_max_width,
        args.safety_margin,
    )
