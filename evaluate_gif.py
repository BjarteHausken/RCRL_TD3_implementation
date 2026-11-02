"""Render one evaluation episode to a gif.

Same episode as ``evaluate_visual.py --save-gif``: one sampled start and goal
on the selected test layout, driven by the deterministic policy. This script
is headless-only and writes the gif to ``gifs/`` by default, with no window
and no evaluation report.
"""

import argparse
import os

import matplotlib

# Must be selected before environment imports matplotlib.pyplot, otherwise a
# GUI backend can be initialized and a window may open.
matplotlib.use("Agg")

from PIL import Image

from agents.td3 import TD3
from agents.rcrl_td3 import RCRLTD3
from environment import create_env
from evaluate import agent_type_from_config, load_config
from evaluation import _inflate_lidar


def render_gif(run_dir, test_layout, layout_set="one", normalize_obs=True,
               output=None, gif_max_width=900, safety_margin=None):
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

    if test_layout == "random":
        layout_idx = int(env.np_random.integers(len(env.test_layouts)))
    else:
        layout_idx = test_layout
        if not 0 <= layout_idx < len(env.test_layouts):
            raise ValueError(
                f"Test layout index must be between 0 and {len(env.test_layouts) - 1}"
            )

    # Same randomized regime as training and as evaluate_visual: one sampled
    # feasible start and one sampled feasible goal, seeded by layout index so
    # the rendered episode is reproducible for a fixed model and layout.
    random_reset_margin = config.get("random_reset_margin", 0.3)
    random_reset_max_speed = config.get("random_reset_max_speed", 1.0)
    # Match the sampled start-goal distribution used in training; older
    # checkpoints fall back to the previous two-goal-radius minimum.
    min_start_goal_distance = config.get("min_start_goal_distance")
    if min_start_goal_distance is None:
        min_start_goal_distance = 2 * env.goal_radius
    state, _ = env.reset(
        layout_type="test",
        layout_index=layout_idx,
        seed=layout_idx,
        sample_start=True,
        start_margin=random_reset_margin,
        start_max_speed=random_reset_max_speed,
        goal_margin=random_reset_margin,
        min_start_goal_distance=min_start_goal_distance,
    )

    if output is None:
        output = os.path.join(
            "gifs", f"{os.path.basename(os.path.normpath(run_dir))}"
            f"_{layout_set}_layout{layout_idx}.gif"
        )
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frames = []
    episode_reward = 0.0
    while True:
        # IMPORTANT: No exploration noise during evaluation. The optional
        # safety margin is applied as observation-space obstacle inflation.
        action = policy.select_action(_inflate_lidar(state, lidar_shift, env))
        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        env.render(
            status=(
                f"Layout {layout_idx} | Reward: {episode_reward:.2f} | "
                f"Collision: {'yes' if info['collision'] else 'no'} | "
                f"Goal: {'yes' if info['goal_reached'] else 'no'}"
            )
        )

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

        if terminated or truncated:
            break

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 * env.dynamics.dt)),
        loop=0,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render one test-layout episode to a gif"
    )

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
        help="Test layout to render: random (default) or a zero-based index",
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
             "checkpoints trained with --no-normalize-obs must be rendered "
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
        "--output",
        metavar="PATH",
        default=None,
        help="Output gif path (default: gifs/<name>_<layout-set>_layout<INDEX>.gif)",
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
        except ValueError:
            parser.error("--test-layout must be 'random' or a zero-based integer index")

    run_dir = os.path.join("models", args.name)

    output = render_gif(
        run_dir,
        args.test_layout,
        args.layout_set,
        args.normalize_obs,
        args.output,
        args.gif_max_width,
        args.safety_margin,
    )
    print(f"Saved gif: {output}")
