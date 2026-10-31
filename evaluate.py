import argparse
import json
import os

import numpy as np

from agents.td3 import TD3
from agents.rcrl_td3 import RCRLTD3
from environment import create_env
from evaluation import evaluate_policy


def load_config(run_dir):
    """Read a run's ``config.json``; empty when the run has none."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as file:
        return json.load(file)


def agent_type_from_config(config, run_dir):
    """Return the agent type recorded in a run's ``config.json``."""
    agent_type = config.get("agent")
    if agent_type not in ("td3", "rcrl_td3"):
        raise ValueError(
            f"{os.path.join(run_dir, 'config.json')} does not record a valid "
            f"agent type (got {agent_type!r}); expected 'td3' or 'rcrl_td3'"
        )
    return agent_type


def _json_safe(value):
    """Convert NumPy arrays/scalars in an evaluation result to JSON types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_evaluation_json(path, results, config, run_dir, layout_set,
                         normalize_obs, safety_margin, reward_scale,
                         min_start_goal_distance):
    """Write an evaluation report, including per-episode arrays, as JSON.

    The per-episode arrays use the same seeded start/goal pairs across runs,
    so reports from different checkpoints can be joined by
    ``(layout_indices, episode index)`` for paired comparisons.
    """
    payload = {
        "format": "rcrl-eval-v1",
        "run": os.path.basename(os.path.normpath(run_dir)),
        "run_dir": run_dir,
        "agent": config.get("agent"),
        "layout_set": layout_set,
        "layout_type": "test",
        "normalize_obs": bool(normalize_obs),
        "episodes_per_layout": int(results["episodes_per_layout"]),
        "safety_margin": float(safety_margin),
        "reward_scale": float(reward_scale),
        "random_reset_margin": config.get("random_reset_margin", 0.3),
        "random_reset_max_speed": config.get("random_reset_max_speed", 1.0),
        "min_start_goal_distance": float(min_start_goal_distance),
        "metrics": _json_safe({
            key: results[key]
            for key in (
                "mean_reward",
                "mean_steps",
                "success_rate",
                "collision_rate",
                "success_rate_ci",
                "collision_rate_ci",
                "margin_violation_rate",
                "margin_violation_rate_ci",
                "margin_step_fraction",
                "mean_clearance",
                "mean_min_clearance",
            )
        }),
        "per_layout": _json_safe({
            "success_rate": results["per_layout_success_rate"],
            "collision_rate": results["per_layout_collision_rate"],
            "margin_violation_rate": results["per_layout_margin_violation_rate"],
            "mean_min_clearance": results["per_layout_min_clearance"],
        }),
        "episodes": _json_safe({
            "layout_indices": results["layout_indices"],
            "rewards": results["rewards"],
            "steps": results["steps"],
            "successes": results["successes"],
            "collisions": results["collisions"],
            "margin_violations": results["margin_violations"],
            "margin_steps": results["margin_steps"],
            "margin_step_fraction": results["margin_step_fraction_per_episode"],
            "mean_clearance": results["mean_clearance_per_episode"],
            "max_h": results["max_h"],
            "min_clearance": results["min_clearance"],
        }),
        "config": config,
    }

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        # allow_nan=False keeps the file valid JSON; every episode takes at
        # least one step, so the recorded arrays are finite.
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")


def main(run_dir, layout_set="one", normalize_obs=True, episodes_per_layout=20,
         safety_margin=None, reward_scale=None, min_start_goal_distance=None,
         save_json=None):
    config = load_config(run_dir)
    agent_type = agent_type_from_config(config, run_dir)
    model_path = os.path.join(run_dir, agent_type)

    # The constraint margin and reward divisor are properties of the run.
    # Checkpoints trained before these were recorded fall back to the
    # environment default margin and unscaled rewards.
    if safety_margin is None:
        safety_margin = config.get("safety_margin", 0.35)
    if reward_scale is None:
        reward_scale = config.get("reward_scale", 1.0)

    env = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        safety_margin=safety_margin,
        reward_scale=reward_scale,
    )

    # The sampled episode distribution is a property of the run: use the
    # distance recorded at training time, falling back to two goal radii for
    # checkpoints written before the field existed.
    if min_start_goal_distance is None:
        min_start_goal_distance = config.get("min_start_goal_distance")
    if min_start_goal_distance is None:
        min_start_goal_distance = 2.0 * env.goal_radius

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

    # Same randomized regime and metrics as the in-training evaluation; all
    # test layouts are stepped in parallel, with several sampled start/goal
    # pairs per layout.
    results = evaluate_policy(
        policy,
        layout_set,
        normalize_obs,
        layout_type="test",
        episodes_per_layout=episodes_per_layout,
        random_reset_margin=config.get("random_reset_margin", 0.3),
        random_reset_max_speed=config.get("random_reset_max_speed", 1.0),
        min_start_goal_distance=min_start_goal_distance,
        safety_margin=safety_margin,
        reward_scale=reward_scale,
    )

    layout_indices = results["layout_indices"]
    num_layouts = len(results["per_layout_success_rate"])
    for layout_idx in range(num_layouts):
        mask = layout_indices == layout_idx
        print(
            f"Layout {layout_idx}: "
            f"success={results['successes'][mask].sum()}/{mask.sum()} "
            f"collision={results['collisions'][mask].sum()}/{mask.sum()} "
            f"margin_violations={results['margin_violations'][mask].sum()}/{mask.sum()} "
            f"reward={results['rewards'][mask].mean():.2f} "
            f"steps={results['steps'][mask].mean():.1f} "
            f"min_clearance={results['min_clearance'][mask].mean():.3f} m"
        )

    num_episodes = len(results["rewards"])
    success_ci = results["success_rate_ci"]
    collision_ci = results["collision_rate_ci"]
    margin_ci = results["margin_violation_rate_ci"]
    print()
    print(
        f"Success rate: {results['successes'].sum()}/{num_episodes} "
        f"({results['success_rate']:.1%}, 95% CI "
        f"{success_ci[0]:.1%}-{success_ci[1]:.1%}) | "
        f"Mean reward: {results['mean_reward']:.2f} | "
        f"Mean steps: {results['mean_steps']:.1f} | "
        f"Collision rate: {results['collision_rate']:.1%} "
        f"(95% CI {collision_ci[0]:.1%}-{collision_ci[1]:.1%})"
    )
    print(
        f"Margin (h > 0) violation rate: "
        f"{results['margin_violations'].sum()}/{num_episodes} "
        f"({results['margin_violation_rate']:.1%}, 95% CI "
        f"{margin_ci[0]:.1%}-{margin_ci[1]:.1%}) | "
        f"Steps in margin: {results['margin_step_fraction']:.2%} | "
        f"Mean min clearance: {results['mean_min_clearance']:.3f} m | "
        f"Mean clearance: {results['mean_clearance']:.3f} m"
    )

    if save_json is not None:
        save_evaluation_json(
            save_json,
            results,
            config,
            run_dir,
            layout_set,
            normalize_obs,
            safety_margin,
            reward_scale,
            min_start_goal_distance,
        )
        print(f"Saved JSON: {save_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a TD3 or RCRL-TD3 agent")

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run directory under models/; its config.json records the agent type"
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
        "--eval-episodes-per-layout",
        type=int,
        default=20,
        help="Episodes with sampled start/goal pairs to run on each test "
             "layout (default: 20)",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=None,
        help="Override the safety margin recorded in config.json for the "
             "margin-violation metrics",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=None,
        help="Override the reward divisor recorded in config.json",
    )
    parser.add_argument(
        "--min-start-goal-distance",
        type=float,
        default=None,
        help="Override the minimum start-goal distance (m) recorded in "
             "config.json",
    )
    parser.add_argument(
        "--save-json",
        metavar="PATH",
        default=None,
        help="Write the evaluation report, including per-episode arrays, to "
             "PATH as JSON",
    )

    args = parser.parse_args()

    if args.eval_episodes_per_layout < 1:
        parser.error("--eval-episodes-per-layout must be positive")
    if args.safety_margin is not None and args.safety_margin <= 0:
        parser.error("--safety-margin must be positive")
    if args.reward_scale is not None and args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if (args.min_start_goal_distance is not None
            and args.min_start_goal_distance < 0):
        parser.error("--min-start-goal-distance must be nonnegative")

    run_dir = os.path.join("models", args.name)

    main(
        run_dir,
        args.layout_set,
        args.normalize_obs,
        args.eval_episodes_per_layout,
        safety_margin=args.safety_margin,
        reward_scale=args.reward_scale,
        min_start_goal_distance=args.min_start_goal_distance,
        save_json=args.save_json,
    )
