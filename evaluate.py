import argparse
import json
import os

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


def main(run_dir, layout_set="one", normalize_obs=True, episodes_per_layout=20):
    config = load_config(run_dir)
    agent_type = agent_type_from_config(config, run_dir)
    model_path = os.path.join(run_dir, agent_type)

    env = create_env(layout_set, normalize_obs=normalize_obs)

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
    )

    layout_indices = results["layout_indices"]
    num_layouts = len(results["per_layout_success_rate"])
    for layout_idx in range(num_layouts):
        mask = layout_indices == layout_idx
        print(
            f"Layout {layout_idx}: "
            f"success={results['successes'][mask].sum()}/{mask.sum()} "
            f"collision={results['collisions'][mask].sum()}/{mask.sum()} "
            f"reward={results['rewards'][mask].mean():.1f} "
            f"steps={results['steps'][mask].mean():.1f}"
        )

    num_episodes = len(results["rewards"])
    success_ci = results["success_rate_ci"]
    collision_ci = results["collision_rate_ci"]
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

    args = parser.parse_args()

    if args.eval_episodes_per_layout < 1:
        parser.error("--eval-episodes-per-layout must be positive")

    run_dir = os.path.join("models", args.name)

    main(
        run_dir,
        args.layout_set,
        args.normalize_obs,
        args.eval_episodes_per_layout,
    )
