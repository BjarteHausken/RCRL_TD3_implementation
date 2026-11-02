"""Side-by-side evaluation of several trained runs.

Evaluates each given ``models/<name>`` checkpoint on the test layouts with
the same seeded start/goal pairs (``evaluation.evaluate_policy`` seeds every
episode by index, so runs are comparable and pairable) and prints one
comparison table with the performance, margin and multiplier metrics.

Multiplier (lambda) columns report the raw value and the reward-scaled
value ``lambda * reward_scale``, so a lambda of 0.1 trained with
``reward_scale=50`` reads as 5.0 in the scaled column.

Usage:
    python evaluate_compare.py run_a run_b run_c
    python evaluate_compare.py run_a run_b --episodes-per-layout 50 --save-json compare.json
    python evaluate_compare.py run_a run_b run_c --save-png compare.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # must be set before environment imports pyplot
import matplotlib.pyplot as plt

import numpy as np

from evaluate import _json_safe, build_policy, load_config
from environment import create_env
from evaluation import evaluate_policy


def evaluate_run(run_dir, episodes_per_layout):
    """Evaluate one run with its own recorded settings.

    Returns ``(config, results)`` where ``config`` is the run's config with
    the evaluation-relevant settings resolved to the effective values used.
    """
    config = load_config(run_dir)
    config = dict(config)
    config["layout_set"] = config.get("layout_set") or "one"
    config["normalize_obs"] = config.get("normalize_obs", True)
    config["safety_margin"] = config.get("safety_margin", 0.35)
    config["reward_scale"] = config.get("reward_scale", 1.0)
    layout_set = config["layout_set"]
    normalize_obs = config["normalize_obs"]
    safety_margin = config["safety_margin"]
    reward_scale = config["reward_scale"]

    env = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        safety_margin=safety_margin,
        reward_scale=reward_scale,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = build_policy(run_dir, config, state_dim, action_dim, max_action)
    print(f"Loaded model: {os.path.join(run_dir, config.get('agent'))}")

    results = evaluate_policy(
        policy,
        layout_set,
        normalize_obs,
        layout_type="test",
        episodes_per_layout=episodes_per_layout,
        random_reset_margin=config.get("random_reset_margin", 0.3),
        random_reset_max_speed=config.get("random_reset_max_speed", 1.0),
        min_start_goal_distance=config.get("min_start_goal_distance"),
        safety_margin=safety_margin,
        reward_scale=reward_scale,
    )
    return config, results


def _percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def _ci(interval):
    return f"[{interval[0]:.1%}-{interval[1]:.1%}]"


def print_comparison_table(entries):
    """Print the comparison table; each entry is (name, config, results)."""
    header = (
        f"{'run':<28} {'agent':<9} {'success (95% CI)':<22} {'collision':<16} "
        f"{'reward':>7} {'steps':>6} {'margin viol':>11} {'steps in margin':>15} "
        f"{'min clear':>9} {'λ̄':>7} {'λ̄ × scale':>10} {'λ̄ max':>7} {'cap %':>6}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, config, results in entries:
        lambda_stats = results["lambda_stats"]
        if lambda_stats is not None:
            reward_scale = float(config.get("reward_scale", 1.0))
            lambda_mean = f"{lambda_stats['mean']:.3f}"
            lambda_scaled = f"{lambda_stats['mean'] * reward_scale:.1f}"
            lambda_max = f"{lambda_stats['max']:.3f}"
            cap = _percent(lambda_stats.get("cap_fraction"))
        else:
            lambda_mean = lambda_scaled = lambda_max = cap = "—"
        row = (
            f"{name:<28} {config.get('agent', '?'):<9} "
            f"{_percent(results['success_rate'])} {_ci(results['success_rate_ci'])} "
            f"{_percent(results['collision_rate'])} "
            f"{results['mean_reward']:>7.2f} {results['mean_steps']:>6.1f} "
            f"{_percent(results['margin_violation_rate']):>11} "
            f"{results['margin_step_fraction']:>15.2%} "
            f"{results['mean_min_clearance']:>8.3f}m "
            f"{lambda_mean:>7} {lambda_scaled:>10} {lambda_max:>7} {cap:>6}"
        )
        print(row)
    print()
    print("margin viol = share of episodes entering the safety margin at least once;")
    print("min clear = mean over episodes of the closest approach (m);")
    print("λ̄ × scale = λ̄ × reward_scale, making lambdas comparable across runs.")


def _short_name(name, limit=24):
    return name if len(name) <= limit else name[: limit - 1] + "…"


def save_comparison_figure(path, entries, episodes_per_layout):
    """Render the comparison as a bar-chart figure for presentations."""
    layout_label = ", ".join(
        sorted({str(config.get("layout_set")) for _, config, _ in entries})
    )
    names = [_short_name(name) for name, _, _ in entries]
    colors = [plt.cm.tab10(i % 10) for i in range(len(entries))]

    # One panel per key metric. ``values`` entries are None where a run has
    # no value (e.g. plain TD3 has no multiplier), so its bar is omitted.
    panels = []
    for title, key, ci_key, scale, fmt in (
        ("Success rate (95% CI)", "success_rate", "success_rate_ci", 100, "{:.1f}%"),
        ("Collision rate (95% CI)", "collision_rate", "collision_rate_ci", 100, "{:.1f}%"),
        ("Margin violations (95% CI)", "margin_violation_rate",
         "margin_violation_rate_ci", 100, "{:.1f}%"),
        ("Steps in margin", "margin_step_fraction", None, 100, "{:.1f}%"),
    ):
        values = [scale * results[key] for _, _, results in entries]
        if ci_key is None:
            yerr = None
        else:
            below = [scale * (results[key] - results[ci_key][0])
                     for _, _, results in entries]
            above = [scale * (results[ci_key][1] - results[key])
                     for _, _, results in entries]
            yerr = (below, above)
        panels.append((title, values, yerr, fmt))

    clearance = [results["mean_min_clearance"] for _, _, results in entries]
    panels.append(("Mean min clearance (m)", clearance, None, "{:.2f}"))

    lambda_values = []
    for _, config, results in entries:
        stats = results["lambda_stats"]
        if stats is None:
            lambda_values.append(None)
        else:
            lambda_values.append(
                stats["mean"] * float(config.get("reward_scale", 1.0))
            )
    panels.append(("Mean λ × reward scale", lambda_values, None, "{:.1f}"))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for axis, (title, values, yerr, fmt) in zip(axes.flat, panels):
        present = [i for i, value in enumerate(values) if value is not None]
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, fontsize=10, rotation=25, ha="right")
        if not present:
            axis.set_yticks([])
            axis.text(0.5, 0.5, "not available", ha="center", va="center",
                      fontsize=11, color="0.5")
            continue
        axis.bar(
            present,
            [values[i] for i in present],
            color=[colors[i] for i in present],
            yerr=yerr and (
                [yerr[0][i] for i in present],
                [yerr[1][i] for i in present],
            ),
            capsize=4,
        )
        y_max = max(max(values[i] for i in present), 1e-6)
        # Place each label clear of its upper CI whisker, then size the axis
        # so the highest label still fits.
        label_positions = {
            i: values[i] + (yerr[1][i] if yerr else 0.0) + 0.02 * y_max
            for i in present
        }
        for i in present:
            axis.text(
                i, label_positions[i], fmt.format(values[i]),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        axis.set_ylim(0, max(label_positions.values()) * 1.08)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Model comparison — {layout_label} test layouts, "
        f"{episodes_per_layout} episodes per layout",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Whiskers are 95% Wilson score intervals. The lambda panel is scaled "
        "by each run's reward_scale, so a lambda of 0.1 at scale 50 reads 5.0.",
        ha="center", fontsize=10, color="0.3",
    )
    # Reserve room for the suptitle, the footnote, and the rotated row-1
    # tick labels hanging below the top panels.
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), h_pad=6)

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def warn_on_setting_mismatches(entries):
    """Warn when runs were trained with different evaluation-affecting settings."""
    for key in ("layout_set", "normalize_obs", "safety_margin"):
        values = {json.dumps(config.get(key)) for _, config, _ in entries}
        if len(values) > 1:
            print(f"Warning: runs differ in {key!r}; the comparison is not "
                  f"strictly like-for-like ({sorted(values)})")
    for key in ("min_start_goal_distance", "random_reset_margin",
                "random_reset_max_speed"):
        values = {json.dumps(config.get(key)) for _, config, _ in entries}
        if len(values) > 1:
            print(f"Warning: runs differ in {key!r}; the sampled episode "
                  f"distributions differ between runs")


def save_comparison_json(path, entries, episodes_per_layout):
    """Write the combined comparison, including per-episode arrays for pairing."""
    payload = {
        "format": "rcrl-eval-compare-v1",
        "episodes_per_layout": int(episodes_per_layout),
        "layout_type": "test",
        "runs": [
            {
                "run": name,
                "agent": config.get("agent"),
                "config": config,
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
                "lambda": _json_safe(results["lambda_stats"] or {}),
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
            }
            for name, config, results in entries
        ],
    }

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")


def main(names, episodes_per_layout=20, save_json=None, save_png=None):
    if len(names) < 2:
        raise SystemExit("Pass at least two run names to compare")

    entries = []
    for name in names:
        run_dir = os.path.join("models", name)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"Run directory not found: {run_dir}")
        print(f"=== Evaluating {name} ===")
        config, results = evaluate_run(run_dir, episodes_per_layout)
        entries.append((name, config, results))
        num = len(results["rewards"])
        print(
            f"{name}: success={results['successes'].sum()}/{num} "
            f"collision={results['collisions'].sum()}/{num} "
            f"margin_violations={results['margin_violations'].sum()}/{num}\n"
        )

    warn_on_setting_mismatches(entries)
    print_comparison_table(entries)

    if save_json is not None:
        save_comparison_json(save_json, entries, episodes_per_layout)
        print(f"Saved JSON: {save_json}")
    if save_png is not None:
        save_comparison_figure(save_png, entries, episodes_per_layout)
        print(f"Saved figure: {save_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare several trained runs side by side"
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="RUN",
        help="Run directory under models/ to evaluate (pass 2 or more)",
    )
    parser.add_argument(
        "--episodes-per-layout",
        type=int,
        default=20,
        help="Episodes with sampled start/goal pairs per test layout "
             "(default: 20)",
    )
    parser.add_argument(
        "--save-json",
        metavar="PATH",
        default=None,
        help="Write the combined comparison, including per-episode arrays, "
             "to PATH as JSON",
    )
    parser.add_argument(
        "--save-png",
        metavar="PATH",
        default=None,
        help="Write a bar-chart comparison figure (one panel per key "
             "metric) to PATH, for use in presentations",
    )

    args = parser.parse_args()

    if args.episodes_per_layout < 1:
        parser.error("--episodes-per-layout must be positive")

    main(
        args.names,
        args.episodes_per_layout,
        save_json=args.save_json,
        save_png=args.save_png,
    )
