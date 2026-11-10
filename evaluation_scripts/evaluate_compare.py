"""Side-by-side evaluation of several trained runs.

Evaluates each given ``models/<name>`` checkpoint on the test layouts with
the same seeded start/goal pairs (``evaluation_scripts.evaluation.evaluate_policy`` seeds every
episode by index, so runs are comparable and pairable) and prints one
comparison table with the performance, margin and multiplier metrics.

Multiplier (lambda) columns report the raw value and the reward-scaled
value ``lambda * reward_scale``, so a lambda of 0.1 trained with
``reward_scale=50`` reads as 5.0 in the scaled column.

Usage:
    python evaluation_scripts/evaluate_compare.py run_a run_b run_c
    python evaluation_scripts/evaluate_compare.py run_a run_b --episodes-per-layout 50 --save-json compare.json
    python evaluation_scripts/evaluate_compare.py run_a run_b run_c --save-png compare.png
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")  # must be set before environment imports pyplot
import matplotlib.pyplot as plt

import numpy as np

from evaluation_scripts.evaluate import _json_safe, build_policy, load_config
from environment import create_env
from evaluation_scripts.evaluation import evaluate_policy


def evaluate_run(run_dir, episodes_per_layout, safety_margin=None,
                 max_steps=None, min_start_goal_distance=None):
    """Evaluate one run with its own recorded settings.

    ``safety_margin`` is an evaluation-only runtime clearance threshold: the
    margin metrics use it, and the actor's lidar observations are shortened by
    the excess over the vehicle radius so the policy keeps that margin. It
    applies to every run in the comparison and defaults to the vehicle radius,
    the constraint used in training.

    Returns ``(config, results)`` where ``config`` is the run's config with
    the evaluation-relevant settings resolved to the effective values used.
    """
    config = load_config(run_dir)
    config = dict(config)
    config["layout_set"] = config.get("layout_set") or "one"
    config["normalize_obs"] = config.get("normalize_obs", True)
    config["reward_scale"] = config.get("reward_scale", 1.0)
    # The step cap is the run's own by default; an explicit override makes
    # caps comparable across families (100 vs 150) and shows how much of the
    # timeout share converts into successes with a longer clock.
    if max_steps is None:
        max_steps = config.get("max_steps", 150)
    config["max_steps"] = max_steps
    # The minimum start-goal distance is the run's own by default; an
    # explicit override gives every run the same sampled episode
    # distribution, which also makes the per-episode pairing valid when the
    # recorded values differ.
    if min_start_goal_distance is None:
        min_start_goal_distance = config.get("min_start_goal_distance")
    config["min_start_goal_distance"] = min_start_goal_distance
    layout_set = config["layout_set"]
    normalize_obs = config["normalize_obs"]
    reward_scale = config["reward_scale"]
    max_steps = config["max_steps"]

    env = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        reward_scale=reward_scale,
        max_steps=max_steps,
    )
    if safety_margin is None:
        safety_margin = env.vehicle_radius
    elif safety_margin < env.vehicle_radius:
        raise SystemExit(
            f"--safety-margin must be at least the vehicle radius "
            f"({env.vehicle_radius} m)"
        )
    if safety_margin > env.lidar_range:
        raise SystemExit(
            f"--safety-margin must not exceed the lidar range "
            f"({env.lidar_range} m)"
        )
    config["safety_margin"] = safety_margin
    config["observation_lidar_shift"] = safety_margin - env.vehicle_radius
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
        min_start_goal_distance=min_start_goal_distance,
        safety_margin=safety_margin,
        reward_scale=reward_scale,
        max_steps=max_steps,
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

    # Cost of safety: how long successful episodes take. None (bar omitted)
    # for runs with no successes.
    cost_of_safety = []
    for _, _, results in entries:
        mask = results["successes"]
        cost_of_safety.append(
            float(results["steps"][mask].mean()) if mask.any() else None
        )
    panels.append(("Mean steps on successes (cost of safety)",
                   cost_of_safety, None, "{:.0f}"))

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


def save_layout_figure(path, entries, episodes_per_layout):
    """Render a per-layout difficulty analysis for presentations.

    Panel 1 shows the success rate on every test layout (one line per run).
    Panels 2 and 3 pair layouts across runs and therefore only include runs
    whose layout set and layout count match the first run's.
    """
    names = [_short_name(name) for name, _, _ in entries]
    colors = [plt.cm.tab10(i % 10) for i in range(len(entries))]

    baseline = 0
    base_config = entries[baseline][1]
    base_results = entries[baseline][2]
    base_n_layouts = len(base_results["per_layout_success_rate"])
    pairable = [
        i for i, (_, config, results) in enumerate(entries)
        if config.get("layout_set") == base_config.get("layout_set")
        and len(results["per_layout_success_rate"]) == base_n_layouts
    ]
    layout_label = str(base_config.get("layout_set"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axis = axes[0]
    for i, (_, _, results) in enumerate(entries):
        per_layout = results["per_layout_success_rate"]
        axis.plot(
            np.arange(len(per_layout)), per_layout,
            color=colors[i], label=names[i], linewidth=1.0, alpha=0.8,
        )
    axis.set_title("Success rate by layout", fontsize=13, fontweight="bold")
    axis.set_xlabel("layout index")
    axis.set_ylabel("success rate")
    axis.set_ylim(0, 1)

    axis = axes[1]
    base_success = base_results["per_layout_success_rate"]
    axis.plot([0, 1], [0, 1], "k--", alpha=0.3)
    for i in pairable:
        if i == baseline:
            continue
        axis.scatter(
            base_success, entries[i][2]["per_layout_success_rate"],
            color=colors[i], label=names[i], alpha=0.6,
        )
    axis.set_title("Per layout vs baseline (above diagonal: better)",
                   fontsize=13, fontweight="bold")
    axis.set_xlabel(f"success rate — {names[baseline]}")
    axis.set_ylabel("success rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis = axes[2]
    for i in pairable:
        results = entries[i][2]
        axis.scatter(
            results["per_layout_margin_violation_rate"],
            results["per_layout_success_rate"],
            color=colors[i], label=names[i], alpha=0.6,
        )
    axis.set_title("Per layout: success vs margin violations",
                   fontsize=13, fontweight="bold")
    axis.set_xlabel("margin violation rate")
    axis.set_ylabel("success rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    for axis in axes:
        axis.legend(fontsize=9)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.grid(linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Per-layout analysis — {layout_label} test layouts, "
        f"{episodes_per_layout} episodes per layout",
        fontsize=16,
        fontweight="bold",
    )
    note = ("Each point/line is one layout; rates use "
            f"{episodes_per_layout} episodes, so they are coarse. Panel 1 "
            "shows each run's own test layouts, so layout indices are only "
            "comparable between runs sharing a layout set.")
    if len(pairable) < len(entries):
        excluded = ", ".join(names[i] for i in range(len(entries))
                             if i not in pairable)
        note += (f" Runs with a different layout set or layout count "
                 f"({excluded}) are excluded from the two paired panels.")
    fig.text(0.5, 0.01, note, ha="center", fontsize=10, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def warn_on_setting_mismatches(entries):
    """Warn when runs were trained with different evaluation-affecting settings."""
    for key in ("layout_set", "normalize_obs"):
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


def main(names, episodes_per_layout=20, save_json=None, save_png=None,
         save_layout_png=None, safety_margin=None, max_steps=None,
         min_start_goal_distance=None):
    if len(names) < 2:
        raise SystemExit("Pass at least two run names to compare")

    entries = []
    for name in names:
        run_dir = os.path.join("models", name)
        if not os.path.isdir(run_dir):
            raise SystemExit(f"Run directory not found: {run_dir}")
        print(f"=== Evaluating {name} ===")
        config, results = evaluate_run(
            run_dir, episodes_per_layout, safety_margin=safety_margin,
            max_steps=max_steps,
            min_start_goal_distance=min_start_goal_distance,
        )
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
    if save_layout_png is not None:
        save_layout_figure(save_layout_png, entries, episodes_per_layout)
        print(f"Saved layout figure: {save_layout_png}")


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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override every run's episode step cap for this comparison "
             "(default: each run's config.json max_steps). Raising it makes "
             "the 100- and 150-step families comparable and converts some "
             "timeouts into successes",
    )
    parser.add_argument(
        "--min-start-goal-distance",
        type=float,
        default=None,
        help="Override every run's recorded minimum start-goal distance (m) "
             "for this comparison (default: each run's config.json). Use it "
             "to keep the sampled episode distribution identical, and the "
             "per-episode pairing valid, when the runs were trained with "
             "different values",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=None,
        help="Optional runtime clearance threshold (m from vehicle centre to "
             "obstacle surface). The actor's lidar is shortened by the "
             "excess over the vehicle radius so the policy keeps that margin, "
             "and the margin-violation metrics use the same threshold. "
             "Default: the vehicle radius, i.e. the training constraint and "
             "no observation shift",
    )
    parser.add_argument(
        "--save-layout-png",
        metavar="PATH",
        default=None,
        help="Write a per-layout difficulty analysis figure (success rate "
             "by layout, per layout vs baseline, per-layout success vs "
             "margin violations) to PATH",
    )

    args = parser.parse_args()

    if args.episodes_per_layout < 1:
        parser.error("--episodes-per-layout must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if (args.min_start_goal_distance is not None
            and args.min_start_goal_distance < 0):
        parser.error("--min-start-goal-distance must be nonnegative")

    main(
        args.names,
        args.episodes_per_layout,
        save_json=args.save_json,
        save_png=args.save_png,
        save_layout_png=args.save_layout_png,
        safety_margin=args.safety_margin,
        max_steps=args.max_steps,
        min_start_goal_distance=args.min_start_goal_distance,
    )
