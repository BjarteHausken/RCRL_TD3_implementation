"""Compare the learned RCRL feasible set with the actual largest feasible set.

The learned feasible set is the sublevel set ``{s : V_h(s) <= 0}`` of the
policy-conditioned reachability value ``V_h(s) = Q_h(s, pi(s))`` of an
RCRL-TD3 checkpoint. The ground truth is the maximal collision-free set (the
viability kernel) computed from the environment's exact double-integrator
dynamics, independently of the trained policy; see
``evaluation_scripts.feasibility``.

Both are evaluated on the same ``(x, y)`` grid for a fixed velocity slice on
each layout. Because the observation contains goal-relative coordinates, the
policy and its safety value depend on the goal even though feasibility does
not, so the metrics are averaged over goals sampled from the training
distribution.

Reported metrics (restricted to collision-free grid positions):

- ``IoU``: intersection over union of the two feasible sets.
- ``false-safe``: share of truly infeasible states the learned value calls
  safe (the dangerous error).
- ``false-unsafe``: share of truly feasible states the learned value calls
  unsafe (the over-cautious error).
- ``true feasible`` / ``learned feasible``: share of free positions each set
  claims.

Usage:
    python evaluation_scripts/evaluate_feasibility.py --name my_rcrl_run
    python evaluation_scripts/evaluate_feasibility.py --name my_rcrl_run \
        --layout-indices 0 1 --slices 0,0 1,0 0,1 --goals-per-layout 8 \
        --save-json results/feasibility.json --save-png results/feasibility.png
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from evaluation_scripts.evaluate import _json_safe, build_policy, load_config
from evaluation_scripts.evaluate_value import (
    INK_MUTED,
    SAFE,
    UNSAFE,
    evaluate_states,
    observation_batch,
)
from evaluation_scripts.feasibility import (
    average_metrics,
    free_positions,
    kernel_slice,
    position_grid,
    set_metrics,
    velocity_grid,
    viability_kernel,
)
from environment import create_env


DEFAULT_SLICES = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
UNSAFE_FILL = "#eceae4"
FALSE_UNSAFE = "#2a78d6"


def parse_slices(values):
    """Parse ``--slices`` entries of the form ``vx,vy``."""
    if values is None:
        return list(DEFAULT_SLICES)
    slices = []
    for text in values:
        parts = text.split(",")
        if len(parts) != 2:
            raise SystemExit(f"--slices entries must be vx,vy (got {text!r})")
        slices.append((float(parts[0]), float(parts[1])))
    return slices


def sample_goals(env, count, margin):
    """Sample ``count`` feasible goal positions on the loaded layout."""
    goals = []
    for _ in range(count):
        state = env.sample_random_state(margin=margin, max_speed=0.0)
        goals.append(np.array([state[0], state[2]], dtype=np.float64))
    return goals


def evaluate_layout(policy, env, layout_type, layout_index, slices, args,
                    goal_margin, seed):
    """Ground truth and learned feasibility maps for one layout."""
    env.reset(layout_type=layout_type, layout_index=layout_index, seed=seed)
    xs, ys, positions = position_grid(
        env.length,
        env.width,
        margin=env.vehicle_radius + 1e-3,
        spacing=args.position_spacing,
    )
    domain = free_positions(env, positions).reshape(len(xs), len(ys))
    lidar = env.lidar_at(positions)

    vxs = velocity_grid(env.dynamics.max_velocity, args.velocity_spacing)
    vys = velocity_grid(env.dynamics.max_velocity, args.velocity_spacing)
    kernel, solver_info = viability_kernel(
        env, xs, ys, vxs, vys,
        action_levels=args.action_levels,
        max_iterations=args.max_iterations,
        device=args.device,
    )
    print(
        f"Layout {layout_index}: kernel over {solver_info['grid_shape']} "
        f"states, {solver_info['iterations']} iterations"
        f"{'' if solver_info['converged'] else ' (not converged)'}"
    )

    goals = sample_goals(env, args.goals_per_layout, goal_margin)
    slice_results = []
    learned_maps = []
    true_maps = []
    for velocity in slices:
        velocity = tuple(float(value) for value in velocity)
        true_map = kernel_slice(kernel, vxs, vys, velocity)
        per_goal = []
        first_learned = None
        for goal in goals:
            observations = observation_batch(
                env, positions, velocity, goal, lidar=lidar
            )
            values, _ = evaluate_states(policy, observations)
            learned_map = values.reshape(len(xs), len(ys)) <= 0.0
            per_goal.append(set_metrics(learned_map, true_map, domain))
            if first_learned is None:
                first_learned = learned_map
        nearest = (
            float(vxs[int(np.argmin(np.abs(vxs - velocity[0])))]),
            float(vys[int(np.argmin(np.abs(vys - velocity[1])))]),
        )
        slice_results.append({
            "velocity": list(velocity),
            "velocity_node": list(nearest),
            "per_goal": _json_safe(per_goal),
            "mean": average_metrics(per_goal),
        })
        learned_maps.append(first_learned)
        true_maps.append(true_map)

    return {
        "layout_type": layout_type,
        "layout_index": int(layout_index),
        "width": float(env.width),
        "pillars": np.asarray(env.pillars, dtype=float).tolist(),
        "solver": solver_info,
        "goals": goals,
        "slices": slice_results,
    }, xs, ys, domain, learned_maps, true_maps


def _draw_layout(ax, env, pillars):
    """Outline the corridor and pillars on a feasibility map."""
    from matplotlib.patches import Circle, Rectangle

    ax.add_patch(Rectangle(
        (0.0, -env.width / 2.0), env.length, env.width,
        fill=False, edgecolor=INK_MUTED, linewidth=1.0,
    ))
    for pillar_x, pillar_y, pillar_radius in pillars:
        ax.add_patch(Circle(
            (pillar_x, pillar_y), pillar_radius,
            fill=False, edgecolor=INK_MUTED, linewidth=0.9,
        ))


def save_figure(path, run_name, layout, xs, ys, domain, env, slices,
                learned_maps, true_maps, goals):
    """Write the learned/true/disagreement maps for one layout and goal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    binary_cmap = ListedColormap([UNSAFE_FILL, SAFE])
    # 0 agree-unsafe, 1 agree-safe, 2 false-safe, 3 false-unsafe.
    disagreement_cmap = ListedColormap(
        [UNSAFE_FILL, SAFE, UNSAFE, FALSE_UNSAFE]
    )
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    rows = len(slices)
    # A narrow label column keeps the per-row metrics horizontal and legible;
    # rotated ylabels cannot fit the long metric line inside a 3-inch row.
    fig, axes = plt.subplots(
        rows, 4, figsize=(14.5, 3.1 * rows), squeeze=False,
        gridspec_kw={"width_ratios": [0.62, 1.0, 1.0, 1.0]},
    )
    for row, (velocity, learned, true, slice_result) in enumerate(
        zip(slices, learned_maps, true_maps, layout["slices"])
    ):
        # Only states an episode can occupy are compared; the critic may
        # predict anything where the vehicle footprint overlaps an obstacle.
        learned = np.asarray(learned, dtype=bool) & domain
        disagreement = np.where(
            learned & true, 1,
            np.where(~learned & ~true, 0,
                     np.where(learned & ~true, 2, 3)),
        )
        mean = slice_result["mean"]
        label = (
            f"v = ({velocity[0]:.1f}, {velocity[1]:.1f}) m/s\n"
            f"IoU {_number(mean['iou'])}\n"
            f"false-safe\n{_rate(mean['false_safe_rate'])}\n"
            f"false-unsafe\n{_rate(mean['false_unsafe_rate'])}"
        )
        label_axis = axes[row, 0]
        label_axis.axis("off")
        label_axis.text(
            0.5, 0.5, label, transform=label_axis.transAxes,
            ha="center", va="center", fontsize=9, linespacing=1.45,
        )
        panels = (
            (learned, binary_cmap, 0, 1),
            (true, binary_cmap, 0, 1),
            (disagreement, disagreement_cmap, -0.5, 3.5),
        )
        for column, (data, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row, column + 1]
            axis.imshow(
                data.T, origin="lower", extent=extent, cmap=cmap,
                vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto",
            )
            _draw_layout(axis, env, layout["pillars"])
            axis.set_xlim(0.0, env.length)
            axis.set_ylim(-env.width / 2.0, env.width / 2.0)
            axis.set_yticks([])
            if row != rows - 1:
                axis.set_xticks([])
            if row == 0:
                axis.set_title(
                    ("Learned feasible set\n"
                     "({V_h <= 0}, averaged metrics)") if column == 0 else
                    ("Actual largest feasible set\n(viability kernel)")
                    if column == 1 else
                    ("Disagreement"),
                    fontsize=11, fontweight="bold",
                )

    handles = [
        Patch(color=SAFE, label="both feasible"),
        Patch(color=UNSAFE_FILL, label="both infeasible"),
        Patch(color=UNSAFE, label="learned feasible, actually infeasible"),
        Patch(color=FALSE_UNSAFE, label="learned infeasible, actually feasible"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.045))
    goal = goals[0]
    fig.suptitle(
        f"{run_name} — {layout['layout_type']} layout "
        f"{layout['layout_index']}: learned vs actual largest feasible set",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, 0.012,
        f"Map is the first sampled goal ({goal[0]:.2f}, {goal[1]:.2f}); "
        f"metrics and totals are averaged over {len(goals)} goals. "
        "Positions where the vehicle overlaps an obstacle are excluded.",
        ha="center", fontsize=9, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93), h_pad=1.6)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _number(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _rate(value):
    return "n/a" if value is None else f"{value:.1%}"


def print_table(slices, layout_results, aggregate):
    """Print the per-slice metrics averaged over layouts and goals."""
    header = (
        f"{'velocity':<14} {'IoU':>7} {'false-safe':>11} {'false-unsafe':>13} "
        f"{'true feasible':>14} {'learned feasible':>17}"
    )
    print()
    print(header)
    print("-" * len(header))
    for index, velocity in enumerate(slices):
        mean = aggregate["slices"][str(index)]
        print(
            f"({velocity[0]:>4.1f},{velocity[1]:>4.1f}) "
            f"{_number(mean['iou']):>7} {_rate(mean['false_safe_rate']):>11} "
            f"{_rate(mean['false_unsafe_rate']):>13} "
            f"{_rate(mean['true_safe_fraction']):>14} "
            f"{_rate(mean['learned_safe_fraction']):>17}"
        )
    mean = aggregate["overall"]
    print("-" * len(header))
    print(
        f"{'overall':<14} {_number(mean['iou']):>7} "
        f"{_rate(mean['false_safe_rate']):>11} "
        f"{_rate(mean['false_unsafe_rate']):>13} "
        f"{_rate(mean['true_safe_fraction']):>14} "
        f"{_rate(mean['learned_safe_fraction']):>17}"
    )
    print()
    print("false-safe = learned feasible but actually infeasible "
          "(dangerous); false-unsafe = learned infeasible but actually "
          "feasible (over-cautious).")
    print(f"{len(layout_results)} layout(s), metrics averaged over sampled "
          f"goals and collision-free grid positions.")


def save_report(path, run_name, config, layout_set, normalize_obs,
                reward_scale, args, slices, layout_results, aggregate):
    """Write the full feasibility report as JSON."""
    payload = {
        "format": "rcrl-feasibility-v1",
        "run": run_name,
        "agent": config.get("agent"),
        "layout_set": layout_set,
        "normalize_obs": bool(normalize_obs),
        "reward_scale": float(reward_scale),
        "solver": {
            "position_spacing": args.position_spacing,
            "velocity_spacing": args.velocity_spacing,
            "action_levels": args.action_levels,
            "max_iterations": args.max_iterations,
            "device": args.device,
            "goals_per_layout": args.goals_per_layout,
            "seed": args.seed,
        },
        "slices": [list(velocity) for velocity in slices],
        "aggregate": aggregate,
        "layouts": layout_results,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, allow_nan=False)
        file.write("\n")


def main(args):
    run_dir = os.path.join(args.output_dir, args.name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")
    config = load_config(run_dir)
    if config.get("agent") != "rcrl_td3":
        raise SystemExit(
            f"{os.path.join(run_dir, 'config.json')} records agent "
            f"{config.get('agent')!r}; only RCRL-TD3 runs learn a safety value"
        )
    layout_set = config.get("layout_set") or "one"
    normalize_obs = config.get("normalize_obs", True)
    reward_scale = config.get("reward_scale", 1.0)
    goal_margin = config.get("random_reset_margin", 0.3)

    env = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        reward_scale=reward_scale,
    )
    # Aligning the velocity grid with one full-braking step keeps the
    # deceleration exactly representable on the grid.
    if args.velocity_spacing is None:
        args.velocity_spacing = (
            env.dynamics.dt * env.max_acceleration
        )
    layouts = {
        "train": env.train_layouts,
        "test": env.test_layouts,
        "val": env.val_layouts,
    }[args.layout_type]
    if args.layout_indices is None:
        layout_indices = list(range(len(layouts)))
    else:
        layout_indices = args.layout_indices
        for index in layout_indices:
            if not 0 <= index < len(layouts):
                raise SystemExit(
                    f"--layout-indices must be in [0, {len(layouts) - 1}] "
                    f"for --layout-type {args.layout_type}"
                )

    slices = parse_slices(args.slices)
    for velocity in slices:
        if np.hypot(*velocity) > env.dynamics.max_velocity + 1e-6:
            raise SystemExit(
                f"--slices velocity {velocity} exceeds the environment speed "
                f"limit ({env.dynamics.max_velocity} m/s)"
            )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = build_policy(run_dir, config, state_dim, action_dim, max_action)
    policy.actor.eval()
    policy.safety_critic.eval()
    print(f"Loaded model: {os.path.join(run_dir, config.get('agent'))}")

    layout_results = []
    figure_data = None
    all_entries = []
    for offset, layout_index in enumerate(layout_indices):
        entry, xs, ys, domain, learned_maps, true_maps = evaluate_layout(
            policy, env, args.layout_type, layout_index, slices, args,
            goal_margin, seed=args.seed + offset,
        )
        layout_results.append(entry)
        for slice_result in entry["slices"]:
            all_entries.append(slice_result["per_goal"])
        if figure_data is None:
            figure_data = (entry, xs, ys, domain, learned_maps, true_maps,
                           entry["goals"])

    # Per-slice aggregate over every (layout, goal) pair, plus an overall row.
    aggregate = {"slices": {}, "overall": average_metrics(
        [metrics for entries in all_entries for metrics in entries]
    )}
    for index in range(len(slices)):
        entries = [
            metrics
            for layout_entry in layout_results
            for metrics in layout_entry["slices"][index]["per_goal"]
        ]
        aggregate["slices"][str(index)] = average_metrics(entries)

    print_table(slices, layout_results, aggregate)

    if args.save_json is not None:
        save_report(
            args.save_json, args.name, config, layout_set, normalize_obs,
            reward_scale, args, slices, layout_results, aggregate,
        )
        print(f"Saved JSON: {args.save_json}")
    if args.save_png is not None:
        entry, xs, ys, domain, learned_maps, true_maps, goals = figure_data
        save_figure(
            args.save_png, args.name, entry, xs, ys, domain, env, slices,
            learned_maps, true_maps, goals,
        )
        print(f"Saved figure: {args.save_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the learned RCRL feasible set with the actual "
                    "largest collision-free set"
    )
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument(
        "--output-dir", default="models", help="Checkpoint root"
    )
    parser.add_argument(
        "--layout-type",
        choices=["train", "test", "val"],
        default="test",
        help="Layout set to evaluate (default: test)",
    )
    parser.add_argument(
        "--layout-indices",
        nargs="+",
        type=int,
        default=None,
        metavar="INDEX",
        help="Layout indices to evaluate (default: all layouts of the type)",
    )
    parser.add_argument(
        "--goals-per-layout",
        type=int,
        default=4,
        help="Goals sampled per layout to average the learned maps over "
             "(default: 4)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--position-spacing",
        type=float,
        default=0.2,
        help="Maximum grid spacing in metres for the position grid "
             "(default: 0.2)",
    )
    parser.add_argument(
        "--velocity-spacing",
        type=float,
        default=None,
        help="Velocity grid spacing in m/s for the viability kernel "
             "(default: one full-braking step, dt * max_acceleration; this "
             "keeps braking exactly representable and avoids rounding the "
             "deceleration away)",
    )
    parser.add_argument(
        "--action-levels",
        type=int,
        default=3,
        help="Levels per action axis used by the kernel: 3 gives "
             "{-a, 0, +a} per axis (default: 3)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=300,
        help="Viability-iteration cap (default: 300)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for the viability iteration (default: auto)",
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        default=None,
        metavar="VX,VY",
        help="Velocity slices to compare, e.g. --slices 0,0 1,0 0,1 "
             "(default: rest plus 1 m/s along each axis)",
    )
    parser.add_argument(
        "--save-json",
        metavar="PATH",
        default=None,
        help="Write the full feasibility report to PATH as JSON",
    )
    parser.add_argument(
        "--save-png",
        metavar="PATH",
        default=None,
        help="Write the learned/true/disagreement maps to PATH as a figure",
    )
    args = parser.parse_args()

    if args.position_spacing <= 0:
        parser.error("--position-spacing must be positive")
    if args.velocity_spacing is not None and args.velocity_spacing <= 0:
        parser.error("--velocity-spacing must be positive")
    if args.action_levels < 2:
        parser.error("--action-levels must be at least 2")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.goals_per_layout < 1:
        parser.error("--goals-per-layout must be positive")

    main(args)
