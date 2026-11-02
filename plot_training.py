"""Plot training curves from the trainers' structured JSONL logs.

Every trainer writes ``models/<name>/training_log.jsonl``. This script overlays
one or more runs and draws training reward/length, validation success and
collision with confidence bands, and RCRL multiplier/safety diagnostics.

Usage:
    python plot_training.py run_a run_b
    python plot_training.py run_a --smooth 50 --save results/run_a_curves.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot, like the other headless scripts
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from training_log import DEFAULT_LOG_NAME


DEFAULT_SAVE = os.path.join("results", "training_curves.png")
RATE_PANELS = (2, 3, 5)


def load_records(path):
    """Read a run's JSONL log, skipping blank and malformed lines."""
    records = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line {line_number} in {path}")
    return records


def load_config(run_dir):
    """Read a run's config.json without importing the evaluation modules."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_run(name, output_dir, log_name):
    """Load one run's config and records, with clear missing-file errors."""
    run_dir = os.path.join(output_dir, name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")
    log_path = os.path.join(run_dir, log_name)
    if not os.path.exists(log_path):
        raise SystemExit(
            f"Training log not found: {log_path}. This run predates structured "
            f"logging or used a different --log-name; retrain to produce one."
        )
    records = load_records(log_path)
    config = load_config(run_dir)
    header = next(
        (record for record in records if record.get("kind") == "run"), {}
    )
    agent = header.get("agent") or config.get("agent", "?")
    counts = {
        kind: sum(1 for record in records if record.get("kind") == kind)
        for kind in ("episode", "progress", "eval")
    }
    print(
        f"Loaded {name}: agent={agent}, {counts['episode']} episode, "
        f"{counts['progress']} progress, {counts['eval']} eval records"
    )
    return {
        "name": name,
        "agent": agent,
        "config": config,
        "records": records,
    }


def series(records, kind, key):
    """Return sorted step/value arrays for records of ``kind`` with ``key``."""
    points = [
        (record["step"], record[key])
        for record in records
        if record.get("kind") == kind and record.get(key) is not None
    ]
    points.sort(key=lambda point: point[0])
    if not points:
        return np.array([]), np.array([])
    steps, values = zip(*points)
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


def interval_series(records, value_key, low_key, high_key):
    """Return sorted eval steps, values, and confidence bounds."""
    points = [
        (
            record["step"],
            record[value_key],
            record.get(low_key),
            record.get(high_key),
        )
        for record in records
        if record.get("kind") == "eval" and record.get(value_key) is not None
    ]
    points.sort(key=lambda point: point[0])
    if not points:
        empty = np.array([])
        return empty, empty, empty, empty
    steps = np.asarray([point[0] for point in points], dtype=float)
    values = np.asarray([point[1] for point in points], dtype=float)
    lows = np.asarray(
        [point[2] if point[2] is not None else np.nan for point in points],
        dtype=float,
    )
    highs = np.asarray(
        [point[3] if point[3] is not None else np.nan for point in points],
        dtype=float,
    )
    return steps, values, lows, highs


def training_series(records, episode_key, progress_key):
    """Per-episode values when available, else vectorized window means."""
    steps, values = series(records, "episode", episode_key)
    if len(values):
        return steps, values, "episode"
    steps, values = series(records, "progress", progress_key)
    return steps, values, "progress"


def prefer_progress_series(records, key):
    """Periodic values when available, else the in-training eval values."""
    steps, values = series(records, "progress", key)
    if len(values):
        return steps, values
    return series(records, "eval", key)


def rolling_mean(values, window):
    """Trailing mean over ``window`` samples; shorter if the series is short."""
    window = min(max(1, int(window)), len(values))
    if window <= 1 or len(values) == 0:
        return values.copy()
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def _plot_series(axis, steps, values, color, **kwargs):
    """Plot a line, using a marker so a single point stays visible."""
    if len(steps) == 1:
        kwargs.setdefault("marker", "o")
    axis.plot(steps, values, color=color, **kwargs)


def _plot_training_curve(axis, steps, values, source, color, smooth_window):
    """Plot a reward/length curve, smoothing only per-episode data."""
    if source == "episode" and smooth_window > 1 and len(values) > 1:
        axis.plot(steps, values, color=color, alpha=0.25, linewidth=0.8)
        smoothed = rolling_mean(values, smooth_window)
        smoothed_steps = steps[len(steps) - len(smoothed):]
        _plot_series(axis, smoothed_steps, smoothed, color, linewidth=1.8)
    else:
        _plot_series(axis, steps, values, color, linewidth=1.5)


def _plot_interval_curve(axis, steps, values, lows, highs, color):
    """Plot an eval metric with its 95% interval."""
    finite = np.isfinite(lows) & np.isfinite(highs)
    if len(steps) == 1:
        if finite[0]:
            axis.errorbar(
                steps,
                values,
                yerr=np.array(
                    [[values[0] - lows[0]], [highs[0] - values[0]]]
                ),
                fmt="o",
                color=color,
                capsize=4,
            )
        else:
            axis.plot(steps, values, "o", color=color)
        return
    _plot_series(
        axis, steps, values, color, linewidth=1.5, marker="o", markersize=4
    )
    if finite.any():
        axis.fill_between(
            steps[finite],
            lows[finite],
            highs[finite],
            color=color,
            alpha=0.15,
            linewidth=0,
        )


def _mark_missing(axis):
    """Annotate a panel that has no data for any run."""
    axis.set_yticks([])
    axis.text(
        0.5,
        0.5,
        "not available",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="0.5",
    )


def _short_name(name, limit=28):
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _header(run):
    return next(
        (record for record in run["records"] if record.get("kind") == "run"), {}
    )


def warn_on_setting_mismatches(runs):
    """Warn when runs use evaluation or reward settings that break comparability."""
    for key in ("reward_scale", "layout_set"):
        values = set()
        for run in runs:
            value = _header(run).get(key, run["config"].get(key))
            if value is not None:
                values.add(value)
        if len(values) > 1:
            print(
                f"Warning: runs differ in {key!r}; curves are not strictly "
                f"like-for-like ({sorted(values)})"
            )


def plot_training(runs, save_path, smooth_window=1):
    """Render the default training-curve figure for one or more runs."""
    names = [_short_name(run["name"]) for run in runs]
    colors = [plt.cm.tab10(index % 10) for index in range(len(runs))]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    has_data = np.zeros(8, dtype=bool)

    for index, run in enumerate(runs):
        records = run["records"]
        color = colors[index]

        steps, values, source = training_series(
            records, "reward", "mean_reward"
        )
        if len(values):
            has_data[0] = True
            _plot_training_curve(
                axes[0, 0], steps, values, source, color, smooth_window
            )

        steps, values, source = training_series(
            records, "length", "mean_length"
        )
        if len(values):
            has_data[1] = True
            _plot_training_curve(
                axes[0, 1], steps, values, source, color, smooth_window
            )

        steps, values, lows, highs = interval_series(
            records, "success_rate", "success_ci_low", "success_ci_high"
        )
        if len(values):
            has_data[2] = True
            _plot_interval_curve(
                axes[0, 2], steps, values, lows, highs, color
            )

        steps, values, lows, highs = interval_series(
            records, "collision_rate", "collision_ci_low", "collision_ci_high"
        )
        if len(values):
            has_data[3] = True
            _plot_interval_curve(
                axes[0, 3], steps, values, lows, highs, color
            )

        mean_steps, mean_values = prefer_progress_series(
            records, "lambda_mean"
        )
        max_steps, max_values = prefer_progress_series(records, "lambda_max")
        if len(mean_values):
            has_data[4] = True
            _plot_series(
                axes[1, 0], mean_steps, mean_values, color, linewidth=1.5
            )
        if len(max_values):
            has_data[4] = True
            _plot_series(
                axes[1, 0],
                max_steps,
                max_values,
                color,
                linewidth=1.0,
                linestyle="--",
            )

        cap_steps, cap_values = prefer_progress_series(
            records, "lambda_cap_fraction"
        )
        floor_steps, floor_values = prefer_progress_series(
            records, "lambda_floor_fraction"
        )
        if len(cap_values):
            has_data[5] = True
            _plot_series(
                axes[1, 1], cap_steps, cap_values, color, linewidth=1.5
            )
        if len(floor_values):
            has_data[5] = True
            _plot_series(
                axes[1, 1],
                floor_steps,
                floor_values,
                color,
                linewidth=1.0,
                linestyle="--",
            )

        steps, values = series(records, "progress", "mean_h_pred")
        if len(values):
            has_data[6] = True
            _plot_series(axes[1, 2], steps, values, color, linewidth=1.5)

        steps, values = series(records, "eval", "mean_steps")
        if len(values):
            has_data[7] = True
            _plot_series(
                axes[1, 3],
                steps,
                values,
                color,
                linewidth=1.5,
                marker="o",
                markersize=4,
            )

    panel_specs = (
        ("Mean reward", "return"),
        ("Episode length", "steps"),
        ("Success rate (validation, 95% CI)", "rate"),
        ("Collision rate (validation, 95% CI)", "rate"),
        ("Multiplier λ (solid mean, dashed max)", "λ"),
        ("Multiplier at cap / floor", "fraction of λ samples"),
        ("Mean predicted safety value", "mean Q_h(s, π(s))"),
        ("Evaluation mean steps", "steps"),
    )
    for index, axis in enumerate(axes.flat):
        title, ylabel = panel_specs[index]
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_ylabel(ylabel)
        if index in RATE_PANELS:
            axis.set_ylim(0.0, 1.0)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        if index in (1, 7):
            axis.set_ylim(bottom=0.0)
        if index == 6 and has_data[index]:
            axis.axhline(0.0, color="0.5", linestyle=":", linewidth=1.0)
        if index >= 4:
            axis.set_xlabel("environment steps")
        if not has_data[index]:
            _mark_missing(axis)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.grid(linestyle=":", alpha=0.5)

    handles = [
        Line2D([0], [0], color=colors[index], linewidth=2.0, label=names[index])
        for index in range(len(runs))
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(runs), 5),
        bbox_to_anchor=(0.5, 0.925),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Training curves — " + ", ".join(names),
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Reward and length use per-episode values for sequential trainers and "
        "episode-window means for vectorized trainers; --smooth applies a "
        "trailing mean to per-episode series. Success, collision, and mean "
        "steps are validation evaluations; shaded bands are 95% Wilson "
        "intervals. Multiplier and safety-value panels require an RCRL run.",
        ha="center",
        fontsize=10,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90), h_pad=2.5, w_pad=2.0)

    directory = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(names, output_dir="models", log_name=DEFAULT_LOG_NAME,
         save=DEFAULT_SAVE, smooth=1):
    runs = [load_run(name, output_dir, log_name) for name in names]
    warn_on_setting_mismatches(runs)
    plot_training(runs, save, smooth)
    print(f"Saved figure: {save}")
    if smooth > 1:
        print(f"Smoothing window: {smooth} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training curves from the trainers' JSONL logs"
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="RUN",
        help="Run directory under --output-dir to plot",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Root directory containing the runs (default: models)",
    )
    parser.add_argument(
        "--log-name",
        default=DEFAULT_LOG_NAME,
        help="Log filename inside each run directory "
             f"(default: {DEFAULT_LOG_NAME})",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        metavar="EPISODES",
        help="Trailing mean window for per-episode reward/length curves; "
             "0 or 1 disables smoothing (default: 1)",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=DEFAULT_SAVE,
        help=f"Output figure path (default: {DEFAULT_SAVE})",
    )
    args = parser.parse_args()

    if args.smooth < 0:
        parser.error("--smooth must be nonnegative")

    main(
        args.names,
        output_dir=args.output_dir,
        log_name=args.log_name,
        save=args.save,
        smooth=args.smooth,
    )
