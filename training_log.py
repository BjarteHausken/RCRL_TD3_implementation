"""Structured JSONL training metrics.

Each trainer writes ``models/<name>/training_log.jsonl`` with one JSON object
per line. Every record has a ``kind`` and a ``step`` (environment
transitions), plus whichever metrics are available:

- ``run``: provenance header written once at the start.
- ``episode``: one completed training episode (sequential trainers).
- ``progress``: periodic diagnostics, plus window aggregates for vectorized
  trainers.
- ``eval``: one validation evaluation.

``plot_training.py`` reads these files to draw training curves.
"""

import json
import math
import os

import numpy as np


DEFAULT_LOG_NAME = "training_log.jsonl"


def _json_safe(value):
    """Convert NumPy values to JSON-native ones and non-finite floats to None."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def lambda_metrics(stats):
    """Map ``lambda_stats`` keys to the flat keys used in the JSONL log."""
    if not stats:
        return {}
    mapping = {
        "mean": "lambda_mean",
        "max": "lambda_max",
        "min": "lambda_min",
        "std": "lambda_std",
        "cap_fraction": "lambda_cap_fraction",
        "floor_fraction": "lambda_floor_fraction",
    }
    return {
        output_key: stats[source_key]
        for source_key, output_key in mapping.items()
        if source_key in stats
    }


def evaluation_metrics(results, layout_type="val"):
    """Flatten ``evaluate_policy`` results into scalar log metrics."""
    metrics = {
        "layout_type": layout_type,
        "episodes": int(len(results["rewards"])),
        "mean_reward": results["mean_reward"],
        "mean_steps": results["mean_steps"],
        "success_rate": results["success_rate"],
        "collision_rate": results["collision_rate"],
        "margin_violation_rate": results["margin_violation_rate"],
        "margin_step_fraction": results["margin_step_fraction"],
        "mean_clearance": results["mean_clearance"],
        "mean_min_clearance": results["mean_min_clearance"],
        "safety_margin": results.get("safety_margin"),
        "observation_lidar_shift": results.get("observation_lidar_shift"),
    }
    for prefix, interval_key in (
        ("success", "success_rate_ci"),
        ("collision", "collision_rate_ci"),
        ("margin", "margin_violation_rate_ci"),
    ):
        interval = results.get(interval_key)
        if interval is not None:
            metrics[f"{prefix}_ci_low"] = interval[0]
            metrics[f"{prefix}_ci_high"] = interval[1]
    metrics.update(lambda_metrics(results.get("lambda_stats")))
    return metrics


class TrainingLogger:
    """Write a run's structured metric records as JSONL."""

    def __init__(self, run_dir, filename=DEFAULT_LOG_NAME):
        self.path = os.path.join(run_dir, filename)
        self._file = open(self.path, "w", encoding="utf-8", buffering=1)

    def log(self, kind, step, **metrics):
        """Append one record, omitting metrics that are unavailable."""
        record = {"kind": kind, "step": int(step)}
        for key, value in metrics.items():
            value = _json_safe(value)
            if value is not None:
                record[key] = value
        self._file.write(
            json.dumps(
                record,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            )
            + "\n"
        )
        self._file.flush()

    def log_run(self, config, trainer, num_envs=1):
        """Write the provenance header for this run."""
        self.log(
            "run",
            0,
            agent=config.get("agent"),
            trainer=trainer,
            num_envs=int(num_envs),
            layout_set=config.get("layout_set"),
            reward_scale=config.get("reward_scale"),
            max_timesteps=config.get("max_timesteps"),
        )

    def close(self):
        """Close the log file; safe to call more than once."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
