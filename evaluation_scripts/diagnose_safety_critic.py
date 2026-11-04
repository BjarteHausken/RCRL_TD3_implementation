"""Offline convergence test for the RCRL safety critic.

The online trainer updates the single safety critic once per reward-critic
step on the same minibatch. This script asks whether that critic would fit
its own Bellman backup much better given more optimization steps, which is
the simplest test of the "the safety critic is undertrained" hypothesis.

Procedure:

1. Load an RCRL-TD3 checkpoint and collect a fixed dataset by rolling out the
   frozen actor with the environment's training reset distribution.
2. Freeze one Bellman target per transition, computed once from the
   checkpoint's target networks (the same formula as
   ``RCRLTD3.safety_target``)::

       continuation = where(not_done, V_target(s', pi_target(s')), h(s'))
       y            = (1 - discount) * h + discount * max(h, continuation)

3. Split by episode (default 80/20) so adjacent states cannot leak between
   the train and held-out sets.
4. Continue training only a copy of the safety critic on the fixed targets
   with the run's safety learning rate, and report held-out error at step 0
   and every ``--eval-every`` steps.
5. Optionally repeat with a freshly initialized critic (``--from-scratch``)
   to separate "not enough optimization" from "not enough capacity".

Metrics are reported overall and per target bin, for terminal subgroups
(collision, goal, truncation), and as sign errors against the target. The
collision-terminal subgroup is the important one: those are the transitions
that teach the critic that crossing the boundary is unsafe. A Monte-Carlo
return computed from the realized constraints is also reported so predictions
can be checked against an outcome-based label, not only against the critic's
own backup.

Caveat: ``RCRLTD3.load`` rebuilds both target networks as deep copies of the
online networks, so the frozen target here is the checkpoint's online critic
rather than the final Polyak average. At ``tau=0.005`` the two are close but
not identical.

Usage::

    python evaluation_scripts/diagnose_safety_critic.py \
        --name rcrl_scalar_l5 --num-transitions 500000 --steps 100000 \
        --from-scratch --save-json results/diagnostics/safety_l5.json
"""

import argparse
import copy
import json
import os
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from agents.rcrl_td3 import SafetyCritic, SafetyCriticTwin
from environment import create_env, step_batch
from evaluation_scripts.evaluate import build_policy, load_config

BIN_EDGES = (-np.inf, -0.2, 0.0, np.inf)
BIN_NAMES = ("deep_safe", "near_boundary", "positive")
NEAR_BOUNDARY_MSE_KEY = "near_boundary"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test whether an RCRL safety critic is undertrained by "
                    "continuing it offline on fixed Bellman targets"
    )
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument("--output-dir", default="models", help="Checkpoint root")
    parser.add_argument(
        "--layout-type",
        choices=["train", "test", "val"],
        default="train",
        help="Layout set used to collect the dataset (default: train)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Parallel environments used for collection (default: 32)",
    )
    parser.add_argument(
        "--num-transitions",
        type=int,
        default=500_000,
        help="Approximate number of transitions to collect (default: 500000)",
    )
    parser.add_argument(
        "--split-by",
        choices=["episode", "transition"],
        default="episode",
        help="Hold-out split unit; episode avoids adjacent-state leakage "
             "(default: episode)",
    )
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=0.8,
        help="Fraction of episodes (or transitions) used for training "
             "(default: 0.8)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Offline safety-critic gradient steps (default: 100000)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10_000,
        help="Evaluate the held-out set every this many steps; 0 disables "
             "intermediate evaluations (default: 10000)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Safety-critic learning rate (default: the run's safety_lr)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Also train a freshly initialized safety critic on the same "
             "fixed targets",
    )
    parser.add_argument(
        "--expl-noise",
        type=float,
        default=0.0,
        help="Action noise scale used during collection, as a fraction of "
             "max_action; 0 uses the deterministic actor (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-json",
        metavar="PATH",
        default=None,
        help="Write the full diagnostic report to PATH as JSON",
    )
    parser.add_argument(
        "--save-png",
        metavar="PATH",
        default=None,
        help="Write held-out error curves to PATH as a figure",
    )
    parser.add_argument(
        "--save-checkpoint",
        metavar="RUN_DIR",
        default=None,
        help="Write a loadable run directory containing the checkpoint with "
             "its continued safety critic (refuses to overwrite the source)",
    )
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.num_transitions < 1:
        parser.error("--num-transitions must be positive")
    if not 0.0 < args.split_fraction < 1.0:
        parser.error("--split-fraction must be between 0 and 1")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.eval_every < 0:
        parser.error("--eval-every must be nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0.0 <= args.expl_noise <= 1.0:
        parser.error("--expl-noise must be between 0 and 1")
    return args


def resolve_run(run_dir, args):
    """Load the checkpoint and the run settings that shape the dataset."""
    config = load_config(run_dir)
    if config.get("agent") != "rcrl_td3":
        raise SystemExit(
            f"{os.path.join(run_dir, 'config.json')} records agent "
            f"{config.get('agent')!r}; only RCRL-TD3 runs have a safety critic"
        )
    layout_set = config.get("layout_set") or "one"
    normalize_obs = config.get("normalize_obs", True)
    reward_scale = config.get("reward_scale", 1.0)
    env = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        reward_scale=reward_scale,
        max_steps=config.get("max_steps", 150),
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy = build_policy(run_dir, config, state_dim, action_dim, max_action)
    policy.actor.eval()
    policy.safety_critic.eval()

    min_start_goal_distance = config.get("min_start_goal_distance")
    if min_start_goal_distance is None:
        min_start_goal_distance = 2.0 * env.goal_radius
    settings = {
        "layout_set": layout_set,
        "normalize_obs": normalize_obs,
        "reward_scale": reward_scale,
        "max_steps": config.get("max_steps", 150),
        "random_reset_margin": config.get("random_reset_margin", 0.3),
        "random_reset_max_speed": config.get("random_reset_max_speed", 1.0),
        "min_start_goal_distance": float(min_start_goal_distance),
        "discount": config.get("discount", 0.99),
        "learning_rate": (
            args.learning_rate
            if args.learning_rate is not None
            else config.get("safety_lr", 1e-4)
        ),
        "grad_clip": config.get("grad_clip", 1.0),
    }
    return config, env, policy, settings


def reset_episode(env, layout_type, num_layouts, episode_id, seed, settings):
    """Reset into a new sampled episode on a rotating layout."""
    return env.reset(
        layout_type=layout_type,
        layout_index=int(episode_id % num_layouts),
        seed=int(seed + episode_id),
        sample_start=True,
        start_margin=settings["random_reset_margin"],
        start_max_speed=settings["random_reset_max_speed"],
        goal_margin=settings["random_reset_margin"],
        min_start_goal_distance=settings["min_start_goal_distance"],
    )[0]


def collect_dataset(policy, env, settings, args, device):
    """Roll out the frozen actor and freeze one Bellman target per transition."""
    layout_sets = {
        "train": env.train_layouts,
        "test": env.test_layouts,
        "val": env.val_layouts,
    }
    num_layouts = len(layout_sets[args.layout_type])
    state_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])
    rng = np.random.default_rng(args.seed)

    envs = [
        create_env(
            settings["layout_set"],
            normalize_obs=settings["normalize_obs"],
            reward_scale=settings["reward_scale"],
            max_steps=settings["max_steps"],
        )
        for _ in range(args.num_envs)
    ]
    observations = np.empty((args.num_envs, state_dim), dtype=np.float32)
    episode_ids = np.arange(args.num_envs, dtype=np.int64)
    step_ids = np.zeros(args.num_envs, dtype=np.int64)
    for index, env_i in enumerate(envs):
        observations[index] = reset_episode(
            env_i,
            args.layout_type,
            num_layouts,
            episode_ids[index],
            args.seed,
            settings,
        )
    next_episode = args.num_envs

    chunks = {
        name: []
        for name in (
            "states",
            "actions",
            "targets",
            "h",
            "next_h",
            "not_done",
            "episode_ids",
            "step_ids",
            "collision",
            "goal",
            "truncated",
        )
    }
    collected = 0
    while collected < args.num_transitions:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            actions = policy.actor(obs_tensor).cpu().numpy()
        if args.expl_noise > 0.0:
            actions = np.clip(
                actions
                + rng.normal(
                    loc=0.0,
                    scale=max_action * args.expl_noise,
                    size=actions.shape,
                ),
                -max_action,
                max_action,
            )

        next_states, _, terminated, truncated, info = step_batch(envs, actions)
        h = np.asarray(info["constraint"], dtype=np.float32)
        next_h = np.asarray(info["next_constraint"], dtype=np.float32)
        # Time-limit truncations bootstrap, matching the trainers' use of
        # terminated (not truncated) as the terminal flag.
        not_done = 1.0 - np.asarray(terminated, dtype=np.float32)

        with torch.no_grad():
            targets = policy.safety_target(
                torch.as_tensor(
                    next_states, dtype=torch.float32, device=device
                ),
                torch.as_tensor(h, device=device).unsqueeze(-1),
                torch.as_tensor(next_h, device=device).unsqueeze(-1),
                torch.as_tensor(not_done, device=device).unsqueeze(-1),
            ).squeeze(-1).cpu().numpy()

        chunks["states"].append(observations.copy())
        chunks["actions"].append(actions.astype(np.float32))
        chunks["targets"].append(targets.astype(np.float32))
        chunks["h"].append(h)
        chunks["next_h"].append(next_h)
        chunks["not_done"].append(not_done.astype(np.float32))
        chunks["episode_ids"].append(episode_ids.copy())
        chunks["step_ids"].append(step_ids.copy())
        chunks["collision"].append(
            np.asarray(terminated & info["collision"], dtype=bool)
        )
        chunks["goal"].append(
            np.asarray(terminated & info["goal_reached"], dtype=bool)
        )
        chunks["truncated"].append(np.asarray(truncated, dtype=bool))
        collected += args.num_envs

        # Advance every environment, then reset the ones that finished.
        observations = next_states.astype(np.float32)
        step_ids += 1
        finished = np.flatnonzero(terminated | truncated)
        for index in finished:
            episode_ids[index] = next_episode
            step_ids[index] = 0
            observations[index] = reset_episode(
                envs[index],
                args.layout_type,
                num_layouts,
                next_episode,
                args.seed,
                settings,
            )
            next_episode += 1

    data = {name: np.concatenate(values) for name, values in chunks.items()}
    data["num_episodes"] = int(next_episode)
    return data


def monte_carlo_targets(data, discount):
    """Realized RCRL recursion value for every transition.

    For each episode the constraint sequence is ``h_0 ... h_{T-1}`` followed
    by the terminal ``h_T`` (the last stored ``next_h``). The value recursion

        V_t = (1 - discount) * h_t + discount * max(h_t, V_{t+1}),
        V_T = h_T

    mirrors the Bellman target but replaces the critic bootstrap with the
    realized continuation. Episodes that were truncated at the time limit are
    excluded from the calibration metrics.
    """
    order = np.lexsort((data["step_ids"], data["episode_ids"]))
    episode_ids = data["episode_ids"][order]
    h = data["h"][order]
    next_h = data["next_h"][order]
    truncated = data["truncated"][order]
    values = np.full(len(order), np.nan, dtype=np.float32)
    available = np.zeros(len(order), dtype=bool)

    if len(order) == 0:
        return values, available

    boundaries = np.flatnonzero(
        np.diff(episode_ids, prepend=episode_ids[0] - 1)
    )
    starts = boundaries
    ends = np.append(starts[1:], len(order))
    for start, end in zip(starts, ends):
        index = order[start:end]
        value = float(next_h[start + (end - start) - 1])
        episode_values = np.empty(end - start, dtype=np.float32)
        for offset in range(end - start - 1, -1, -1):
            current_h = float(h[start + offset])
            value = (1.0 - discount) * current_h + discount * max(
                current_h, value
            )
            episode_values[offset] = value
        values[index] = episode_values
        # The last transition's truncated flag says whether the episode hit
        # the time limit. Collisions and goals set terminated, so they are
        # valid realized returns.
        available[index] = not bool(truncated[end - 1])

    return values, available


def subset_metrics(pred, target, mask):
    """Error statistics on a boolean subset, with ``None`` for an empty set."""
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "count": 0,
            "mse": None,
            "bias": None,
            "mean_pred": None,
            "mean_target": None,
            "predicted_unsafe_fraction": None,
            "target_unsafe_fraction": None,
        }
    error = pred[mask] - target[mask]
    return {
        "count": count,
        "mse": float(np.mean(error ** 2)),
        "bias": float(np.mean(error)),
        "mean_pred": float(np.mean(pred[mask])),
        "mean_target": float(np.mean(target[mask])),
        "predicted_unsafe_fraction": float(np.mean(pred[mask] > 0.0)),
        "target_unsafe_fraction": float(np.mean(target[mask] > 0.0)),
    }


def bin_masks(values):
    """Disjoint masks for deep-safe, near-boundary, and positive values."""
    return {
        "deep_safe": values < BIN_EDGES[1],
        "near_boundary": (
            (values >= BIN_EDGES[1]) & (values <= BIN_EDGES[2])
        ),
        "positive": values > BIN_EDGES[2],
    }


def sign_metrics(pred, target):
    """Sign agreement and the two directional error rates."""
    predicted_unsafe = pred > 0.0
    target_unsafe = target > 0.0
    target_unsafe_count = int(np.count_nonzero(target_unsafe))
    target_safe_count = int(np.count_nonzero(~target_unsafe))
    false_safe_count = int(np.count_nonzero(~predicted_unsafe & target_unsafe))
    false_unsafe_count = int(np.count_nonzero(predicted_unsafe & ~target_unsafe))
    return {
        "sign_accuracy": float(np.mean(predicted_unsafe == target_unsafe)),
        "false_safe_count": false_safe_count,
        "false_unsafe_count": false_unsafe_count,
        "false_safe_rate": (
            false_safe_count / target_unsafe_count
            if target_unsafe_count else None
        ),
        "false_unsafe_rate": (
            false_unsafe_count / target_safe_count
            if target_safe_count else None
        ),
        "target_unsafe_fraction": (
            target_unsafe_count / len(target) if len(target) else None
        ),
    }


def compute_metrics(pred, data, mask, train_target_mean, mc_target, mc_mask):
    """All held-out diagnostics for one set of predictions."""
    pred = np.asarray(pred, dtype=np.float64)[mask]
    target = np.asarray(data["targets"], dtype=np.float64)[mask]
    h = np.asarray(data["h"], dtype=np.float64)[mask]
    not_done = np.asarray(data["not_done"], dtype=np.float64)[mask]
    collision = np.asarray(data["collision"], dtype=bool)[mask]
    goal = np.asarray(data["goal"], dtype=bool)[mask]
    truncated = np.asarray(data["truncated"], dtype=bool)[mask]

    error = pred - target
    metrics = {
        "count": int(len(pred)),
        "mse": float(np.mean(error ** 2)) if len(pred) else None,
        "rmse": float(np.sqrt(np.mean(error ** 2))) if len(pred) else None,
        "mae": float(np.mean(np.abs(error))) if len(pred) else None,
        "bias": float(np.mean(error)) if len(pred) else None,
        "mean_pred": float(np.mean(pred)) if len(pred) else None,
        "mean_target": float(np.mean(target)) if len(pred) else None,
        "target_std": float(np.std(target)) if len(pred) else None,
        # Baselines that make the absolute MSE interpretable.
        "constant_baseline_mse": (
            float(np.mean((target - train_target_mean) ** 2))
            if len(pred) else None
        ),
        "h_baseline_mse": (
            float(np.mean((h - target) ** 2)) if len(pred) else None
        ),
    }
    metrics.update(sign_metrics(pred, target))
    metrics["by_target_bin"] = {
        name: subset_metrics(pred, target, values)
        for name, values in bin_masks(target).items()
    }
    metrics["by_h_bin"] = {
        name: subset_metrics(pred, target, values)
        for name, values in bin_masks(h).items()
    }
    terminal_masks = {
        "collision_terminal": collision & (not_done == 0.0),
        "goal_terminal": goal & (not_done == 0.0),
        "truncated": truncated,
        "non_terminal": (not_done == 1.0) & ~truncated,
    }
    metrics["terminal"] = {
        name: subset_metrics(pred, target, values)
        for name, values in terminal_masks.items()
    }

    calibration_mask = mc_mask[mask]
    if np.any(calibration_mask):
        mc_pred = pred[calibration_mask]
        mc = np.asarray(mc_target, dtype=np.float64)[mask][calibration_mask]
        calibration = sign_metrics(mc_pred, mc)
        calibration["mse"] = float(np.mean((mc_pred - mc) ** 2))
        calibration["count"] = int(len(mc))
        calibration["mean_mc"] = float(np.mean(mc))
    else:
        calibration = {"count": 0}
    metrics["monte_carlo"] = calibration
    return metrics


def predict(critic, states, actions, batch_size, device):
    """Deterministic safety values for a set of state-action pairs."""
    critic.eval()
    predictions = np.empty(len(states), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            state = torch.as_tensor(
                states[start:end], dtype=torch.float32, device=device
            )
            action = torch.as_tensor(
                actions[start:end], dtype=torch.float32, device=device
            )
            value = critic(state, action)
            # A twin critic returns (head1, head2); report head 1, matching
            # the value the agent uses for the actor and logging.
            if isinstance(value, tuple):
                value = value[0]
            predictions[start:end] = value.squeeze(-1).cpu()
    return predictions


def train_critic(critic, data, train_mask, val_mask, settings, args, device,
                 label):
    """Continue ``critic`` on fixed targets and evaluate held-out error."""
    optimizer = torch.optim.Adam(
        critic.parameters(), lr=settings["learning_rate"]
    )
    train_indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(args.seed)
    # A fixed train subset gives a stable curve for detecting overfitting.
    monitor_indices = rng.choice(
        train_indices, size=min(4096, len(train_indices)), replace=False
    )
    states = data["states"]
    actions = data["actions"]
    targets = data["targets"]
    mc_target = data["mc_target"]
    mc_mask = data["mc_available"]
    train_target_mean = float(np.mean(targets[train_mask]))

    def evaluate_held_out(step):
        predictions = predict(
            critic, states, actions, args.batch_size, device
        )
        metrics = compute_metrics(
            predictions, data, val_mask, train_target_mean, mc_target,
            mc_mask,
        )
        monitor_predictions = predictions[monitor_indices]
        metrics["train_mse"] = float(
            np.mean((monitor_predictions - targets[monitor_indices]) ** 2)
        )
        metrics["step"] = int(step)
        return metrics

    history = [evaluate_held_out(0)]
    print(
        f"{label}: step 0 held-out MSE={history[0]['mse']:.5f}, "
        f"collision-terminal MSE={history[0]['terminal']['collision_terminal']['mse']}"
    )
    for step in range(1, args.steps + 1):
        batch_indices = train_indices[
            rng.integers(0, len(train_indices), size=args.batch_size)
        ]
        state = torch.as_tensor(
            states[batch_indices], dtype=torch.float32, device=device
        )
        action = torch.as_tensor(
            actions[batch_indices], dtype=torch.float32, device=device
        )
        target = torch.as_tensor(
            targets[batch_indices], dtype=torch.float32, device=device
        ).unsqueeze(-1)
        critic.train()
        critic_output = critic(state, action)
        if isinstance(critic_output, tuple):
            # A twin critic is trained on both heads, matching the agent.
            head1, head2 = critic_output
            loss = F.mse_loss(head1, target) + F.mse_loss(head2, target)
        else:
            loss = F.mse_loss(critic_output, target)
        optimizer.zero_grad()
        loss.backward()
        if settings["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), settings["grad_clip"]
            )
        optimizer.step()

        if args.eval_every > 0 and step % args.eval_every == 0:
            metrics = evaluate_held_out(step)
            history.append(metrics)
            print(
                f"{label}: step {step} held-out MSE={metrics['mse']:.5f}, "
                f"train MSE={metrics['train_mse']:.5f}, "
                f"collision predicted unsafe="
                f"{metrics['terminal']['collision_terminal']['predicted_unsafe_fraction']}"
            )

    final = evaluate_held_out(args.steps)
    if not history or history[-1]["step"] != args.steps:
        history.append(final)
    critic.eval()
    return final, history, optimizer


def relative_change(initial, final):
    """Fractional reduction from ``initial`` to ``final``, if defined."""
    if initial is None or final is None or initial == 0.0:
        return None
    return float((initial - final) / initial)


def summarize(checkpoint_history, scratch_history):
    """Pre-registered decision summary for the report."""
    step0 = checkpoint_history[0]
    final = checkpoint_history[-1]
    summary = {
        "overall_change": relative_change(step0["mse"], final["mse"]),
        "near_boundary_change": relative_change(
            step0["by_target_bin"]["near_boundary"]["mse"],
            final["by_target_bin"]["near_boundary"]["mse"],
        ),
        "collision_terminal_change": relative_change(
            step0["terminal"]["collision_terminal"]["mse"],
            final["terminal"]["collision_terminal"]["mse"],
        ),
        "from_scratch_gain": None,
    }
    if scratch_history is not None:
        scratch_final = scratch_history[-1]
        if final["mse"] and scratch_final["mse"] is not None:
            summary["from_scratch_gain"] = float(
                (final["mse"] - scratch_final["mse"]) / final["mse"]
            )

    overall = summary["overall_change"]
    near = summary["near_boundary_change"]
    collision = summary["collision_terminal_change"]
    if overall is not None and near is not None and collision is not None:
        if overall >= 0.20 and near >= 0.20 and collision >= 0.25:
            verdict = "consistent with undertraining"
        elif overall < 0.10:
            verdict = "flat: not undertrained on this dataset"
        else:
            verdict = "mixed: safe majority improves, boundary does not"
    else:
        verdict = "insufficient data for the decision rule"
    summary["verdict"] = verdict
    return summary


def _finite(value):
    """Plot ``None`` (an empty subset) as a gap instead of an object array."""
    return np.nan if value is None else value


def save_figure(path, name, checkpoint_history, scratch_history):
    """Held-out error and collision-terminal safety curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [(checkpoint_history, "checkpoint critic")]
    if scratch_history is not None:
        series.append((scratch_history, "from scratch"))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    for history, label in series:
        steps = [entry["step"] for entry in history]
        axes[0].plot(
            steps,
            [_finite(entry["mse"]) for entry in history],
            label=label,
        )
        axes[1].plot(
            steps,
            [
                _finite(entry["by_target_bin"]["near_boundary"]["mse"])
                for entry in history
            ],
            label=label,
        )
        axes[2].plot(
            steps,
            [
                _finite(entry["terminal"]["collision_terminal"][
                    "predicted_unsafe_fraction"
                ])
                for entry in history
            ],
            label=label,
        )
    axes[0].set_title("Held-out MSE")
    axes[1].set_title("Near-boundary MSE (target in [-0.2, 0])")
    axes[2].set_title("Collision terminals predicted unsafe")
    axes[2].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.set_xlabel("safety-critic step")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("MSE")
    axes[1].set_ylabel("MSE")
    axes[2].set_ylabel("fraction")
    axes[0].legend(frameon=False)
    fig.suptitle(f"{name}: offline safety-critic convergence")
    fig.tight_layout()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(name, dataset, split, checkpoint_history, scratch_history,
                  decision):
    """Print the comparison table and the pre-registered verdict."""
    final = checkpoint_history[-1]
    scratch_final = scratch_history[-1] if scratch_history else None
    rows = [
        ("held-out MSE", final["mse"],
         scratch_final["mse"] if scratch_final else None),
        ("near-boundary MSE",
         final["by_target_bin"]["near_boundary"]["mse"],
         scratch_final["by_target_bin"]["near_boundary"]["mse"]
         if scratch_final else None),
        ("collision-terminal MSE",
         final["terminal"]["collision_terminal"]["mse"],
         scratch_final["terminal"]["collision_terminal"]["mse"]
         if scratch_final else None),
        ("collision predicted unsafe",
         final["terminal"]["collision_terminal"][
             "predicted_unsafe_fraction"
         ],
         scratch_final["terminal"]["collision_terminal"][
             "predicted_unsafe_fraction"
         ] if scratch_final else None),
        ("false-safe rate", final["false_safe_rate"],
         scratch_final["false_safe_rate"] if scratch_final else None),
        ("false-unsafe rate", final["false_unsafe_rate"],
         scratch_final["false_unsafe_rate"] if scratch_final else None),
    ]
    print()
    print(f"{name}: {dataset['num_transitions']} transitions, "
          f"{dataset['num_episodes']} episodes")
    print(
        f"Split by {split['by']}: {split['train_episodes']} train / "
        f"{split['val_episodes']} val episodes, "
        f"{split['train_transitions']} / {split['val_transitions']} transitions"
    )
    header = f"{'metric':<32} {'checkpoint':>12} {'from scratch':>13}"
    print(header)
    print("-" * len(header))
    for label, checkpoint_value, scratch_value in rows:
        checkpoint_text = (
            "n/a" if checkpoint_value is None else f"{checkpoint_value:.4f}"
        )
        scratch_text = (
            "n/a" if scratch_value is None else f"{scratch_value:.4f}"
        )
        print(f"{label:<32} {checkpoint_text:>12} {scratch_text:>13}")
    print()
    for key in (
        "overall_change",
        "near_boundary_change",
        "collision_terminal_change",
        "from_scratch_gain",
    ):
        value = decision[key]
        if value is not None:
            print(f"{key:<32} {value:+.1%}")
    print(f"verdict: {decision['verdict']}")


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_dir = os.path.join(args.output_dir, args.name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")
    config, env, policy, settings = resolve_run(run_dir, args)
    device = next(policy.actor.parameters()).device
    print(f"Loaded model: {os.path.join(run_dir, 'rcrl_td3')} on {device}")
    print(
        f"Collection: layout_type={args.layout_type}, "
        f"num_envs={args.num_envs}, expl_noise={args.expl_noise}, "
        f"min_start_goal_distance={settings['min_start_goal_distance']}"
    )

    data = collect_dataset(policy, env, settings, args, device)
    data["mc_target"], data["mc_available"] = monte_carlo_targets(
        data, settings["discount"]
    )
    num_transitions = len(data["states"])
    num_episodes = data["num_episodes"]

    if args.split_by == "episode":
        rng = np.random.default_rng(args.seed)
        episode_ids = np.unique(data["episode_ids"])
        shuffled = rng.permutation(episode_ids)
        num_val = max(1, int(round((1.0 - args.split_fraction) * len(shuffled))))
        val_episode_ids = shuffled[:num_val]
        is_train_episode = np.zeros(num_episodes, dtype=bool)
        is_train_episode[shuffled[num_val:]] = True
        train_mask = is_train_episode[data["episode_ids"]]
        val_mask = ~train_mask
        train_episodes = int(np.count_nonzero(is_train_episode))
        val_episodes = int(len(shuffled) - train_episodes)
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = rng.permutation(num_transitions)
        num_val = max(1, int(round((1.0 - args.split_fraction) * num_transitions)))
        train_mask = np.zeros(num_transitions, dtype=bool)
        train_mask[shuffled[num_val:]] = True
        val_mask = ~train_mask
        train_episodes = num_episodes
        val_episodes = num_episodes

    if not np.any(train_mask) or not np.any(val_mask):
        raise SystemExit("The split produced an empty train or held-out set")

    split = {
        "by": args.split_by,
        "fraction": args.split_fraction,
        "train_transitions": int(np.count_nonzero(train_mask)),
        "val_transitions": int(np.count_nonzero(val_mask)),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
    }
    collision_count = int(np.count_nonzero(data["collision"]))
    goal_count = int(np.count_nonzero(data["goal"]))
    truncated_count = int(np.count_nonzero(data["truncated"]))
    print(
        f"Collected {num_transitions} transitions from {num_episodes} "
        f"episodes ({collision_count} collisions, {goal_count} goals, "
        f"{truncated_count} truncations)"
    )
    print(
        f"Split: {split['train_transitions']} train / "
        f"{split['val_transitions']} held-out transitions"
    )

    checkpoint_critic = copy.deepcopy(policy.safety_critic)
    checkpoint_final, checkpoint_history, checkpoint_optimizer = train_critic(
        checkpoint_critic, data, train_mask, val_mask, settings, args, device,
        label="checkpoint",
    )

    scratch_final = None
    scratch_history = None
    scratch_optimizer = None
    if args.from_scratch:
        # Match the checkpoint's architecture so the fresh critic is a
        # like-for-like control.
        scratch_critic_class = (
            SafetyCriticTwin
            if config.get("safety_twin", False)
            else SafetyCritic
        )
        scratch_critic = scratch_critic_class(
            env.observation_space.shape[0], env.action_space.shape[0]
        ).to(device)
        scratch_final, scratch_history, scratch_optimizer = train_critic(
            scratch_critic, data, train_mask, val_mask, settings, args,
            device, label="from-scratch",
        )

    decision = summarize(checkpoint_history, scratch_history)
    dataset = {
        "layout_type": args.layout_type,
        "num_transitions": num_transitions,
        "num_episodes": num_episodes,
        "num_envs": args.num_envs,
        "expl_noise": args.expl_noise,
        "random_reset_margin": settings["random_reset_margin"],
        "random_reset_max_speed": settings["random_reset_max_speed"],
        "min_start_goal_distance": settings["min_start_goal_distance"],
        "collision_terminals": collision_count,
        "goal_terminals": goal_count,
        "truncations": truncated_count,
        "target_mean": float(np.mean(data["targets"])),
        "target_std": float(np.std(data["targets"])),
        "target_min": float(np.min(data["targets"])),
        "target_max": float(np.max(data["targets"])),
        "h_mean": float(np.mean(data["h"])),
        "h_min": float(np.min(data["h"])),
        "h_max": float(np.max(data["h"])),
        "monte_carlo_available": int(np.count_nonzero(data["mc_available"])),
    }
    print_summary(
        args.name, dataset, split, checkpoint_history, scratch_history,
        decision,
    )

    payload = {
        "format": "rcrl-safety-diagnostic-v1",
        "run": args.name,
        "run_dir": run_dir,
        "agent": config.get("agent"),
        "device": str(device),
        "settings": settings,
        "dataset": dataset,
        "split": split,
        "target_formula": (
            "(1 - discount) * h + discount * "
            "max(h, where(not_done, V_target(s', pi_target(s')), h'))"
        ),
        "caveats": [
            "Targets are frozen from the checkpoint's target networks; "
            "load() rebuilds those as copies of the online networks, so they "
            "are not the final Polyak average.",
            "A low residual means the critic can fit its own backup, not "
            "that the learned value is correct.",
            "Truncated episodes are excluded from the Monte-Carlo "
            "calibration metrics.",
        ],
        "checkpoint_critic": {
            "initial": checkpoint_history[0],
            "final": checkpoint_final,
            "history": checkpoint_history,
        },
        "from_scratch": (
            {
                "initial": scratch_history[0],
                "final": scratch_final,
                "history": scratch_history,
            }
            if scratch_history is not None else None
        ),
        "decision": decision,
    }

    if args.save_json is not None:
        directory = os.path.dirname(os.path.abspath(args.save_json))
        os.makedirs(directory, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        print(f"Saved JSON: {args.save_json}")
    if args.save_png is not None:
        save_figure(
            args.save_png, args.name, checkpoint_history, scratch_history
        )
        print(f"Saved figure: {args.save_png}")
    if args.save_checkpoint is not None:
        destination = os.path.abspath(args.save_checkpoint)
        if destination == os.path.abspath(run_dir):
            raise SystemExit(
                "--save-checkpoint must not overwrite the source run directory"
            )
        os.makedirs(destination, exist_ok=True)
        shutil.copy(
            os.path.join(run_dir, "config.json"),
            os.path.join(destination, "config.json"),
        )
        policy.safety_critic.load_state_dict(checkpoint_critic.state_dict())
        policy.safety_critic_optimizer = checkpoint_optimizer
        policy.save(os.path.join(destination, "rcrl_td3"))
        print(f"Saved continued checkpoint: {destination}")


if __name__ == "__main__":
    main(parse_args())
