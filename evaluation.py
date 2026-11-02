"""Batched policy evaluation on the corridor layouts.

Every evaluation episode is stepped in parallel with one batched actor
forward pass per timestep and ``environment.step_batch`` instead of running
complete single-environment episodes one after another. Each layout is
evaluated on ``episodes_per_layout`` independently sampled start/goal pairs
(the same randomized regime as training), seeded by episode index so runs are
reproducible. Finished episodes leave the batch, so evaluation ends as soon
as every episode has terminated or been truncated.
"""

import numpy as np
import torch

from environment import create_env, step_batch


def _wilson_interval(count, total, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = count / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _inflate_lidar(observations, shift, reference):
    """Shorten the lidar part of observations by ``shift`` metres.

    The observation layout is velocity (2), lidar (``num_lidar_rays``), and
    goal-relative position (2). Shifting the lidar makes the actor perceive
    every obstacle as ``shift`` metres closer, implementing the runtime
    safety margin in observation space; the environment and the evaluation
    metrics keep using the true lidar. Normalized observations store lidar
    divided by the lidar range, so the shift is scaled the same way.
    """
    if shift <= 0.0:
        return observations
    if reference.normalize_obs:
        shift = shift / reference.lidar_range
    observations = np.array(observations, copy=True)
    lidar = slice(2, 2 + reference.num_lidar_rays)
    observations[..., lidar] = np.maximum(
        observations[..., lidar] - shift, 0.0
    )
    return observations


def _multiplier_stats(policy, states):
    """Statistics of the RCRL multiplier lambda(s) over visited states."""
    device = next(policy.actor.parameters()).device
    with torch.inference_mode():
        values = policy.multiplier_for(
            torch.as_tensor(states, dtype=torch.float32, device=device)
        ).squeeze(-1)
        stats = {
            "mean": float(values.mean()),
            "std": float(values.std()) if values.numel() > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
        if policy.multiplier_type == "network":
            stats["cap_fraction"] = float(
                (values >= 0.99 * policy.max_lambda).float().mean()
            )
            floor_tolerance = 0.01 * (policy.max_lambda - policy.min_lambda)
            stats["floor_fraction"] = float(
                (values <= policy.min_lambda + floor_tolerance).float().mean()
            )
        return stats


def evaluate_policy(
    policy,
    layout_set,
    normalize_obs=True,
    layout_type="val",
    episodes_per_layout=20,
    random_reset_margin=0.3,
    random_reset_max_speed=1.0,
    min_start_goal_distance=None,
    safety_margin=None,
    reward_scale=1.0,
):
    """Evaluate a deterministic policy on every layout of ``layout_type``.

    Each layout runs ``episodes_per_layout`` episodes, each with a freshly
    sampled feasible start and goal. Sampling uses the same clearance margin
    and speed bound as training, so evaluation and training share an episode
    distribution. All episodes are stepped in parallel; once an episode
    terminates or is truncated it leaves the batch.

    ``min_start_goal_distance`` bounds the sampled start-to-goal distance
    (default ``2 * goal_radius``); it should match the value used in training
    so evaluation and training share an episode distribution.

    ``safety_margin`` optionally sets the clearance threshold (metres from the
    vehicle centre to the nearest obstacle surface) used for the violation
    metrics: ``h(s) = safety_margin - min(lidar)`` and ``h(s) > 0`` is a
    margin violation. At runtime it also acts as an observation-space safety
    margin: the lidar the actor sees is shortened by
    ``safety_margin - vehicle_radius``, so the policy reacts to obstacles as
    if they were that much closer. Collision, termination, rewards, and all
    metrics still use the true lidar. The threshold defaults to
    ``vehicle_radius``, the boundary the policy was trained against, which
    means no observation shift and no extra safety margin. ``reward_scale``
    must match the environment reward divisor used for training so reported
    rewards are comparable.

    Returns the aggregate metrics ``mean_reward``, ``mean_steps``,
    ``success_rate``, ``collision_rate`` and their 95% Wilson intervals
    (``success_rate_ci``, ``collision_rate_ci``), the per-episode arrays
    ``rewards``, ``steps``, ``successes``, ``collisions`` and
    ``layout_indices``, and the per-layout means ``per_layout_success_rate``
    and ``per_layout_collision_rate``.

    Margin metrics: ``margin_violation_rate`` is the fraction of episodes
    that entered the safety margin at least once
    (``margin_violations``), ``margin_step_fraction`` is the mean fraction of
    an episode spent inside the margin (``margin_steps`` per episode), and
    ``min_clearance`` / ``max_h`` record the closest approach and largest
    constraint value per episode. ``mean_clearance`` is the mean clearance
    over all visited states. All clearance values are distances from the
    vehicle centre to the nearest obstacle surface; the collision boundary is
    ``vehicle_radius``.

    For RCRL policies the result also includes ``lambda_stats``: the mean,
    standard deviation, min and max of the multiplier lambda(s) over a
    subsample of visited states (plus ``cap_fraction`` and ``floor_fraction``
    for network multipliers), measured on the same visited distribution the
    margin metrics use. Plain TD3 policies return ``lambda_stats=None``.
    """
    if episodes_per_layout < 1:
        raise ValueError("episodes_per_layout must be positive")

    reference = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        reward_scale=reward_scale,
    )
    # The environment constrains at the vehicle radius. Evaluation may report
    # the margin metrics against a wider clearance threshold; the default is
    # the training boundary itself.
    if safety_margin is None:
        safety_margin = reference.vehicle_radius
    elif safety_margin < reference.vehicle_radius:
        raise ValueError(
            "safety_margin must be at least vehicle_radius; otherwise the "
            "constraint would be satisfied in collision states"
        )
    if safety_margin > reference.lidar_range:
        raise ValueError(
            "safety_margin must not exceed lidar_range; the sensor could not "
            "observe the full margin"
        )
    if min_start_goal_distance is None:
        min_start_goal_distance = 2 * reference.goal_radius
    layouts = (
        reference.val_layouts if layout_type == "val" else reference.test_layouts
    )
    num_layouts = len(layouts)
    num_episodes = num_layouts * episodes_per_layout

    envs = []
    states = np.empty(
        (num_episodes, reference.observation_space.shape[0]), dtype=np.float32
    )
    layout_indices = np.repeat(np.arange(num_layouts), episodes_per_layout)
    for episode_index in range(num_episodes):
        env = create_env(
            layout_set,
            normalize_obs=normalize_obs,
            reward_scale=reward_scale,
        )
        states[episode_index], _ = env.reset(
            layout_type=layout_type,
            layout_index=int(layout_indices[episode_index]),
            seed=episode_index,
            sample_start=True,
            start_margin=random_reset_margin,
            start_max_speed=random_reset_max_speed,
            goal_margin=random_reset_margin,
            min_start_goal_distance=min_start_goal_distance,
        )
        envs.append(env)

    device = next(policy.actor.parameters()).device
    episode_reward = np.zeros(num_episodes)
    episode_steps = np.zeros(num_episodes, dtype=int)
    successes = np.zeros(num_episodes, dtype=bool)
    collisions = np.zeros(num_episodes, dtype=bool)
    active = np.ones(num_episodes, dtype=bool)

    # Margin-violation bookkeeping. ``constraint`` in info is h(s) before the
    # action; ``next_constraint`` is h(s') after it. The former counts time
    # spent in the margin, while both feed the per-episode closest approach.
    margin_steps = np.zeros(num_episodes, dtype=int)
    max_h = np.full(num_episodes, -np.inf)
    min_clearance = np.full(num_episodes, np.inf)
    clearance_sum = np.zeros(num_episodes)
    constraint_margin = safety_margin
    # Runtime safety margin as observation-space obstacle inflation. Zero by
    # default (threshold = vehicle radius), so training and plain evaluation
    # see the true lidar.
    lidar_shift = safety_margin - reference.vehicle_radius

    # RCRL policies: sample visited states (every fifth batch step, up to a
    # cap) so the multiplier lambda(s) can be measured on the same visited
    # distribution the margin metrics use.
    collect_lambda = hasattr(policy, "multiplier_for")
    lambda_sample_interval = 5
    lambda_pool_cap = 50000
    lambda_states = []
    lambda_state_count = 0
    step_count = 0

    while active.any():
        step_count += 1
        active_indices = np.flatnonzero(active)
        with torch.inference_mode():
            observations = torch.as_tensor(
                _inflate_lidar(states[active_indices], lidar_shift, reference),
                dtype=torch.float32,
                device=device,
            )
            actions = policy.actor(observations).cpu().numpy()

        next_states, rewards, terminated, truncated, info = step_batch(
            [envs[index] for index in active_indices], actions
        )

        states[active_indices] = next_states
        episode_reward[active_indices] += rewards
        episode_steps[active_indices] += 1
        if (
            collect_lambda
            and step_count % lambda_sample_interval == 0
            and lambda_state_count < lambda_pool_cap
        ):
            chunk = states[active_indices].copy()
            lambda_states.append(chunk)
            lambda_state_count += chunk.shape[0]

        # The environment reports h(s) = vehicle_radius - min(lidar), so the
        # clearance to the nearest obstacle is vehicle_radius - h(s).
        constraint = info["constraint"]
        next_constraint = info["next_constraint"]
        clearance = reference.vehicle_radius - constraint
        next_clearance = reference.vehicle_radius - next_constraint
        margin_steps[active_indices] += clearance < constraint_margin
        clearance_sum[active_indices] += clearance
        max_h[active_indices] = np.maximum(
            max_h[active_indices],
            np.maximum(
                constraint_margin - clearance,
                constraint_margin - next_clearance,
            ),
        )
        min_clearance[active_indices] = np.minimum(
            min_clearance[active_indices],
            np.minimum(clearance, next_clearance),
        )

        # Record each episode's outcome at its first terminal or truncated
        # step, then drop it from the batch.
        finished = terminated | truncated
        finished_indices = active_indices[finished]
        successes[finished_indices] = info["goal_reached"][finished]
        collisions[finished_indices] = info["collision"][finished]
        active[finished_indices] = False

    margin_violations = max_h > 0.0
    margin_step_fraction = np.divide(
        margin_steps,
        episode_steps,
        out=np.zeros(num_episodes, dtype=float),
        where=episode_steps > 0,
    )
    mean_clearance = np.divide(
        clearance_sum,
        episode_steps,
        out=np.zeros(num_episodes, dtype=float),
        where=episode_steps > 0,
    )

    lambda_stats = None
    if collect_lambda and lambda_states:
        lambda_stats = _multiplier_stats(policy, np.concatenate(lambda_states))

    return {
        "lambda_stats": lambda_stats,
        "safety_margin": float(safety_margin),
        "observation_lidar_shift": float(lidar_shift),
        "mean_reward": float(episode_reward.mean()),
        "mean_steps": float(episode_steps.mean()),
        "success_rate": float(successes.mean()),
        "collision_rate": float(collisions.mean()),
        "success_rate_ci": _wilson_interval(successes.sum(), num_episodes),
        "collision_rate_ci": _wilson_interval(collisions.sum(), num_episodes),
        "margin_violation_rate": float(margin_violations.mean()),
        "margin_violation_rate_ci": _wilson_interval(
            margin_violations.sum(), num_episodes
        ),
        "margin_step_fraction": float(margin_step_fraction.mean()),
        "mean_clearance": float(mean_clearance.mean()),
        "mean_min_clearance": float(min_clearance.mean()),
        "rewards": episode_reward,
        "steps": episode_steps,
        "successes": successes,
        "collisions": collisions,
        "margin_violations": margin_violations,
        "margin_steps": margin_steps,
        "margin_step_fraction_per_episode": margin_step_fraction,
        "mean_clearance_per_episode": mean_clearance,
        "max_h": max_h,
        "min_clearance": min_clearance,
        "layout_indices": layout_indices,
        "episodes_per_layout": episodes_per_layout,
        "per_layout_success_rate": successes.reshape(
            num_layouts, episodes_per_layout
        ).mean(axis=1),
        "per_layout_collision_rate": collisions.reshape(
            num_layouts, episodes_per_layout
        ).mean(axis=1),
        "per_layout_margin_violation_rate": margin_violations.reshape(
            num_layouts, episodes_per_layout
        ).mean(axis=1),
        "per_layout_min_clearance": min_clearance.reshape(
            num_layouts, episodes_per_layout
        ).mean(axis=1),
    }
