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


def evaluate_policy(
    policy,
    layout_set,
    normalize_obs=True,
    layout_type="val",
    episodes_per_layout=20,
    random_reset_margin=0.3,
    random_reset_max_speed=1.0,
):
    """Evaluate a deterministic policy on every layout of ``layout_type``.

    Each layout runs ``episodes_per_layout`` episodes, each with a freshly
    sampled feasible start and goal. Sampling uses the same clearance margin
    and speed bound as training, so evaluation and training share an episode
    distribution. All episodes are stepped in parallel; once an episode
    terminates or is truncated it leaves the batch.

    Returns the aggregate metrics ``mean_reward``, ``mean_steps``,
    ``success_rate``, ``collision_rate`` and their 95% Wilson intervals
    (``success_rate_ci``, ``collision_rate_ci``), the per-episode arrays
    ``rewards``, ``steps``, ``successes``, ``collisions`` and
    ``layout_indices``, and the per-layout means ``per_layout_success_rate``
    and ``per_layout_collision_rate``.
    """
    if episodes_per_layout < 1:
        raise ValueError("episodes_per_layout must be positive")

    reference = create_env(layout_set, normalize_obs=normalize_obs)
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
        env = create_env(layout_set, normalize_obs=normalize_obs)
        states[episode_index], _ = env.reset(
            layout_type=layout_type,
            layout_index=int(layout_indices[episode_index]),
            seed=episode_index,
            sample_start=True,
            start_margin=random_reset_margin,
            start_max_speed=random_reset_max_speed,
            goal_margin=random_reset_margin,
            min_start_goal_distance=2 * env.goal_radius,
        )
        envs.append(env)

    device = next(policy.actor.parameters()).device
    episode_reward = np.zeros(num_episodes)
    episode_steps = np.zeros(num_episodes, dtype=int)
    successes = np.zeros(num_episodes, dtype=bool)
    collisions = np.zeros(num_episodes, dtype=bool)
    active = np.ones(num_episodes, dtype=bool)

    while active.any():
        active_indices = np.flatnonzero(active)
        with torch.inference_mode():
            observations = torch.as_tensor(
                states[active_indices], dtype=torch.float32, device=device
            )
            actions = policy.actor(observations).cpu().numpy()

        next_states, rewards, terminated, truncated, info = step_batch(
            [envs[index] for index in active_indices], actions
        )

        states[active_indices] = next_states
        episode_reward[active_indices] += rewards
        episode_steps[active_indices] += 1

        # Record each episode's outcome at its first terminal or truncated
        # step, then drop it from the batch.
        finished = terminated | truncated
        finished_indices = active_indices[finished]
        successes[finished_indices] = info["goal_reached"][finished]
        collisions[finished_indices] = info["collision"][finished]
        active[finished_indices] = False

    return {
        "mean_reward": float(episode_reward.mean()),
        "mean_steps": float(episode_steps.mean()),
        "success_rate": float(successes.mean()),
        "collision_rate": float(collisions.mean()),
        "success_rate_ci": _wilson_interval(successes.sum(), num_episodes),
        "collision_rate_ci": _wilson_interval(collisions.sum(), num_episodes),
        "rewards": episode_reward,
        "steps": episode_steps,
        "successes": successes,
        "collisions": collisions,
        "layout_indices": layout_indices,
        "episodes_per_layout": episodes_per_layout,
        "per_layout_success_rate": successes.reshape(
            num_layouts, episodes_per_layout
        ).mean(axis=1),
        "per_layout_collision_rate": collisions.reshape(
            num_layouts, episodes_per_layout
        ).mean(axis=1),
    }
