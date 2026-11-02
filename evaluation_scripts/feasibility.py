"""Ground-truth maximal collision-free set for the corridor environment.

The corridor is static and the agent is a bounded-acceleration, bounded-speed
double integrator, so the largest set of states from which *some* control
sequence keeps the vehicle collision-free forever -- the maximal controlled
invariant set, or viability kernel -- can be computed on a discretized state
grid with the standard viability iteration

    K_0     = collision-free grid states
    K_{k+1} = K_k  intersect  { s : exists u with f(s, u) in K_k }

The sequence is decreasing, so it converges to the largest controlled
invariant subset of the collision-free states. No policy is involved: this is
the "actual largest feasible set" the learned safety value is compared
against.

Because the vehicle can always come to rest (the zero action is admissible),
on a static map every state that can reach a safe standstill is feasible; the
iteration also keeps states that can stay safe indefinitely by moving, so it
does not assume stopping.

The state grid is (x, y, vx, vy). Transitions mirror
``environment.dynamics.DoubleIntegratorExactDiscretization`` (midpoint
position update, velocity clipped to the speed limit) and land on the nearest
grid state, which makes the kernel the exact viability kernel of the
discretized system.
"""

import numpy as np


def position_grid(length, width, margin, spacing):
    """Regular (x, y) grid inset by ``margin`` from the walls.

    ``spacing`` is an upper bound: the grid is chosen with square-ish cells
    whose step is at most the requested spacing. Returns ``(xs, ys,
    positions)`` where ``positions`` is an ``(nx * ny, 2)`` array in
    ``meshgrid(indexing="ij")`` order.
    """
    x_min, x_max = margin, length - margin
    y_min, y_max = -width / 2.0 + margin, width / 2.0 - margin
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("margin leaves no free space in the corridor")
    nx = max(2, int(np.floor((x_max - x_min) / spacing)) + 1)
    ny = max(2, int(np.floor((y_max - y_min) / spacing)) + 1)
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    return xs, ys, np.column_stack((grid_x.ravel(), grid_y.ravel()))


def velocity_grid(max_velocity, spacing):
    """Symmetric velocity nodes from ``-max_velocity`` to ``max_velocity``."""
    half = max(1, int(np.floor(max_velocity / spacing)))
    return np.linspace(-max_velocity, max_velocity, 2 * half + 1)


def free_positions(env, positions):
    """Collision-free positions, mirroring ``CorridorEnvironment`` exactly.

    The environment treats touching a wall or pillar as a collision, so the
    free region uses the complementary strict inequalities.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    x = positions[:, 0]
    y = positions[:, 1]
    radius = env.vehicle_radius
    free = (
        (x - radius > 0.0)
        & (x + radius < env.length)
        & (y + radius < env.width / 2.0)
        & (y - radius > -env.width / 2.0)
    )
    if len(env.pillars) > 0:
        pillar_x, pillar_y, pillar_radius = np.asarray(env.pillars).T
        distance_sq = (
            (x[:, None] - pillar_x[None, :]) ** 2
            + (y[:, None] - pillar_y[None, :]) ** 2
        )
        free &= (
            distance_sq > (radius + pillar_radius)[None, :] ** 2
        ).all(axis=1)
    return free


def _nearest_index(values, origin, step, count):
    """Nearest grid index and an in-range mask for continuous values."""
    import torch

    index = torch.round((values - origin) / step).long()
    valid = (index >= 0) & (index < count)
    return index.clamp_(0, count - 1), valid


def _free_mask_torch(next_x, next_y, env, pillars, radius):
    """Collision-free mask for continuous positions, as a bool tensor."""
    import torch

    free = (
        (next_x - radius > 0.0)
        & (next_x + radius < env.length)
        & (next_y + radius < env.width / 2.0)
        & (next_y - radius > -env.width / 2.0)
    )
    for pillar in pillars:
        pillar_x, pillar_y, pillar_radius = pillar
        distance_sq = (next_x - pillar_x) ** 2 + (next_y - pillar_y) ** 2
        free &= distance_sq > (radius + pillar_radius) ** 2
    return free


def viability_kernel(env, xs, ys, vxs, vys, action_levels=3,
                     max_iterations=300, device=None):
    """Maximal collision-free set on the ``(x, y, vx, vy)`` grid.

    ``action_levels`` discretizes the normalized action box: with 3 the
    accelerations are ``{-a, 0, +a}`` per axis, with 5 they are
    ``{-a, -a/2, 0, +a/2, +a}``. Returns ``(kernel, info)`` where ``kernel``
    is a bool array of shape ``(len(xs), len(ys), len(vxs), len(vys))`` and
    ``info`` records the iteration count and whether the fixed point was
    reached within ``max_iterations``.
    """
    import torch

    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    dt = float(env.dynamics.dt)
    max_velocity = float(env.dynamics.max_velocity)
    max_acceleration = float(env.max_acceleration)
    radius = float(env.vehicle_radius)
    nx, ny, nvx, nvy = len(xs), len(ys), len(vxs), len(vys)

    def axis_tensor(values, shape):
        return torch.as_tensor(
            np.asarray(values, dtype=np.float32), device=device
        ).view(shape)

    X = axis_tensor(xs, (-1, 1, 1, 1))
    Y = axis_tensor(ys, (1, -1, 1, 1))
    VX = axis_tensor(vxs, (1, 1, -1, 1))
    VY = axis_tensor(vys, (1, 1, 1, -1))
    X, Y, VX, VY = torch.broadcast_tensors(X, Y, VX, VY)

    x_min, x_step = float(xs[0]), float(xs[1] - xs[0])
    y_min, y_step = float(ys[0]), float(ys[1] - ys[0])
    vx_min, vx_step = float(vxs[0]), float(vxs[1] - vxs[0])
    vy_min, vy_step = float(vys[0]), float(vys[1] - vys[0])

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    free = free_positions(
        env, np.column_stack((grid_x.ravel(), grid_y.ravel()))
    )
    safe = torch.as_tensor(
        free.reshape(nx, ny), device=device
    ).unsqueeze(-1).unsqueeze(-1)
    safe = safe.expand(nx, ny, nvx, nvy).clone()

    pillars = [
        (float(pillar[0]), float(pillar[1]), float(pillar[2]))
        for pillar in np.asarray(env.pillars).reshape(-1, 3)
    ] if len(env.pillars) > 0 else []
    levels = [float(level) for level in np.linspace(-1.0, 1.0, action_levels)]

    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        predecessor = torch.zeros_like(safe)
        for level_x in levels:
            for level_y in levels:
                # Same update as DoubleIntegratorExactDiscretization.
                next_vx = VX + level_x * max_acceleration * dt
                next_vy = VY + level_y * max_acceleration * dt
                speed = torch.sqrt(next_vx * next_vx + next_vy * next_vy)
                scale = torch.clamp(max_velocity / speed, max=1.0)
                next_vx = next_vx * scale
                next_vy = next_vy * scale
                next_x = X + 0.5 * (VX + next_vx) * dt
                next_y = Y + 0.5 * (VY + next_vy) * dt

                index_x, _ = _nearest_index(next_x, x_min, x_step, nx)
                index_y, _ = _nearest_index(next_y, y_min, y_step, ny)
                index_vx, _ = _nearest_index(next_vx, vx_min, vx_step, nvx)
                index_vy, _ = _nearest_index(next_vy, vy_min, vy_step, nvy)
                flat_index = (
                    ((index_x * ny + index_y) * nvx + index_vx) * nvy
                    + index_vy
                )
                next_safe = safe.reshape(-1)[flat_index]
                # The continuous next state must also be collision-free:
                # nearest-node rounding alone could step through a thin pillar.
                next_safe &= _free_mask_torch(
                    next_x, next_y, env, pillars, radius
                )
                predecessor.logical_or_(next_safe)

        new_safe = safe & predecessor
        if torch.equal(new_safe, safe):
            converged = True
            break
        safe = new_safe

    info = {
        "iterations": iteration,
        "converged": converged,
        "device": str(device),
        "action_levels": int(action_levels),
        "max_acceleration": max_acceleration,
        "max_velocity": max_velocity,
        "dt": dt,
        "grid_shape": (int(nx), int(ny), int(nvx), int(nvy)),
    }
    return safe.cpu().numpy(), info


def kernel_slice(kernel, vxs, vys, velocity):
    """Kernel values at the velocity grid node nearest ``velocity``."""
    velocity = np.asarray(velocity, dtype=np.float64).reshape(2)
    index_x = int(np.argmin(np.abs(np.asarray(vxs) - velocity[0])))
    index_y = int(np.argmin(np.abs(np.asarray(vys) - velocity[1])))
    return np.asarray(kernel[:, :, index_x, index_y], dtype=bool)


def set_metrics(learned_safe, true_safe, domain):
    """Compare two safe/unsafe grids over the free-position ``domain``.

    Both sets are restricted to the domain (grid positions where the vehicle
    does not currently overlap an obstacle), because those are the only
    states an episode can occupy. ``false_safe_rate`` is the share of truly
    infeasible states the learned value calls safe -- the dangerous error;
    ``false_unsafe_rate`` is the share of truly feasible states it calls
    unsafe -- the over-cautious error.
    """
    learned = np.asarray(learned_safe, dtype=bool) & domain
    true = np.asarray(true_safe, dtype=bool) & domain
    union = np.count_nonzero(learned | true)
    intersection = np.count_nonzero(learned & true)
    true_unsafe = np.count_nonzero(domain & ~true)
    true_safe_count = np.count_nonzero(true)
    domain_count = np.count_nonzero(domain)
    return {
        "iou": (intersection / union) if union else None,
        "false_safe_rate": (
            np.count_nonzero(learned & ~true) / true_unsafe
            if true_unsafe else None
        ),
        "false_unsafe_rate": (
            np.count_nonzero(~learned & true) / true_safe_count
            if true_safe_count else None
        ),
        "learned_safe_fraction": (
            np.count_nonzero(learned) / domain_count if domain_count else None
        ),
        "true_safe_fraction": (
            true_safe_count / domain_count if domain_count else None
        ),
        "domain_count": int(domain_count),
        "false_safe_count": int(np.count_nonzero(learned & ~true)),
        "false_unsafe_count": int(np.count_nonzero(~learned & true)),
    }


def average_metrics(entries):
    """Mean and standard deviation of the rate metrics over ``entries``."""
    result = {}
    for key in (
        "iou",
        "false_safe_rate",
        "false_unsafe_rate",
        "learned_safe_fraction",
        "true_safe_fraction",
    ):
        values = [
            entry[key] for entry in entries if entry.get(key) is not None
        ]
        result[key] = float(np.mean(values)) if values else None
        result[key + "_std"] = float(np.std(values)) if values else None
    return result
