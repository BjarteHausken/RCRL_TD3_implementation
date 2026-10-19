from environment.dynamics import DoubleIntegratorExactDiscretization
from environment.environment import CorridorEnvironment


def create_env(
    layout_set="one",
    dt=0.1,
    max_velocity=3.0,
    length=30.0,
    width=5.0,
    vehicle_radius=0.2,
    lidar_range=5.0,
    num_lidar_rays=64,
    max_acceleration=2.0,
    pillars_positions=None,
    goal_radius=0.5,
    max_steps=1000,
    normalize_obs=True,
):
    """Corridor environment with the standard training/evaluation parameters.

    Accepts every constructor parameter of
    ``DoubleIntegratorExactDiscretization`` and ``CorridorEnvironment``;
    the defaults are the values the training and evaluation scripts use.
    """
    dynamics = DoubleIntegratorExactDiscretization(dt=dt, max_velocity=max_velocity)
    return CorridorEnvironment(
        dynamics=dynamics,
        length=length,
        width=width,
        vehicle_radius=vehicle_radius,
        lidar_range=lidar_range,
        num_lidar_rays=num_lidar_rays,
        max_acceleration=max_acceleration,
        pillars_positions=pillars_positions,
        layout_set=layout_set,
        goal_radius=goal_radius,
        max_steps=max_steps,
        normalize_obs=normalize_obs,
    )
