import os
import sys

# Allow ``python utils/test.py`` from the repository root: the script
# directory is on sys.path, not the repository root, so add the root explicitly
# before importing the project's packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from environment import create_env, step_batch
from environment.dynamics import DoubleIntegratorExactDiscretization
from environment.environment import CorridorEnvironment
from gymnasium.utils.env_checker import check_env

dynamics = DoubleIntegratorExactDiscretization(dt=0.1, max_velocity=5.0)
env = CorridorEnvironment(dynamics=dynamics, length=30.0, width=5.0)

env.pillars = np.array([(10.0, 0.0, 0.5)])


def test_collisions(env):
    # The cases below assume the constructor's 5.0 m width, independent of
    # whichever layout the environment adopted at construction.
    env._apply_layout({"width": env.default_width, "pillars": env.pillars})
    # A reset gives render() a goal; the per-case states below then override
    # the dynamics state directly.
    env.reset(layout_type="fixed", state=[5.0, 0.0, 0.0, 0.0], goal=[15.0, 0.0])

    test_cases = [
        # name, state, expected collision

        ("Middle of corridor",
         [5.0, 0.0, 0.0, 0.0],
         False),

        ("Close to upper wall, but safe",
         [5.0, 0.0, 2.2, 0.0],
         False),

        ("Touching upper wall",
         [5.0, 0.0, 2.3, 0.0],
         True),

        ("Through upper wall",
         [5.0, 0.0, 2.5, 0.0],
         True),

        ("Close to pillar, but safe",
         [9.2, 0.0, 0.0, 0.0],
         False),

        ("Touching pillar",
         [9.3, 0.0, 0.0, 0.0],
         True),

        ("Inside pillar",
         [10.0, 0.0, 0.0, 0.0],
         True),
    ]

    for name, state, expected in test_cases:

        env.dynamics.state = state

        result = env.check_collision()

        status = "PASS" if result == expected else "FAIL"
        env.render()
        print(
            f"{status}: {name:30} "
            f"expected={expected}, got={result}"
        )


def test_render(env):
    env.reset(state=[5.0, 2.0, 0.0, 0.5])
    env.render()
    env.step([10.0, 0.0])
    env.render()

def test_lidar(env):

    # Test pillar ray intersection
    origin = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])

    center = np.array([3.0, 0.0])
    radius = 0.5

    distance = env._ray_circle_intersection(
        origin,
        direction,
        center,
        radius
    )

    print("Hit pillar, ", distance)
    
    direction = np.array([0.0, 1.0])

    distance = env._ray_circle_intersection(
        origin,
        direction,
        center,
        radius
    )

    print("Miss pillar, ", distance)
    
    # Test wall ray intersection
    origin = np.array([5.0, 0.0])
    direction = np.array([0.0, 1.0])

    distance = env._ray_wall_intersection(
        origin,
        direction
    )

    print("Hit wall, ", distance)
    
    # Test many lidar rays
    env.reset(state=[5.0, 0.0, 0.0, 0.0])
    lidar = env.get_lidar()
    print(lidar)
    print(lidar.shape)
    print(lidar.min())
    print(lidar.max())

def test_lidar_at(env):
    # The batched lidar helper used for off-trajectory value queries must
    # reproduce get_lidar exactly, on the current layout.
    env.reset(layout_type="train", layout_index=0,
              state=[5.0, 0.0, 0.0, 0.0], goal=[20.0, 0.0])
    positions = [(1.0, 0.0), (5.0, 1.0), (14.5, -2.0), (28.0, 0.5)]
    batched = env.lidar_at(np.array(positions))
    for position, distances in zip(positions, batched):
        env.dynamics.state = np.array([position[0], 0.0, position[1], 0.0])
        np.testing.assert_allclose(distances, env.get_lidar(), atol=1e-12)
    print("PASS: lidar_at matches get_lidar")

def test_gymnasium_env(env):
    check_env(env, warn=True)
    print("Gymnasium environment check passes")
    obs, info = env.reset()

    print("Observation shape:", obs.shape)
    print("Observation:", obs)

    for i in range(10):

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {i}: "
            f"action={action}, "
            f"reward={reward}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

        if terminated or truncated:
            obs, info = env.reset()

def test_velocity_limit(dynamics):
    dynamics.reset()

    for _ in range(1000):
        dynamics.update([1.0, 1.0])

    x, x_dot, y, y_dot = dynamics.get_state()

    speed = np.sqrt(x_dot**2 + y_dot**2)

    print("x velocity:", x_dot)
    print("y velocity:", y_dot)
    print("Speed:", speed)
    
    assert speed <= dynamics.max_velocity + 1e-6
    
    print("PASS: velocity bounded")

def test_reward_and_constraints(env):
    obs, info = env.reset()

    print("Observation shape:", obs.shape)
    
    episodes = 10

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            x, x_dot, y, y_dot = env.dynamics.get_state()
            speed = np.sqrt(x_dot**2 + y_dot**2)

            lidar = env.get_lidar()

            # Sanity checks
            assert env.observation_space.contains(obs)
            assert np.all(np.isfinite(obs))
            assert speed <= env.dynamics.max_velocity + 1e-6
            assert np.all(np.isfinite(lidar))
            assert np.all(lidar <= env.lidar_range + 1e-6)

            if terminated or truncated:
                break    
          
        print(
            f"Episode {episode}: "
            f"steps={step_count}, "
            f"reward={total_reward:.2f}, "
            f"collision={info['collision']}, "
            f"goal={info['goal_reached']}, "
        )

def test_random_state_sampling(env):
    # Sampled states must be collision-free and keep at least the requested
    # clearance margin from walls and pillars
    margin = 0.3
    max_speed = 1.0
    for _ in range(50):
        state = env.sample_random_state(margin=margin, max_speed=max_speed)
        x, x_dot, y, y_dot = state
        env.dynamics.state = state

        assert not env.check_collision(), f"sampled state collides: {state}"
        assert env.width / 2 - abs(y) - env.vehicle_radius >= margin - 1e-6
        assert x - env.vehicle_radius - margin >= 0.0 - 1e-6
        for pillar_x, pillar_y, pillar_radius in env.pillars:
            assert np.hypot(x - pillar_x, y - pillar_y) >= (
                pillar_radius + env.vehicle_radius + margin - 1e-6
            ), f"sampled state too close to pillar: {state}"
        assert np.hypot(x_dot, y_dot) <= max_speed + 1e-6

    print("PASS: random state sampling respects collision margins")

    # reset(state=...) must apply the sampled state and keep the
    # observation inside the observation space
    state = env.sample_random_state(margin=margin, max_speed=max_speed)
    obs, _ = env.reset(layout_type="fixed", state=state)
    np.testing.assert_allclose(env.dynamics.get_state(), state, atol=1e-6)
    assert env.observation_space.contains(obs)

    print("PASS: reset with a custom state")

def test_goal(env):
    goal = np.array([10.0, 0.0])
    state = np.array([5.0, 0.0, 0.0, 0.0])

    # The goal passed to reset is stored, and the relative goal coordinates
    # appear in the observation as (goal - position), scaled when normalizing.
    obs, _ = env.reset(layout_type="fixed", state=state, goal=goal)
    np.testing.assert_allclose(env.goal, goal, atol=1e-6)
    relative = np.array([goal[0] - 5.0, goal[1] - 0.0])
    if env.normalize_obs:
        relative = relative / np.array([env.length, env.width])
    np.testing.assert_allclose(obs[-2:], relative, atol=1e-6)

    # Reward must be higher when moving toward the goal than when moving away.
    _, _ = env.reset(layout_type="fixed", state=state, goal=goal)
    _, toward_reward, *_ = env.step([1.0, 0.0])
    env.reset(layout_type="fixed", state=state, goal=goal)
    _, away_reward, *_ = env.step([-1.0, 0.0])
    assert toward_reward > away_reward, (toward_reward, away_reward)
    print(f"PASS: reward favors moving toward the goal ({toward_reward:.3f} > {away_reward:.3f})")

    # Within goal_radius of the goal, stepping must terminate with the bonus.
    near_state = np.array([10.0 - 0.3 * env.goal_radius, 0.0, 0.0, 0.0])
    env.reset(layout_type="fixed", state=near_state, goal=goal)
    _, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert info["goal_reached"], info
    assert terminated and not truncated
    assert reward >= 50.0 - 0.1 - 1e-6, reward
    print(f"PASS: goal reached terminates with bonus (reward={reward:.3f})")

def test_batch_matches_single():
    # The hand-vectorized step_batch must reproduce scalar step() exactly,
    # so a fixed layout, goal, and start state are shared by both envs.
    state = [10.0, 0.5, 1.0, -0.3]
    goal = [20.0, -1.0]
    env_a = create_env("one", reward_scale=50.0)
    env_b = create_env("one", reward_scale=50.0)
    for env in (env_a, env_b):
        env.reset(layout_type="train", layout_index=0, state=state, goal=goal)

    action = np.array([0.5, -0.25])
    obs_a, reward_a, terminated_a, truncated_a, info_a = env_a.step(action)
    obs_b, reward_b, terminated_b, truncated_b, info_b = step_batch(
        [env_b], [action]
    )

    np.testing.assert_allclose(
        env_a.dynamics.get_state(), env_b.dynamics.get_state(), atol=1e-6
    )
    np.testing.assert_allclose(obs_a, obs_b[0], atol=1e-6)
    assert np.isclose(reward_a, reward_b[0], atol=1e-9), (reward_a, reward_b)
    assert bool(terminated_a) == bool(terminated_b[0])
    assert bool(truncated_a) == bool(truncated_b[0])
    np.testing.assert_allclose(
        info_a["constraint"], info_b["constraint"][0], atol=1e-9
    )
    np.testing.assert_allclose(
        info_a["next_constraint"], info_b["next_constraint"][0], atol=1e-9
    )
    assert bool(info_a["collision"]) == bool(info_b["collision"][0])
    assert bool(info_a["goal_reached"]) == bool(info_b["goal_reached"][0])
    print("PASS: step_batch matches step")

def test_randomized_reset(env):
    # Sampled starts and goals from reset() must clear walls (including the
    # back wall) and pillars by the requested margin, and the goal must stay
    # at least two goal radii from the start.
    margin = 0.3
    for _ in range(20):
        env.reset(
            layout_type="train",
            sample_start=True,
            start_margin=margin,
            start_max_speed=1.0,
            goal_margin=margin,
            min_start_goal_distance=2 * env.goal_radius,
        )
        x, x_dot, y, y_dot = env.dynamics.get_state()
        assert not env.check_collision(), f"sampled start collides: {x}, {y}"
        assert env.width / 2 - abs(y) - env.vehicle_radius >= margin - 1e-6
        assert x - env.vehicle_radius - margin >= -1e-6
        assert env.length - x - env.vehicle_radius - margin >= -1e-6
        for pillar_x, pillar_y, pillar_radius in env.pillars:
            assert np.hypot(x - pillar_x, y - pillar_y) >= (
                pillar_radius + env.vehicle_radius + margin - 1e-6
            ), f"start too close to pillar: {x}, {y}"
        assert np.hypot(x_dot, y_dot) <= 1.0 + 1e-6

        goal_x, goal_y = env.goal
        assert env.width / 2 - abs(goal_y) - env.vehicle_radius >= margin - 1e-6
        assert goal_x - env.vehicle_radius - margin >= -1e-6
        assert env.length - goal_x - env.vehicle_radius - margin >= -1e-6
        for pillar_x, pillar_y, pillar_radius in env.pillars:
            assert np.hypot(goal_x - pillar_x, goal_y - pillar_y) >= (
                pillar_radius + env.vehicle_radius + margin - 1e-6
            ), f"goal too close to pillar: {goal_x}, {goal_y}"
        assert np.hypot(x - goal_x, y - goal_y) >= 2 * env.goal_radius - 1e-6

    print("PASS: randomized reset start and goal respect margins")

def test_min_start_goal_distance(env):
    # A feasible minimum distance is respected, and an infeasible one raises
    # instead of resampling the goal forever.
    distance = 12.0
    env.reset(
        layout_type="train",
        layout_index=0,
        seed=0,
        sample_start=True,
        min_start_goal_distance=distance,
    )
    x, _, y, _ = env.dynamics.get_state()
    goal_x, goal_y = env.goal
    assert np.hypot(x - goal_x, y - goal_y) > distance, (x, y, goal_x, goal_y)

    try:
        env.reset(
            layout_type="train",
            layout_index=0,
            sample_start=True,
            min_start_goal_distance=-1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative minimum distance was accepted")

    try:
        env.reset(
            layout_type="train",
            layout_index=0,
            seed=0,
            sample_start=True,
            min_start_goal_distance=100.0,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("infeasible minimum distance did not raise")

    print("PASS: minimum start-goal distance and infeasibility guard")

def test_goal_near(env):
    # sample_goal_near places a feasible goal inside the requested radius of
    # a sampled start, beyond the goal radius so the episode does not start
    # already finished; a radius at or below the goal radius is rejected.
    margin = 0.3
    max_distance = 1.0
    env.reset(layout_type="train", layout_index=0, seed=0)
    for _ in range(20):
        state = env.sample_random_state(margin=0.0, max_speed=1.0,
                                        min_speed=1.0)
        goal = env.sample_goal_near(state, max_distance=max_distance,
                                    margin=margin)
        assert goal is not None, "no near goal found"
        x, y = state[0], state[2]
        goal_x, goal_y = goal
        distance = np.hypot(x - goal_x, y - goal_y)
        assert distance > env.goal_radius, (distance, env.goal_radius)
        assert distance <= max_distance + 1e-9, distance
        assert env.width / 2 - abs(goal_y) - env.vehicle_radius >= margin - 1e-6
        assert goal_x - env.vehicle_radius - margin >= -1e-6
        assert env.length - goal_x - env.vehicle_radius - margin >= -1e-6
        for pillar_x, pillar_y, pillar_radius in env.pillars:
            assert np.hypot(goal_x - pillar_x, goal_y - pillar_y) >= (
                pillar_radius + env.vehicle_radius + margin - 1e-6
            ), f"near goal too close to pillar: {goal_x}, {goal_y}"

        # Reset onto the sampled state and goal: the episode must start
        # unfinished and running on the same layout.
        _, _ = env.reset(state=state, goal=goal, layout_type="fixed")
        np.testing.assert_allclose(env.goal, goal, atol=1e-9)
        np.testing.assert_allclose(
            env.dynamics.get_state(), state, atol=1e-6
        )
        assert not env.check_collision()

    try:
        env.sample_goal_near(state, max_distance=env.goal_radius)
    except ValueError:
        pass
    else:
        raise AssertionError("max_distance at the goal radius was accepted")

    print("PASS: near-goal sampling respects the radius and margins")

def test_variable_layouts():
    # Generated layouts carry their own width and pillar count. reset() must
    # adopt both before sampling the start and goal, so both are feasible for
    # the geometry the episode runs on, and step_batch must use each
    # environment's own width.
    from environment.layouts.generate_layouts import (
        MAX_CORRIDOR_WIDTH,
        MAX_PILLARS,
        MAX_RADIUS,
        MIN_CORRIDOR_WIDTH,
        MIN_PILLARS,
        MIN_RADIUS,
        generate_dataset,
    )

    layouts = generate_dataset(seed=0, num_layouts=20)
    counts = [len(layout["pillars"]) for layout in layouts]
    widths = [layout["width"] for layout in layouts]
    radii = [pillar[2] for layout in layouts for pillar in layout["pillars"]]
    assert all(MIN_PILLARS <= count <= MAX_PILLARS for count in counts), counts
    assert all(
        MIN_CORRIDOR_WIDTH <= width <= MAX_CORRIDOR_WIDTH for width in widths
    ), widths
    assert all(MIN_RADIUS <= radius <= MAX_RADIUS for radius in radii), radii

    layout_env = create_env("one")
    layout_env.train_layouts = layouts
    margin = 0.3
    for index, layout in enumerate(layouts):
        layout_env.reset(
            layout_type="train",
            layout_index=index,
            sample_start=True,
            start_margin=margin,
            start_max_speed=1.0,
            goal_margin=margin,
            min_start_goal_distance=2 * layout_env.goal_radius,
        )
        assert layout_env.width == layout["width"]
        np.testing.assert_allclose(layout_env.pillars, layout["pillars"])
        x, x_dot, y, y_dot = layout_env.dynamics.get_state()
        assert not layout_env.check_collision(), (index, x, y)
        assert layout_env.width / 2 - abs(y) - layout_env.vehicle_radius >= (
            margin - 1e-6
        )
        goal_x, goal_y = layout_env.goal
        assert layout_env.width / 2 - abs(goal_y) - layout_env.vehicle_radius >= (
            margin - 1e-6
        )
        for pillar_x, pillar_y, pillar_radius in layout_env.pillars:
            clearance = pillar_radius + layout_env.vehicle_radius + margin
            assert np.hypot(x - pillar_x, y - pillar_y) >= clearance - 1e-6
            assert np.hypot(goal_x - pillar_x, goal_y - pillar_y) >= (
                clearance - 1e-6
            )
    print("PASS: variable-width layouts sample compatible starts and goals")

    # Two environments with different widths in one batched step must match
    # their scalar step() results.
    assert layouts[0]["width"] != layouts[1]["width"]
    actions = np.array([[0.5, -0.25], [-0.4, 0.3]], dtype=np.float32)
    batch_envs = []
    scalar_results = []
    for index, action in zip((0, 1), actions):
        batch_env, scalar_env = create_env("one"), create_env("one")
        for env in (batch_env, scalar_env):
            env.train_layouts = layouts
            env.reset(
                layout_type="train",
                layout_index=index,
                state=[5.0, 0.0, 0.0, 0.0],
                goal=[20.0, 0.0],
            )
        batch_envs.append(batch_env)
        scalar_results.append(scalar_env.step(action))

    observations, rewards, terminated, truncated, info = step_batch(
        batch_envs, actions
    )
    for index, scalar in enumerate(scalar_results):
        scalar_obs, scalar_reward, scalar_term, scalar_trunc, scalar_info = scalar
        np.testing.assert_allclose(observations[index], scalar_obs, atol=1e-6)
        np.testing.assert_allclose(rewards[index], scalar_reward, atol=1e-6)
        assert bool(terminated[index]) == bool(scalar_term)
        assert bool(truncated[index]) == bool(scalar_trunc)
        np.testing.assert_allclose(
            info["next_constraint"][index],
            scalar_info["next_constraint"],
            atol=1e-6,
        )
    print("PASS: step_batch matches step across differing widths")

def test_constraint_and_reward_scale():
    # Isolate the wall constraint (no pillars) so the nearest lidar return is
    # exactly the distance to the top wall.
    clear_env = create_env("one")
    clear_env.pillars = np.zeros((0, 3), dtype=np.float32)
    clear_env.width = 5.0
    clear_env.length = 30.0
    clear_env._update_observation_scale()
    radius = clear_env.vehicle_radius

    # Clearance = width/2 - y. Inside the collision radius -> h > 0.
    for clearance, expected_positive in (
        (radius - 0.05, True),
        (radius + 0.05, False),
    ):
        state = [15.0, 0.0, clear_env.width / 2 - clearance, 0.0]
        clear_env.reset(layout_type="fixed", state=state, goal=[20.0, 0.0])
        lidar = clear_env.get_lidar()
        h = clear_env._get_lidar_constraint(lidar)
        np.testing.assert_allclose(h, radius - np.min(lidar), atol=1e-9)
        assert (h > 0.0) == expected_positive, (clearance, h)
    print("PASS: vehicle radius defines the constraint boundary")

    # The reward divisor must scale both step and goal rewards consistently.
    state = [10.0, 0.0, 0.0, 0.0]
    goal = [20.0, 0.0]
    world = [create_env("one", reward_scale=scale) for scale in (1.0, 50.0)]
    rewards = []
    for env in world:
        env.reset(layout_type="fixed", state=state, goal=goal)
        _, reward, *_ = env.step([0.5, 0.0])
        rewards.append(reward)
    np.testing.assert_allclose(rewards[0] / 50.0, rewards[1], atol=1e-9)
    print("PASS: reward_scale divides the environment reward")

def test_back_wall(env):
    # The corridor is closed at the goal end: touching or crossing
    # x = length - vehicle_radius collides, and lidar sees the wall.
    env.dynamics.state = np.array(
        [env.length - env.vehicle_radius + 0.01, 0.0, 0.0, 0.0]
    )
    assert env.check_collision(), "back wall collision not detected"

    env.dynamics.state = np.array(
        [env.length - env.vehicle_radius - 0.1, 0.0, 0.0, 0.0]
    )
    assert not env.check_collision(), "back wall false positive"

    env.reset(layout_type="fixed", state=[env.length - 1.0, 0.0, 0.0, 0.0],
              goal=[0.0, 0.0])
    lidar = env.get_lidar()
    # Ray 0 points straight down the +x axis: exactly 1.0 m to the wall.
    assert np.isfinite(lidar[0])
    np.testing.assert_allclose(lidar[0], env.length - (env.length - 1.0),
                               atol=1e-6)
    print("PASS: back wall collision and lidar")


def test_lidar_encoder():
    import torch
    from agents.encoders import LidarStateEncoder
    from agents.rcrl_td3 import Actor, Critic, MultiplierNetwork
    from agents.td3 import TD3

    num_rays = env.num_lidar_rays
    state_dim = env.observation_space.shape[0]
    encoder = LidarStateEncoder(state_dim, num_rays, 16)
    assert encoder(torch.randn(4, state_dim)).shape == (4, 20)
    # A single unbatched observation keeps its leading shape.
    assert encoder(torch.randn(state_dim)).shape == (20,)

    state = torch.randn(3, state_dim)
    action = torch.randn(3, 2)
    assert Actor(state_dim, 2, 1.0, encoder=LidarStateEncoder(state_dim, num_rays, 16))(
        state
    ).shape == (3, 2)
    assert Critic(state_dim, 2, encoder=LidarStateEncoder(state_dim, num_rays, 16))(
        state, action
    )[0].shape == (3, 1)
    assert MultiplierNetwork(
        state_dim, 100.0, 10.0, encoder=LidarStateEncoder(state_dim, num_rays, 16)
    )(state).shape == (3, 1)

    # Checkpoints record whether the encoder was used; the encoder lives
    # inside the agent so the config decides which state dict loads.
    assert any(
        key.startswith("encoder.") for key in TD3(
            state_dim, 2, 1.0, lidar_encoder=True, num_lidar_rays=num_rays
        ).actor.state_dict()
    )
    assert not any(
        key.startswith("encoder.") for key in TD3(
            state_dim, 2, 1.0, lidar_encoder=False
        ).actor.state_dict()
    )

    # Structure options: the dense bottleneck control and the extra exact
    # minimum features. Both keep the observation layout unchanged.
    import math

    linear = LidarStateEncoder(state_dim, num_rays, 16, kind="linear")
    assert linear(torch.randn(4, state_dim)).shape == (4, 20)
    with_min = LidarStateEncoder(state_dim, num_rays, 16, min_features=True)
    assert with_min.input_dim == 23
    assert with_min(torch.randn(4, state_dim)).shape == (4, 23)
    linear_min = LidarStateEncoder(
        state_dim, num_rays, 16, kind="linear", min_features=True
    )
    assert linear_min(torch.randn(4, state_dim)).shape == (4, 23)
    # The min-feature buffer exists only when used, so the state dict of the
    # original encoder is unchanged.
    assert "ray_angles" not in encoder.state_dict()
    assert "ray_angles" in with_min.state_dict()

    # The extra inputs are the exact minimum range and its unit bearing.
    lidar = torch.full((1, num_rays), 5.0)
    lidar[0, 7] = 0.4
    observation = torch.cat(
        (torch.zeros(1, 2), lidar, torch.zeros(1, 2)), dim=-1
    )
    encoded = with_min(observation)
    bearing = 2.0 * math.pi * 7 / num_rays
    assert torch.isclose(encoded[0, 18], torch.tensor(0.4))
    assert torch.isclose(encoded[0, 19], torch.tensor(math.cos(bearing)))
    assert torch.isclose(encoded[0, 20], torch.tensor(math.sin(bearing)))
    print("PASS: lidar encoder shapes, state-dict keys and structure options")


def test_actor_encoder_every_step():
    """The every-step flag moves the encoder but not the MLP head."""
    import torch
    from agents.rcrl_td3 import RCRLTD3
    from agents.td3 import TD3
    from utils.replay_buffer import (
        ReplayBufferRCRLTD3Torch,
        ReplayBufferTD3Torch,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = env.observation_space.shape[0]
    num_rays = env.num_lidar_rays
    batch = 64

    def snapshot(module):
        return {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }

    def unchanged(before, module):
        current = dict(module.named_parameters())
        return all(
            torch.equal(value, current[name].detach())
            for name, value in before.items()
        )

    def changed(before, module):
        current = dict(module.named_parameters())
        return any(
            not torch.equal(value, current[name].detach())
            for name, value in before.items()
        )

    td3 = TD3(
        state_dim, 2, 1.0, policy_freq=4, lidar_encoder=True,
        num_lidar_rays=num_rays, actor_encoder_every_step=True,
    )
    td3_buffer = ReplayBufferTD3Torch(state_dim, 2, max_size=1024, device=device)
    td3_buffer.add_batch(
        np.random.randn(batch, state_dim),
        np.random.randn(batch, 2),
        np.random.randn(batch, state_dim),
        np.random.randn(batch),
        np.ones(batch),
    )
    encoder_before = snapshot(td3.actor.encoder)
    head_before = snapshot(td3.actor.l1)
    td3.train(td3_buffer, batch_size=batch)  # update 1: non-policy step
    assert changed(encoder_before, td3.actor.encoder)
    assert unchanged(head_before, td3.actor.l1)
    head_before = snapshot(td3.actor.l1)
    for _ in range(3):
        td3.train(td3_buffer, batch_size=batch)  # update 4: policy step
    assert changed(head_before, td3.actor.l1)

    rcrl = RCRLTD3(
        state_dim, 2, 1.0, policy_freq=4, lidar_encoder=True,
        num_lidar_rays=num_rays, actor_encoder_every_step=True,
        initial_lambda=5.0,
    )
    rcrl_buffer = ReplayBufferRCRLTD3Torch(
        state_dim, 2, max_size=1024, device=device
    )
    rcrl_buffer.add_batch(
        np.random.randn(batch, state_dim),
        np.random.randn(batch, 2),
        np.random.randn(batch, state_dim),
        np.random.randn(batch),
        np.full((batch, 1), 0.5),
        np.full((batch, 1), 0.5),
        np.ones(batch),
    )
    encoder_before = snapshot(rcrl.actor.encoder)
    head_before = snapshot(rcrl.actor.l1)
    rcrl.train(rcrl_buffer, batch_size=batch)  # update 1: non-policy step
    assert changed(encoder_before, rcrl.actor.encoder)
    assert unchanged(head_before, rcrl.actor.l1)
    head_before = snapshot(rcrl.actor.l1)
    for _ in range(3):
        rcrl.train(rcrl_buffer, batch_size=batch)  # update 4: policy step
    assert changed(head_before, rcrl.actor.l1)
    print("PASS: actor encoder every-step updates")


def test_ray_encoder():
    """The ray encoder is dimension-agnostic, invariant and exact."""
    import math

    import torch
    from agents.encoders import LidarStateEncoder, RayEncoder
    from agents.rcrl_td3 import RCRLTD3
    from agents.td3 import TD3

    num_rays = env.num_lidar_rays
    state_dim = env.observation_space.shape[0]

    def observation(batch=4):
        # Observation-like ranges: velocity ~1 m/s, lidar in [0, 5] m,
        # goal offsets ~ several metres.
        state = torch.rand(batch, state_dim)
        state[:, :2] = 2.0 * state[:, :2] - 1.0
        state[:, 2:2 + num_rays] *= 5.0
        state[:, -2:] = 6.0 * state[:, -2:] - 3.0
        return state

    encoder = LidarStateEncoder(state_dim, num_rays, kind="ray")
    # d = 2, C = 16: inv 8 + vectors 8*2 + norms 8 + min 1 + u* 2 = 35.
    assert encoder.lidar.output_dim == 35
    assert encoder.input_dim == 4 + 35
    state = observation()
    assert encoder(state).shape == (4, 39)
    assert encoder(state[0]).shape == (39,)

    # Dropping the exact-min skip removes 1 + d inputs.
    no_min = LidarStateEncoder(
        state_dim, num_rays, kind="ray", ray_exact_min=False
    )
    assert no_min.input_dim == 39 - 3
    assert no_min(state).shape == (4, 36)

    # A three-dimensional sensor: d is inferred from the state size and the
    # ray directions are supplied, so the same module runs unchanged.
    directions3 = torch.randn(8, 3)
    directions3 = directions3 / directions3.norm(dim=-1, keepdim=True)
    encoder3 = LidarStateEncoder(
        2 * 3 + 8, 8, kind="ray", ray_directions=directions3
    )
    assert encoder3.input_dim == 2 * 3 + (8 + 8 * 3 + 8 + 4)
    assert encoder3.input_dim == 50
    assert encoder3(torch.rand(5, 14)).shape == (5, 50)
    try:
        RayEncoder(num_rays=8, spatial_dim=3)
    except ValueError:
        pass
    else:
        raise AssertionError("3-D ray encoder needs supplied directions")
    try:
        RayEncoder(num_rays=8, spatial_dim=2, channels=1)
    except ValueError:
        pass
    else:
        raise AssertionError("channels < 2 must be rejected")

    # Permuting the rays together with their directions leaves the output
    # unchanged: no angular ordering is used anywhere.
    ray = encoder.lidar
    lidar = torch.rand(2, num_rays) * 5.0
    permutation = torch.randperm(num_rays)
    permuted = RayEncoder(num_rays, 2, 16)
    permuted.load_state_dict(ray.state_dict())
    permuted.directions = ray.directions[permutation]
    torch.testing.assert_close(ray(lidar), permuted(lidar[:, permutation]))

    # Rotating the world rotates the vector block and the argmin direction,
    # while the invariant block, the vector norms and the minimum stay put.
    angle = 0.7
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    rotated = RayEncoder(num_rays, 2, 16)
    rotated.load_state_dict(ray.state_dict())
    rotated.directions = ray.directions @ rotation.T
    before = ray(lidar)
    after = rotated(lidar)
    invariant_dim = ray.invariant_dim
    vector_dim = ray.vector_dim
    vector_end = invariant_dim + vector_dim * 2
    norm_end = vector_end + vector_dim
    torch.testing.assert_close(
        before[:, :invariant_dim], after[:, :invariant_dim]
    )
    torch.testing.assert_close(
        before[:, vector_end:norm_end], after[:, vector_end:norm_end]
    )
    torch.testing.assert_close(before[:, norm_end], after[:, norm_end])
    before_vectors = before[:, invariant_dim:vector_end].reshape(-1, 2)
    after_vectors = after[:, invariant_dim:vector_end].reshape(-1, 2)
    torch.testing.assert_close(after_vectors, before_vectors @ rotation.T)
    torch.testing.assert_close(after[:, -2:], before[:, -2:] @ rotation.T)

    # The exact-min skip carries the minimum and its argmin direction.
    lidar = torch.full((1, num_rays), 5.0)
    lidar[0, 7] = 0.4
    observation_min = torch.cat(
        (torch.zeros(1, 2), lidar, torch.zeros(1, 2)), dim=-1
    )
    # Encoded state is [velocity(2), z(35), goal(2)]; z ends with
    # [min(1), u*(2)] before the goal block.
    encoded = encoder(observation_min)
    assert torch.isclose(encoded[0, -5], torch.tensor(0.4))
    torch.testing.assert_close(encoded[0, -4:-2], ray.directions[7])

    # Registered buffers: the ray encoder owns its ray geometry, and the
    # conv/linear kinds keep their original state dicts.
    keys = encoder.state_dict().keys()
    assert "lidar.per_ray.0.weight" in keys
    assert "lidar.directions" in keys
    assert "ray_angles" not in keys
    assert "lidar.net.0.weight" in LidarStateEncoder(
        state_dim, num_rays, 16
    ).state_dict()
    assert "lidar.net.0.weight" in LidarStateEncoder(
        state_dim, num_rays, 16, kind="linear"
    ).state_dict()

    # The agents build every network's encoder with the ray kind and the
    # actor-encoder machinery works because the encoder has parameters.
    td3 = TD3(
        state_dim, 2, 1.0, lidar_encoder=True, num_lidar_rays=num_rays,
        lidar_encoder_kind="ray",
    )
    assert td3.actor.l1.in_features == 39
    with torch.no_grad():
        td3.critic.encoder.lidar.per_ray[0].weight.add_(1.0)
    assert td3.warm_start_actor_encoder()
    torch.testing.assert_close(
        td3.actor.encoder.lidar.per_ray[0].weight,
        td3.critic.encoder.lidar.per_ray[0].weight,
    )

    rcrl = RCRLTD3(
        state_dim, 2, 1.0, lidar_encoder=True, num_lidar_rays=num_rays,
        lidar_encoder_kind="ray", safety_twin=True, initial_lambda=5.0,
        multiplier_type="network",
    )
    assert rcrl.actor.l1.in_features == 39
    assert rcrl.safety_critic.encoder.input_dim == 39
    assert rcrl.multiplier.encoder.input_dim == 39
    print("PASS: ray encoder shapes, symmetry, exactness and agents")


#test_render(env)
test_collisions(env)
test_lidar(env)
test_lidar_at(env)
test_gymnasium_env(env)
test_random_state_sampling(env)
test_randomized_reset(env)
test_min_start_goal_distance(env)
test_goal_near(env)
test_variable_layouts()
test_goal(env)
test_batch_matches_single()
test_constraint_and_reward_scale()
test_back_wall(env)
test_lidar_encoder()
test_actor_encoder_every_step()
test_ray_encoder()
#test_velocity_limit(dynamics)
test_reward_and_constraints(env)
