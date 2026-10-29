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
    env_a = create_env("one")
    env_b = create_env("one")
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


#test_render(env)
test_collisions(env)
test_lidar(env)
test_lidar_at(env)
test_gymnasium_env(env)
test_random_state_sampling(env)
test_randomized_reset(env)
test_variable_layouts()
test_goal(env)
test_batch_matches_single()
test_back_wall(env)
#test_velocity_limit(dynamics)
test_reward_and_constraints(env)
