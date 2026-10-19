import numpy as np
from environment.dynamics import DoubleIntegratorExactDiscretization
from environment.environment import CorridorEnvironment
from gymnasium.utils.env_checker import check_env

dynamics = DoubleIntegratorExactDiscretization(dt=0.1, max_velocity=5.0)
env = CorridorEnvironment(dynamics=dynamics, length=30.0, width=5.0)

env.pillars = np.array([(10.0, 0.0, 0.5)])
"""
state = env.reset()
print(state)

for i in range(60):
    state, collision = env.step(action=[1.0, 0.0])
    print(state)
    print("Collision:", collision)
"""

def test_collisions(env):

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
    env.reset([5.0, 2.0, 0.0, 0.5])
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
    env.reset([5.0, 0.0, 0.0, 0.0])
    lidar = env.get_lidar()
    print(lidar)
    print(lidar.shape)
    print(lidar.min())
    print(lidar.max())

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


#test_collisions(env)
#test_render(env)
#test_lidar(env)
test_gymnasium_env(env)
test_random_state_sampling(env)
test_goal(env)
#test_velocity_limit(dynamics)
#test_reward_and_constraints(env)
