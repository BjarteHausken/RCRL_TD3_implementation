import numpy as np
from double_integrator_sim.dynamics import DoubleIntegratorExactDiscretization
from double_integrator_sim.environment import CorridorEnvironment
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

    distance = env.ray_circle_intersection(
        origin,
        direction,
        center,
        radius
    )

    print("Hit pillar, ", distance)
    
    direction = np.array([0.0, 1.0])

    distance = env.ray_circle_intersection(
        origin,
        direction,
        center,
        radius
    )

    print("Miss pillar, ", distance)
    
    # Test wall ray intersection
    origin = np.array([5.0, 0.0])
    direction = np.array([0.0, 1.0])

    distance = env.ray_wall_intersection(
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

#test_collisions(env)
#test_render(env)
#test_lidar(env)
test_gymnasium_env(env)
#test_velocity_limit(dynamics)
#test_reward_and_constraints(env)
