import numpy as np

from double_integrator_sim.dynamics import DoubleIntegratorExactDiscretization
from double_integrator_sim.environment import CorridorEnvironment
from agents.td3 import TD3
from utils import ReplayBuffer

dynamics = DoubleIntegratorExactDiscretization(dt=0.1, max_velocity=5.0)
env = CorridorEnvironment(dynamics=dynamics)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(state_dim, action_dim, max_action)

replay_buffer = ReplayBuffer(state_dim, action_dim)

max_timesteps = 100_000
start_timesteps = 10_000
batch_size = 256
expl_noise = 0.1

state, info = env.reset()

episode_reward = 0
episode_steps = 0
episode_num = 0

for t in range(max_timesteps):
    episode_steps += 1

    # Select action
    if t < start_timesteps:
        action = env.action_space.sample()

    else:
        action = policy.select_action(state)
        action = (action + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(-max_action, max_action)
    # Environment
    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    # Store transition
    replay_buffer.add(
        state,
        action,
        next_state,
        reward,
        float(terminated)
    )

    state = next_state
    episode_reward += reward

    # Train after initial exploration
    if t >= start_timesteps:
        policy.train(
        replay_buffer,
        batch_size
        )

    # End episode
    if done:
        print(
            f"Episode {episode_num} | "
            f"Steps: {episode_steps} | "
            f"Reward: {episode_reward:.2f} | "
            f"Collision: {info['collision']} | "
            f"Goal: {info['goal_reached']}"
        )

        state, info = env.reset()

        episode_reward = 0
        episode_steps = 0
        episode_num += 1
