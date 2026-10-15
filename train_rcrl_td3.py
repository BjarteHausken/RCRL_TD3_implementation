"""Train the reachability-constrained TD3 agent in the corridor environment."""

import argparse
import json
import os
import random

import numpy as np
import torch

from agents.rcrl_td3 import RCRLTD3
from agents.td3 import TD3
from double_integrator_sim.dynamics import DoubleIntegratorExactDiscretization
from double_integrator_sim.environment import CorridorEnvironment
from utils import ReplayBuffer, ReplayBufferRCRLTD3


def create_env(layout_set):
    dynamics = DoubleIntegratorExactDiscretization(dt=0.1, max_velocity=3.0)
    return CorridorEnvironment(
        dynamics=dynamics,
        length=30.0,
        width=5.0,
        max_acceleration=2.0,
        lidar_range=5.0,
        num_lidar_rays=64,
        layout_set=layout_set,
    )


def evaluate_policy(policy, layout_set):
    """Evaluate a deterministic policy once on every validation layout."""
    env = create_env(layout_set)
    total_reward = 0.0
    total_steps = 0
    successes = 0
    collisions = 0

    for layout_index in range(len(env.val_layouts)):
        state, _ = env.reset(layout_type="val", layout_index=layout_index)
        episode_reward = 0.0
        episode_steps = 0

        while True:
            action = policy.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            if terminated or truncated:
                break

        total_reward += episode_reward
        total_steps += episode_steps
        successes += int(info["goal_reached"])
        collisions += int(info["collision"])

    num_layouts = len(env.val_layouts)
    return {
        "mean_reward": total_reward / num_layouts,
        "mean_steps": total_steps / num_layouts,
        "success_rate": successes / num_layouts,
        "collision_rate": collisions / num_layouts,
    }


def reset_for_episode(
    env,
    reset_mode,
    random_reset_frac,
    random_reset_margin,
    random_reset_max_speed,
    random_phase,
):
    """Reset for a new episode; returns (observation, used_random_state).

    During the random-exploration phase every episode starts from a random
    safe state so the critics are trained over a broad slice of the feasible
    set. In the learning phase, ``fixed`` keeps the standard start and
    ``mixed`` additionally starts from a random state with probability
    ``random_reset_frac``.
    """
    use_random = random_phase or (
        reset_mode == "mixed"
        and env.np_random.uniform() < random_reset_frac
    )

    if use_random:
        state = env.sample_random_state(
            margin=random_reset_margin, max_speed=random_reset_max_speed
        )
        observation, _ = env.reset(layout_type="train", state=state)
    else:
        observation, _ = env.reset(layout_type="train")
    return observation, use_random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    env = create_env(args.layout_set)
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.agent == "rcrl_td3":
        policy = RCRLTD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            critic_lr=args.critic_lr,
            actor_lr=args.actor_lr,
            safety_lr=args.safety_lr,
            multiplier_lr=args.multiplier_lr,
            multiplier_freq=args.multiplier_freq,
            max_lambda=args.max_lambda,
            initial_lambda=args.initial_lambda,
        )
    else:
        policy = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
        )
    if args.random_reset_max_speed > env.dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    buffer_class = ReplayBufferRCRLTD3 if args.agent == "rcrl_td3" else ReplayBuffer
    replay_buffer = buffer_class(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args.replay_size,
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, args.agent)
    config = vars(args) | {
        "agent": args.agent,
        "layout_set": args.layout_set,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "dt": 0.1,
        "max_velocity": 3.0,
        "corridor_length": 30.0,
        "corridor_width": 5.0,
        "lidar_range": 5.0,
        "num_lidar_rays": 64,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Saving checkpoints to: {model_path}")

    # The first reset seeds the environment RNG; the episode start state is
    # chosen by reset_for_episode (random during the exploration phase).
    env.reset(layout_type="train", seed=args.seed)
    state, _ = reset_for_episode(
        env,
        args.reset_mode,
        args.random_reset_frac,
        args.random_reset_margin,
        args.random_reset_max_speed,
        random_phase=True,
    )
    warmup_announced = args.start_timesteps <= 0
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0

    for timestep in range(args.max_timesteps):
        episode_steps += 1
        random_phase = timestep < args.start_timesteps
        if random_phase:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state)
            noise = np.random.normal(
                loc=0.0,
                scale=max_action * args.expl_noise,
                size=action_dim,
            )
            action = np.clip(action + noise, -max_action, max_action)

        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Time-limit truncations retain bootstrap targets; collisions and goals do not.
        if args.agent == "rcrl_td3":
            # info["constraint"] is h(s) and info["next_constraint"] is h(s').
            replay_buffer.add(
                state,
                action,
                next_state,
                reward,
                info["constraint"],
                info["next_constraint"],
                float(terminated),
            )
        else:
            replay_buffer.add(state, action, next_state, reward, float(terminated))
        state = next_state

        if replay_buffer.size >= args.batch_size:
            if random_phase:
                # RCRL critic warmup: both critics learn from the broad random
                # coverage while the actor and multiplier stay frozen. Plain
                # TD3 only collects transitions during the random phase.
                if args.agent == "rcrl_td3":
                    policy.train_warmup(replay_buffer, batch_size=args.batch_size)
            else:
                policy.train(replay_buffer, batch_size=args.batch_size)

        if not random_phase and not warmup_announced:
            warmup_announced = True
            training_name = "full RCRL training" if args.agent == "rcrl_td3" else "TD3 training"
            print(
                f"Random exploration complete at step {timestep + 1}; "
                f"{training_name} started"
            )

        if terminated or truncated:
            print(
                f"Episode {episode_num:4d} | step {timestep + 1:7d} | "
                f"length {episode_steps:4d} | reward {episode_reward:8.3f} | "
                f"collision {info['collision']} | goal {info['goal_reached']}"
            )
            state, used_random = reset_for_episode(
                env,
                args.reset_mode,
                args.random_reset_frac,
                args.random_reset_margin,
                args.random_reset_max_speed,
                random_phase=(timestep + 1) < args.start_timesteps,
            )
            if used_random:
                x, _, y, _ = env.dynamics.get_state()
                print(f"  -> random start at ({x:.2f}, {y:.2f})")
            episode_reward = 0.0
            episode_steps = 0
            episode_num += 1

        if args.log_freq > 0 and (timestep + 1) % args.log_freq == 0:
            if args.agent == "rcrl_td3":
                with torch.no_grad():
                    multiplier_state = torch.as_tensor(
                        state.reshape(1, -1),
                        dtype=torch.float32,
                        device=next(policy.multiplier.parameters()).device,
                    )
                    mean_lambda = policy.multiplier(multiplier_state).item()
                print(
                    f"Step {timestep + 1}: replay_size={replay_buffer.size}, "
                    f"mean_lambda={mean_lambda:.4f}"
                )
            else:
                print(
                    f"Step {timestep + 1}: replay_size={replay_buffer.size}"
                )

        if args.eval_freq > 0 and (timestep + 1) % args.eval_freq == 0:
            results = evaluate_policy(policy, args.layout_set)
            print(
                f"Evaluation at {timestep + 1}: "
                f"mean_reward={results['mean_reward']:.3f}, "
                f"mean_steps={results['mean_steps']:.1f}, "
                f"success_rate={results['success_rate']:.1%}, "
                f"collision_rate={results['collision_rate']:.1%}"
            )
            policy.save(model_path)

    policy.save(model_path)
    print(f"Training complete. Model saved to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument(
        "--agent",
        choices=("rcrl_td3", "td3"),
        default="rcrl_td3",
        help="Agent to train: 'rcrl_td3' adds the reachability safety "
             "constraint, 'td3' trains the plain reward agent. Checkpoints "
             "are saved under models/<name>/<agent>/ and load with the "
             "matching --agent in the evaluate scripts",
    )
    parser.add_argument("--output-dir", default="models", help="Checkpoint root")
    parser.add_argument(
        "--layout-set",
        default="one",
        help="Layout set name; loads layouts/<name>_{train,test,validation}.npy",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=30_000)
    parser.add_argument("--start-timesteps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--expl-noise", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=2)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--safety-lr", type=float, default=1e-4)
    parser.add_argument("--multiplier-lr", type=float, default=1e-6)
    parser.add_argument("--multiplier-freq", type=int, default=6)
    parser.add_argument("--max-lambda", type=float, default=100.0)
    parser.add_argument("--initial-lambda", type=float, default=0.1)
    parser.add_argument(
        "--reset-mode",
        choices=["fixed", "mixed"],
        default="fixed",
        help="Learning-phase episode start: 'fixed' keeps the standard start; "
             "'mixed' additionally starts each episode from a random safe "
             "state with probability --random-reset-frac. The random-"
             "exploration phase always uses random starts for coverage",
    )
    parser.add_argument("--random-reset-frac", type=float, default=0.15,
                        help="Probability of a random start in 'mixed' mode")
    parser.add_argument("--random-reset-margin", type=float, default=0.3,
                        help="Extra clearance (m) beyond the collision "
                             "boundary for random start states")
    parser.add_argument("--random-reset-max-speed", type=float, default=1.0,
                        help="Maximum speed (m/s) of the random initial velocity")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Set to 0 to disable evaluation")
    parser.add_argument("--log-freq", type=int, default=1_000,
                        help="Set to 0 to disable progress logging")
    args = parser.parse_args()

    if args.max_timesteps < 1:
        parser.error("--max-timesteps must be positive")
    if not 0 <= args.start_timesteps <= args.max_timesteps:
        parser.error("--start-timesteps must be between 0 and --max-timesteps")
    if args.batch_size < 1 or args.replay_size < args.batch_size:
        parser.error("--replay-size must be at least --batch-size, both positive")
    if args.policy_freq < 1:
        parser.error("--policy-freq must be positive")
    if args.agent == "rcrl_td3":
        if args.multiplier_freq < args.policy_freq:
            parser.error("--multiplier-freq must be at least --policy-freq")
        if args.max_lambda <= 0 or not 0 < args.initial_lambda <= args.max_lambda:
            parser.error("--initial-lambda must be in (0, --max-lambda]")
    if not 0 <= args.random_reset_frac <= 1:
        parser.error("--random-reset-frac must be in [0, 1]")
    if args.random_reset_margin < 0:
        parser.error("--random-reset-margin must be nonnegative")
    if args.random_reset_max_speed <= 0:
        parser.error("--random-reset-max-speed must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
