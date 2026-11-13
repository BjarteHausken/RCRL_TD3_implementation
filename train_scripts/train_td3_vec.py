"""Train plain TD3 with vectorized environments and a device-resident buffer.

Runs ``--num-envs`` independent corridor environments. Each vectorized step
collects one transition per environment and then performs one gradient
update per collected transition, so the update-to-data ratio and every
per-transition semantic match the sequential trainer
``train_scripts/train_td3.py``. Only the execution schedule differs:

- the actor runs once per vectorized step on the stacked observations
  instead of once per environment step;
- the replay buffer lives on the training device, so sampled batches never
  round-trip through the host;
- the per-step gradient updates run back-to-back, letting the GPU overlap
  kernel execution with launch overhead.

The episode regime is the same randomized start+goal sampling used by the
sequential trainer: no separate warm-up phase, exploration comes from the
per-step action noise.

Examples:
  python -m train_scripts.train_td3_vec --name td3_vec --num-envs 32
  ./train.sh td3-vec --name quick --max-timesteps 8 --eval-freq 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from agents.td3 import TD3, device
from environment import create_env, step_batch
from train_scripts.train_td3 import (
    episode_reset,
    evaluate_policy,
    set_seed,
)
from utils.training_log import TrainingLogger, evaluation_metrics
from utils import ReplayBufferTD3Torch


def random_train_reset(env, min_speed, goal_distance=None,
                       near_goal_fraction=0.0, seed=None):
    """Reset into a uniformly random free-corridor state with nonzero speed.

    Used during the random phase (``--random-train-fraction``): the state
    is drawn uniformly over the whole free corridor with a speed in
    ``[min_speed, dynamics.max_velocity]`` and a random direction, so the
    replay buffer sees every part of the map, including the fast
    near-obstacle states normal training rarely visits. The layout is
    selected before the state is sampled, so the state is valid for the
    geometry the episode runs on.

    ``near_goal_fraction`` splits the random phase between two goal
    regimes: with that probability the goal is placed within
    ``goal_distance`` of the start (between one goal radius and that
    distance, uniformly); otherwise the goal is sampled uniformly over the
    corridor. A nonpositive ``goal_distance`` keeps a uniformly random goal
    for every episode. This matches the phase in the RCRL vectorized
    trainer, so the baselines share the same replay distribution.
    """
    env.reset(layout_type="train", seed=seed)
    state = env.sample_random_state(
        margin=0.0,
        max_speed=env.dynamics.max_velocity,
        min_speed=min_speed,
    )
    goal = None
    if (goal_distance is not None and goal_distance > 0.0
            and near_goal_fraction > 0.0
            and env.np_random.random() < near_goal_fraction):
        goal = env.sample_goal_near(state, max_distance=goal_distance)
    if goal is None:
        return env.reset(state=state, layout_type="fixed")
    return env.reset(state=state, goal=goal, layout_type="fixed")


def main(args):
    set_seed(args.seed)

    envs = [
        create_env(
            args.layout_set,
            normalize_obs=args.normalize_obs,
            reward_scale=args.reward_scale,
            goal_radius=args.goal_radius,
            max_steps=args.max_steps,
        )
        for _ in range(args.num_envs)
    ]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = float(envs[0].action_space.high[0])
    if args.min_start_goal_distance is None:
        args.min_start_goal_distance = 2.0 * envs[0].goal_radius
        if args.goal_radius != 0.5:
            print(
                f"warning: --min-start-goal-distance defaults to "
                f"2 * goal_radius = {args.min_start_goal_distance:.1f} m; "
                "pass it explicitly to keep the previous episode distribution"
            )

    policy = TD3(
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
        grad_clip=args.grad_clip,
        critic_lr_final=args.critic_lr_final,
        actor_lr_final=args.actor_lr_final,
        # One update per collected transition, so the anneal spans the
        # same number of updates as env steps.
        anneal_steps=(
            args.max_timesteps if args.lr_decay == "linear" else None
        ),
        lidar_encoder=args.lidar_encoder,
        lidar_embed_dim=args.lidar_embed_dim,
        num_lidar_rays=envs[0].num_lidar_rays,
        actor_encoder_lr=args.actor_encoder_lr,
        actor_encoder_lr_final=args.actor_encoder_lr_final,
        lidar_encoder_kind=args.lidar_encoder_kind,
        lidar_encoder_min=args.lidar_encoder_min,
        actor_encoder_every_step=args.actor_encoder_every_step,
        lidar_ray_channels=args.lidar_ray_channels,
        lidar_ray_exact_min=args.lidar_ray_exact_min,
    )
    if args.random_reset_max_speed > envs[0].dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    if args.random_train_min_speed > envs[0].dynamics.max_velocity:
        raise ValueError(
            "--random-train-min-speed must not exceed the dynamics max_velocity"
        )
    if (args.random_train_goal_distance > 0.0
            and args.random_train_goal_distance <= envs[0].goal_radius):
        raise ValueError(
            "--random-train-goal-distance must exceed the goal radius "
            f"({envs[0].goal_radius:.2f} m) when nonzero, so the sampled "
            "episode does not start with the goal already reached"
        )
    replay_buffer = ReplayBufferTD3Torch(
        state_dim,
        action_dim,
        max_size=args.replay_size,
        device=device,
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "td3")
    config = vars(args) | {
        "agent": "td3",
        "trainer": "vectorized",
        "layout_set": args.layout_set,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": str(device),
        "dt": envs[0].dynamics.dt,
        "max_velocity": envs[0].dynamics.max_velocity,
        "corridor_length": envs[0].length,
        "corridor_width": envs[0].width,
        "vehicle_radius": envs[0].vehicle_radius,
        "lidar_range": envs[0].lidar_range,
        "num_lidar_rays": envs[0].num_lidar_rays,
        "max_acceleration": envs[0].max_acceleration,
        "goal_radius": envs[0].goal_radius,
        "max_steps": envs[0].max_steps,
        "reward_scale": envs[0].reward_scale,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    training_log = TrainingLogger(run_dir)
    training_log.log_run(
        config, trainer="vectorized", num_envs=args.num_envs
    )

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Device: {device}")
    print(f"Saving checkpoints to: {model_path}")

    # Per-environment RNGs are offset by index so the environments explore
    # decorrelated action/layout sequences.
    observations = np.empty((args.num_envs, state_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        env.action_space.seed(args.seed + i)
        # The first episode reset also seeds the environment RNG. The
        # random phase starts at step zero, so it owns that first reset
        # whenever it is enabled.
        if args.random_train_fraction > 0.0:
            observation, _ = random_train_reset(
                env,
                args.random_train_min_speed,
                goal_distance=args.random_train_goal_distance,
                near_goal_fraction=args.random_train_near_goal_fraction,
                seed=args.seed + i,
            )
        else:
            observation, _ = episode_reset(
                env,
                args.random_reset_margin,
                args.random_reset_max_speed,
                min_start_goal_distance=args.min_start_goal_distance,
                seed=args.seed + i,
            )
        observations[i] = observation

    # Per-step batched transition storage, filled by the stepping loop.
    states = np.empty((args.num_envs, state_dim), dtype=np.float32)
    actions = np.empty((args.num_envs, action_dim), dtype=np.float32)
    rewards = np.empty((args.num_envs, 1), dtype=np.float32)
    dones = np.empty((args.num_envs, 1), dtype=np.float32)

    # Per-environment episode tracking, plus aggregates since the last log.
    episode_reward = np.zeros(args.num_envs)
    episode_steps = np.zeros(args.num_envs, dtype=int)
    window_rewards = []
    window_lengths = []
    window_collisions = 0
    window_goals = 0

    env_steps_done = 0
    warm_start_done = args.warm_start_actor_encoder <= 0
    next_log = args.log_freq if args.log_freq > 0 else None
    next_eval = args.eval_freq if args.eval_freq > 0 else None
    next_save = args.save_freq if args.save_freq > 0 else None
    start_time = time.perf_counter()

    meta_steps = -(-args.max_timesteps // args.num_envs)  # ceil division
    try:
        for _ in range(meta_steps):
            # The first ``--random-train-fraction`` of the run fills the
            # replay buffer with fully random transitions; the rest uses
            # the usual actor plus exploration noise.
            random_phase = (
                env_steps_done
                < args.random_train_fraction * args.max_timesteps
            )
            if random_phase:
                actions[:] = np.random.uniform(
                    -max_action, max_action,
                    size=(args.num_envs, action_dim),
                )
            else:
                # One actor forward pass for all environments.
                obs_tensor = torch.as_tensor(
                    observations, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    policy_actions = policy.actor(obs_tensor).cpu().numpy()
                noise = np.random.normal(
                    loc=0.0,
                    scale=max_action * args.expl_noise,
                    size=(args.num_envs, action_dim),
                )
                actions[:] = np.clip(
                    policy_actions + noise, -max_action, max_action
                )

            states[:] = observations
            next_states, step_rewards, terminated, truncated, info = step_batch(
                envs, actions
            )
            rewards[:, 0] = step_rewards
            # Collisions and goals terminate; time-limit truncations keep
            # bootstrap targets, matching the sequential trainer.
            dones[:, 0] = terminated

            # Episode bookkeeping and resets stay per environment; only the
            # environment stepping itself is batched.
            for i, env in enumerate(envs):
                episode_reward[i] += step_rewards[i]
                episode_steps[i] += 1

                if terminated[i] or truncated[i]:
                    window_rewards.append(episode_reward[i])
                    window_lengths.append(episode_steps[i])
                    window_collisions += int(info["collision"][i])
                    window_goals += int(info["goal_reached"][i])
                    if random_phase:
                        observation, _ = random_train_reset(
                            env,
                            args.random_train_min_speed,
                            goal_distance=args.random_train_goal_distance,
                            near_goal_fraction=(
                                args.random_train_near_goal_fraction
                            ),
                        )
                    else:
                        observation, _ = episode_reset(
                            env,
                            args.random_reset_margin,
                            args.random_reset_max_speed,
                            min_start_goal_distance=(
                                args.min_start_goal_distance
                            ),
                        )
                    observations[i] = observation
                    episode_reward[i] = 0.0
                    episode_steps[i] = 0
                else:
                    observations[i] = next_states[i]

            replay_buffer.add_batch(
                states,
                actions,
                next_states,
                rewards,
                dones,
            )
            env_steps_done += args.num_envs

            # One update per collected transition, back-to-back so the GPU can
            # overlap kernel execution with launch overhead.
            if replay_buffer.size >= args.batch_size:
                for _ in range(args.num_envs):
                    policy.train(replay_buffer, batch_size=args.batch_size)

            if (
                not warm_start_done
                and env_steps_done >= args.warm_start_actor_encoder
            ):
                warm_start_done = True
                if policy.warm_start_actor_encoder():
                    print(
                        f"Step {env_steps_done}: copied the critic lidar "
                        f"encoder into the actor and its target"
                    )

            if next_log is not None and env_steps_done >= next_log:
                next_log += args.log_freq
                elapsed = time.perf_counter() - start_time
                window_count = len(window_rewards)
                message = (
                    f"Step {env_steps_done}: replay_size={replay_buffer.size}, "
                    f"{env_steps_done / elapsed:.0f} steps/s"
                )
                if window_count:
                    message += (
                        f" | episodes={window_count}, "
                        f"mean_reward={np.mean(window_rewards):.3f}, "
                        f"mean_length={np.mean(window_lengths):.1f}, "
                        f"collision_rate={window_collisions / window_count:.1%}, "
                        f"goal_rate={window_goals / window_count:.1%}"
                    )
                print(message)
                log_metrics = {
                    "replay_size": replay_buffer.size,
                    "steps_per_second": env_steps_done / elapsed,
                    "random_phase": int(random_phase),
                }
                if window_count:
                    log_metrics.update(
                        episodes=window_count,
                        mean_reward=float(np.mean(window_rewards)),
                        mean_length=float(np.mean(window_lengths)),
                        collision_rate=window_collisions / window_count,
                        goal_rate=window_goals / window_count,
                    )
                training_log.log("progress", env_steps_done, **log_metrics)
                window_rewards.clear()
                window_lengths.clear()
                window_collisions = 0
                window_goals = 0

            if next_save is not None and env_steps_done >= next_save:
                next_save += args.save_freq
                policy.save(model_path)
                print(f"Step {env_steps_done}: checkpoint saved to {model_path}")

            if next_eval is not None and env_steps_done >= next_eval:
                next_eval += args.eval_freq
                results = evaluate_policy(
                    policy,
                    args.layout_set,
                    args.normalize_obs,
                    episodes_per_layout=args.eval_episodes_per_layout,
                    random_reset_margin=args.random_reset_margin,
                    random_reset_max_speed=args.random_reset_max_speed,
                    min_start_goal_distance=args.min_start_goal_distance,
                    reward_scale=args.reward_scale,
                )
                print(
                    f"Evaluation at {env_steps_done}: "
                    f"mean_reward={results['mean_reward']:.3f}, "
                    f"mean_steps={results['mean_steps']:.1f}, "
                    f"success_rate={results['success_rate']:.1%} "
                    f"(95% CI {results['success_rate_ci'][0]:.1%}-"
                    f"{results['success_rate_ci'][1]:.1%}), "
                    f"collision_rate={results['collision_rate']:.1%}, "
                    f"margin_violation_rate={results['margin_violation_rate']:.1%}, "
                    f"margin_step_fraction={results['margin_step_fraction']:.2%}, "
                    f"mean_min_clearance={results['mean_min_clearance']:.3f} m"
                )
                training_log.log(
                    "eval", env_steps_done, **evaluation_metrics(results)
                )
                policy.save(model_path)
    except KeyboardInterrupt:
        policy.save(model_path)
        print(f"Interrupted. Checkpoint saved to: {model_path}")
        return
    finally:
        training_log.close()

    policy.save(model_path)
    print(f"Training complete. Model saved to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument("--output-dir", default="models", help="Checkpoint root")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Number of parallel environments; each vectorized step "
             "collects one transition and runs one update per environment",
    )
    parser.add_argument(
        "--layout-set",
        default="one",
        help="Layout set name; loads layouts/<name>_{train,test,validation}.npy",
    )
    parser.add_argument(
        "--normalize-obs",
        dest="normalize_obs",
        action="store_true",
        default=True,
        help="Scale observation components by their known ranges (default); "
             "checkpoints trained with --no-normalize-obs must be evaluated "
             "with --no-normalize-obs",
    )
    parser.add_argument(
        "--no-normalize-obs",
        dest="normalize_obs",
        action="store_false",
        help="Disable observation scaling",
    )
    parser.add_argument(
        "--lidar-encoder",
        dest="lidar_encoder",
        action="store_true",
        default=True,
        help="Encode the lidar block with a two-layer circular 1-D CNN before "
             "the MLP networks (default); evaluation reads the setting back "
             "from config.json",
    )
    parser.add_argument(
        "--no-lidar-encoder",
        dest="lidar_encoder",
        action="store_false",
        help="Feed the raw lidar rays to the MLPs (the pre-encoder behaviour)",
    )
    parser.add_argument(
        "--lidar-embed-dim",
        type=int,
        default=16,
        help="Number of angular bins the conv and linear lidar encoders "
             "produce (default 16); with 64 rays and stride 4 only 16 bins "
             "reproduce the ray count. Ignored by --lidar-encoder-kind ray, "
             "which uses --lidar-ray-channels",
    )
    parser.add_argument(
        "--actor-encoder-lr",
        type=float,
        default=None,
        help="Learning rate for the actor's lidar encoder, on its own Adam "
             "param group; default uses --actor-lr. The actor encoder only "
             "updates on policy steps, so this rate is paid once per "
             "--policy-freq steps while the critic encoder trains every step",
    )
    parser.add_argument(
        "--actor-encoder-lr-final",
        type=float,
        default=None,
        help="Final value of the linear anneal for --actor-encoder-lr; "
             "default uses --actor-lr-final",
    )
    parser.add_argument(
        "--warm-start-actor-encoder",
        type=int,
        default=0,
        metavar="STEPS",
        help="Copy the critic's lidar encoder into the actor and its target "
             "once at this many environment steps (0 disables); the critic "
             "encoder trains every step and is useful much earlier",
    )
    parser.add_argument(
        "--lidar-encoder-kind",
        choices=["conv", "linear", "ray"],
        default="conv",
        help="Structure of the lidar encoder: the circular 1-D CNN (default), "
             "a single dense layer over all rays (same bottleneck, no weight "
             "sharing, a control that separates the convolution prior from "
             "the compression), or the shared per-ray MLP with invariant and "
             "equivariant pooling, which uses no angular ordering and works "
             "for any ray count and spatial dimension",
    )
    parser.add_argument(
        "--lidar-encoder-min",
        dest="lidar_encoder_min",
        action="store_true",
        default=False,
        help="Append the exact minimum lidar range and the unit bearing of "
             "the closest ray (three extra MLP inputs) so the encoder cannot "
             "average away the statistic the constraint is built on",
    )
    parser.add_argument(
        "--no-lidar-encoder-min",
        dest="lidar_encoder_min",
        action="store_false",
        help="Do not append the exact minimum range and bearing (default)",
    )
    parser.add_argument(
        "--lidar-ray-channels",
        type=int,
        default=16,
        metavar="C",
        help="Width of the shared per-ray MLP in the ray encoder (default 16); "
             "half of the channels are pooled invariantly and half act as "
             "weights on the ray directions",
    )
    parser.add_argument(
        "--lidar-ray-exact-min",
        dest="lidar_ray_exact_min",
        action="store_true",
        default=True,
        help="Append the exact minimum range and the direction of its argmin "
             "ray after the pooled features (default); the ray encoder's "
             "learned pool cannot be guaranteed to preserve that statistic",
    )
    parser.add_argument(
        "--no-lidar-ray-exact-min",
        dest="lidar_ray_exact_min",
        action="store_false",
        help="Drop the exact minimum and argmin direction from the ray encoder",
    )
    parser.add_argument(
        "--actor-encoder-every-step",
        dest="actor_encoder_every_step",
        action="store_true",
        default=False,
        help="Train the actor's lidar encoder on every update: on non-policy "
             "steps backprop the actor loss into the encoder only, leaving "
             "the MLP head frozen until the next policy step",
    )
    parser.add_argument(
        "--no-actor-encoder-every-step",
        dest="actor_encoder_every_step",
        action="store_false",
        help="Update the actor encoder only on policy steps (default)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--expl-noise", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.1)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument(
        "--critic-lr-final",
        type=float,
        default=1e-6,
        help="Final critic learning rate of the linear anneal (paper: 1e-6)",
    )
    parser.add_argument(
        "--actor-lr-final",
        type=float,
        default=1e-6,
        help="Final actor learning rate of the linear anneal (paper: 1e-6)",
    )
    parser.add_argument(
        "--lr-decay",
        choices=["linear", "none"],
        default="linear",
        help="'linear' anneals every learning rate from its initial to its "
             "final value over --max-timesteps, as in the paper (Table 1); "
             "'none' keeps the initial rates constant",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global gradient norm for all networks; 0 disables clipping")
    parser.add_argument("--random-reset-margin", type=float, default=0.3,
                        help="Extra clearance (m) beyond the collision "
                             "boundary for sampled start and goal states")
    parser.add_argument("--random-reset-max-speed", type=float, default=1.0,
                        help="Maximum speed (m/s) of the sampled start velocity")
    parser.add_argument("--random-train-fraction", type=float, default=0.0,
                        help="Fraction of --max-timesteps whose training "
                             "episodes start from a uniformly sampled free-"
                             "corridor state with a nonzero speed and take "
                             "uniformly random actions. These transitions "
                             "fill the replay buffer with the fast "
                             "near-obstacle states normal training rarely "
                             "visits; 0 disables the phase (default)")
    parser.add_argument("--random-train-min-speed", type=float, default=1.0,
                        help="Minimum initial speed (m/s) of the states "
                             "sampled during --random-train-fraction; the "
                             "maximum is the dynamics' max_velocity "
                             "(default 1.0, guaranteeing nonzero velocity)")
    parser.add_argument("--random-train-goal-distance", type=float, default=1.0,
                        help="Maximum start-goal distance (m) of the "
                             "near-goal share of --random-train-fraction "
                             "episodes: the goal is placed uniformly "
                             "between one goal radius and this distance "
                             "from the sampled start. 0 disables the "
                             "near-goal share and samples every goal "
                             "uniformly over the corridor (default: 1.0)")
    parser.add_argument("--random-train-near-goal-fraction", type=float,
                        default=0.5,
                        help="Fraction of --random-train-fraction episodes "
                             "that get the near goal from "
                             "--random-train-goal-distance; the rest keep a "
                             "uniformly sampled goal (default: 0.5)")
    parser.add_argument("--goal-radius", type=float, default=0.5,
                        help="Radius (m) of the circular goal used while "
                             "training. Evaluation always uses the standard "
                             "0.5 m radius, so --goal-radius 0.3 trains on a "
                             "stricter target and is still scored at 0.5 m. "
                             "Also changes the default of "
                             "--min-start-goal-distance (2 * radius)")
    parser.add_argument("--min-start-goal-distance", type=float, default=None,
                        help="Minimum start-goal distance (m) for sampled "
                             "episodes; default is 2 * goal_radius (1.0 m). "
                             "Distances close to the corridor length may be "
                             "infeasible on some layouts, in which case reset "
                             "raises an error")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Fixed reward divisor used by the environment; "
                             "1.0 uses the unscaled reward. Match it across "
                             "agents so rewards are comparable")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="Maximum episode length N in environment steps "
                             "(paper: 360); episodes are truncated at this "
                             "length")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Set to 0 to disable evaluation")
    parser.add_argument("--save-freq", type=int, default=5_000,
                        help="Save a checkpoint every this many steps, "
                             "independent of evaluation; 0 disables "
                             "intermediate saves (default: 5000)")
    parser.add_argument("--eval-episodes-per-layout", type=int, default=20,
                        help="Episodes with sampled start/goal pairs to run "
                             "on each validation layout during evaluation")
    parser.add_argument("--log-freq", type=int, default=1_000,
                        help="Set to 0 to disable progress logging")
    args = parser.parse_args()

    if args.max_timesteps < 1:
        parser.error("--max-timesteps must be positive")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.batch_size < 1 or args.replay_size < args.batch_size:
        parser.error("--replay-size must be at least --batch-size, both positive")
    if args.policy_freq < 1:
        parser.error("--policy-freq must be positive")
    for name in ("critic_lr", "actor_lr", "critic_lr_final",
                 "actor_lr_final"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.random_reset_margin < 0:
        parser.error("--random-reset-margin must be nonnegative")
    if args.random_reset_max_speed <= 0:
        parser.error("--random-reset-max-speed must be positive")
    if not 0.0 <= args.random_train_fraction <= 1.0:
        parser.error("--random-train-fraction must be between 0 and 1")
    if args.random_train_min_speed <= 0:
        parser.error("--random-train-min-speed must be positive")
    if args.random_train_goal_distance < 0:
        parser.error("--random-train-goal-distance must be nonnegative")
    if not 0.0 <= args.random_train_near_goal_fraction <= 1.0:
        parser.error(
            "--random-train-near-goal-fraction must be between 0 and 1"
        )
    if args.goal_radius <= 0:
        parser.error("--goal-radius must be positive")
    if (args.min_start_goal_distance is not None
            and args.min_start_goal_distance < 0):
        parser.error("--min-start-goal-distance must be nonnegative")
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if args.eval_episodes_per_layout < 1:
        parser.error("--eval-episodes-per-layout must be positive")
    if args.save_freq < 0:
        parser.error("--save-freq must be nonnegative")
    if args.lidar_embed_dim < 1:
        parser.error("--lidar-embed-dim must be positive")
    if args.actor_encoder_lr is not None and args.actor_encoder_lr <= 0:
        parser.error("--actor-encoder-lr must be positive")
    if args.actor_encoder_lr_final is not None:
        if args.actor_encoder_lr is None:
            parser.error(
                "--actor-encoder-lr-final requires --actor-encoder-lr"
            )
        if args.actor_encoder_lr_final <= 0:
            parser.error("--actor-encoder-lr-final must be positive")
    if args.warm_start_actor_encoder < 0:
        parser.error("--warm-start-actor-encoder must be nonnegative")
    if (
        args.actor_encoder_lr is not None
        or args.warm_start_actor_encoder > 0
    ) and not args.lidar_encoder:
        parser.error(
            "the actor encoder options require the lidar encoder"
        )
    if args.lidar_ray_channels < 2:
        parser.error("--lidar-ray-channels must be at least 2")
    if (
        args.lidar_encoder_kind != "conv"
        or args.lidar_encoder_min
        or args.actor_encoder_every_step
    ) and not args.lidar_encoder:
        parser.error(
            "the lidar encoder kind, min features and every-step updates "
            "require --lidar-encoder"
        )
    if args.warm_start_actor_encoder > args.max_timesteps:
        parser.error(
            "--warm-start-actor-encoder must not exceed --max-timesteps"
        )
    return args


if __name__ == "__main__":
    main(parse_args())
