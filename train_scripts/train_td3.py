"""Train the plain TD3 agent in the corridor environment.

Same randomized start+goal episode regime as train_rcrl_td3.py, but
without the safety critic and multiplier.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from agents.td3 import TD3
from environment import create_env
from evaluation_scripts.evaluation import evaluate_policy
from utils.training_log import TrainingLogger, evaluation_metrics
from utils import ReplayBuffer


def episode_reset(env, random_reset_margin, random_reset_max_speed,
                  min_start_goal_distance=None, seed=None):
    """Reset into a new randomized training episode in one call.

    ``reset`` selects the layout first and then samples a feasible start and
    goal on that layout, so both always satisfy the clearance margins for the
    layout the episode actually runs on. The goal is resampled until it is at
    least ``min_start_goal_distance`` from the start; when that is None the
    default is two goal radii.
    """
    if min_start_goal_distance is None:
        min_start_goal_distance = 2 * env.goal_radius
    return env.reset(
        layout_type="train",
        seed=seed,
        sample_start=True,
        start_margin=random_reset_margin,
        start_max_speed=random_reset_max_speed,
        goal_margin=random_reset_margin,
        min_start_goal_distance=min_start_goal_distance,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    env = create_env(
        args.layout_set,
        normalize_obs=args.normalize_obs,
        reward_scale=args.reward_scale,
        goal_radius=args.goal_radius,
        max_steps=args.max_steps,
    )
    env.action_space.seed(args.seed)
    if args.min_start_goal_distance is None:
        args.min_start_goal_distance = 2.0 * env.goal_radius
        if args.goal_radius != 0.5:
            print(
                f"warning: --min-start-goal-distance defaults to "
                f"2 * goal_radius = {args.min_start_goal_distance:.1f} m; "
                "pass it explicitly to keep the previous episode distribution"
            )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

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
        anneal_steps=(
            args.max_timesteps if args.lr_decay == "linear" else None
        ),
        lidar_encoder=args.lidar_encoder,
        lidar_embed_dim=args.lidar_embed_dim,
        num_lidar_rays=env.num_lidar_rays,
        actor_encoder_lr=args.actor_encoder_lr,
        actor_encoder_lr_final=args.actor_encoder_lr_final,
        lidar_encoder_kind=args.lidar_encoder_kind,
        lidar_encoder_min=args.lidar_encoder_min,
        actor_encoder_every_step=args.actor_encoder_every_step,
        lidar_ray_channels=args.lidar_ray_channels,
        lidar_ray_exact_min=args.lidar_ray_exact_min,
    )
    if args.random_reset_max_speed > env.dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args.replay_size,
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "td3")
    config = vars(args) | {
        "agent": "td3",
        "layout_set": args.layout_set,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "dt": env.dynamics.dt,
        "max_velocity": env.dynamics.max_velocity,
        "corridor_length": env.length,
        "corridor_width": env.width,
        "vehicle_radius": env.vehicle_radius,
        "lidar_range": env.lidar_range,
        "num_lidar_rays": env.num_lidar_rays,
        "max_acceleration": env.max_acceleration,
        "goal_radius": env.goal_radius,
        "max_steps": env.max_steps,
        "reward_scale": env.reward_scale,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    training_log = TrainingLogger(run_dir)
    training_log.log_run(config, trainer="sequential")

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Saving checkpoints to: {model_path}")

    # The first episode reset also seeds the environment RNG.
    state, _ = episode_reset(
        env,
        args.random_reset_margin,
        args.random_reset_max_speed,
        min_start_goal_distance=args.min_start_goal_distance,
        seed=args.seed,
    )
    episode_reward = 0.0
    episode_steps = 0
    episode_num = 0
    warm_start_done = args.warm_start_actor_encoder <= 0

    try:
        for timestep in range(args.max_timesteps):
            episode_steps += 1
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
            replay_buffer.add(state, action, next_state, reward, float(terminated))
            state = next_state

            if replay_buffer.size >= args.batch_size:
                policy.train(replay_buffer, batch_size=args.batch_size)

            if (
                not warm_start_done
                and timestep + 1 >= args.warm_start_actor_encoder
            ):
                warm_start_done = True
                if policy.warm_start_actor_encoder():
                    print(
                        f"Step {timestep + 1}: copied the critic lidar encoder "
                        f"into the actor and its target"
                    )

            if terminated or truncated:
                print(
                    f"Episode {episode_num:4d} | step {timestep + 1:7d} | "
                    f"length {episode_steps:4d} | reward {episode_reward:8.3f} | "
                    f"collision {info['collision']} | goal {info['goal_reached']}"
                )
                training_log.log(
                    "episode",
                    timestep + 1,
                    episode=episode_num,
                    reward=episode_reward,
                    length=episode_steps,
                    collision=bool(info["collision"]),
                    goal=bool(info["goal_reached"]),
                )
                state, _ = episode_reset(
                    env,
                    args.random_reset_margin,
                    args.random_reset_max_speed,
                    min_start_goal_distance=args.min_start_goal_distance,
                )
                episode_reward = 0.0
                episode_steps = 0
                episode_num += 1

            if args.log_freq > 0 and (timestep + 1) % args.log_freq == 0:
                print(f"Step {timestep + 1}: replay_size={replay_buffer.size}")
                training_log.log(
                    "progress",
                    timestep + 1,
                    replay_size=replay_buffer.size,
                )

            if args.save_freq > 0 and (timestep + 1) % args.save_freq == 0:
                policy.save(model_path)
                print(f"Step {timestep + 1}: checkpoint saved to {model_path}")

            if args.eval_freq > 0 and (timestep + 1) % args.eval_freq == 0:
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
                    f"Evaluation at {timestep + 1}: "
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
                    "eval", timestep + 1, **evaluation_metrics(results)
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
