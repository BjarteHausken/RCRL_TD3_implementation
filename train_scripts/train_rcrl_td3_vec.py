"""Train RCRL-TD3 with vectorized environments and a device-resident buffer.

Runs ``--num-envs`` independent corridor environments. Each vectorized step
collects one transition per environment and then performs one gradient
update per collected transition, so the update-to-data ratio and every
per-transition semantic match the sequential trainer
``train_scripts/train_rcrl_td3.py``. Only the execution schedule differs:

- the actor runs once per vectorized step on the stacked observations
  instead of once per environment step;
- the replay buffer lives on the training device, so sampled batches never
  round-trip through the host;
- the per-step gradient updates run back-to-back, letting the GPU overlap
  kernel execution with launch overhead.

This amortization is what makes GPU training worthwhile for the small
(3x256) networks; the sequential trainer pays the full per-step overhead
once per environment step.

Examples:
  python -m train_scripts.train_rcrl_td3_vec --name rcrl_vec --num-envs 32
  ./train.sh rcrl-vec --name quick --max-timesteps 8 --eval-freq 0
"""

import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from agents.rcrl_td3 import RCRLTD3, device
from environment import create_env, step_batch
from train_scripts.train_rcrl_td3 import (
    episode_reset,
    evaluate_policy,
    set_seed,
)
from utils.training_log import TrainingLogger, evaluation_metrics, lambda_metrics
from utils import ReplayBufferRCRLTD3Torch


def main(args):
    # Treat SIGTERM like Ctrl-C so a killed run still writes a checkpoint.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    set_seed(args.seed)

    envs = [
        create_env(
            args.layout_set,
            normalize_obs=args.normalize_obs,
            reward_scale=args.reward_scale,
            max_steps=args.max_steps,
        )
        for _ in range(args.num_envs)
    ]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = float(envs[0].action_space.high[0])
    if args.min_start_goal_distance is None:
        args.min_start_goal_distance = 2.0 * envs[0].goal_radius

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
        grad_clip=args.grad_clip,
        multiplier_lr=args.multiplier_lr,
        critic_lr_final=args.critic_lr_final,
        actor_lr_final=args.actor_lr_final,
        safety_lr_final=args.safety_lr_final,
        multiplier_lr_final=args.multiplier_lr_final,
        # One update per collected transition, so the anneal spans the
        # same number of updates as env steps.
        anneal_steps=(
            args.max_timesteps if args.lr_decay == "linear" else None
        ),
        multiplier_freq=args.multiplier_freq,
        max_lambda=None if args.max_lambda <= 0 else args.max_lambda,
        initial_lambda=args.initial_lambda,
        min_lambda=args.min_lambda,
        multiplier_type=args.multiplier,
        safety_updates=args.safety_updates,
        safety_batch_size=args.safety_batch_size,
        safety_mix=args.safety_mix,
        safety_mix_warmup=args.safety_mix_warmup,
        safety_twin=args.safety_twin,
        safety_aggregation=args.safety_aggregation,
    )
    if args.random_reset_max_speed > envs[0].dynamics.max_velocity:
        raise ValueError(
            "--random-reset-max-speed must not exceed the dynamics max_velocity"
        )
    replay_buffer = ReplayBufferRCRLTD3Torch(
        state_dim,
        action_dim,
        max_size=args.replay_size,
        device=device,
        # The sub-buffer only exists when the mix is enabled, so the
        # default runs keep the original memory footprint.
        boundary_threshold=(
            args.safety_mix_threshold if args.safety_mix > 0 else None
        ),
    )

    run_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "rcrl_td3")
    config = vars(args) | {
        "agent": "rcrl_td3",
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
        "multiplier_parameterization": "softplus",
        # None records an uncapped multiplier, which is what evaluation and
        # the agent receive; the CLI's 0 is only a sentinel for "no cap".
        "max_lambda": None if args.max_lambda <= 0 else args.max_lambda,
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
        # The first episode reset also seeds the environment RNG.
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
    constraints = np.empty((args.num_envs, 1), dtype=np.float32)
    next_constraints = np.empty((args.num_envs, 1), dtype=np.float32)
    dones = np.empty((args.num_envs, 1), dtype=np.float32)

    # Per-environment episode tracking, plus aggregates since the last log.
    episode_reward = np.zeros(args.num_envs)
    episode_steps = np.zeros(args.num_envs, dtype=int)
    episode_margin_steps = np.zeros(args.num_envs, dtype=int)
    episode_max_h = np.full(args.num_envs, -np.inf)
    window_rewards = []
    window_lengths = []
    window_collisions = 0
    window_goals = 0
    window_margin_violations = 0
    window_violation_steps = 0
    window_steps = 0
    window_critic_losses = []
    window_safety_losses = []

    env_steps_done = 0
    next_log = args.log_freq if args.log_freq > 0 else None
    next_eval = args.eval_freq if args.eval_freq > 0 else None
    next_save = args.save_freq if args.save_freq > 0 else None
    start_time = time.perf_counter()

    meta_steps = -(-args.max_timesteps // args.num_envs)  # ceil division
    try:
        for _ in range(meta_steps):
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
            actions[:] = np.clip(policy_actions + noise, -max_action, max_action)

            states[:] = observations
            next_states, step_rewards, terminated, truncated, info = step_batch(
                envs, actions
            )
            rewards[:, 0] = step_rewards
            constraints[:, 0] = info["constraint"]
            next_constraints[:, 0] = info["next_constraint"]
            # Collisions and goals terminate; time-limit truncations keep
            # bootstrap targets, matching the sequential trainer.
            dones[:, 0] = terminated

            # Episode bookkeeping and resets stay per environment; only the
            # environment stepping itself is batched.
            for i, env in enumerate(envs):
                episode_reward[i] += step_rewards[i]
                episode_steps[i] += 1
                episode_margin_steps[i] += int(info["constraint"][i] > 0.0)
                episode_max_h[i] = max(
                    episode_max_h[i],
                    info["constraint"][i],
                    info["next_constraint"][i],
                )

                if terminated[i] or truncated[i]:
                    window_rewards.append(episode_reward[i])
                    window_lengths.append(episode_steps[i])
                    window_collisions += int(info["collision"][i])
                    window_goals += int(info["goal_reached"][i])
                    window_margin_violations += int(episode_max_h[i] > 0.0)
                    window_violation_steps += int(episode_margin_steps[i])
                    window_steps += int(episode_steps[i])
                    observation, _ = episode_reset(
                        env,
                        args.random_reset_margin,
                        args.random_reset_max_speed,
                        min_start_goal_distance=args.min_start_goal_distance,
                    )
                    observations[i] = observation
                    episode_reward[i] = 0.0
                    episode_steps[i] = 0
                    episode_margin_steps[i] = 0
                    episode_max_h[i] = -np.inf
                else:
                    observations[i] = next_states[i]

            replay_buffer.add_batch(
                states,
                actions,
                next_states,
                rewards,
                constraints,
                next_constraints,
                dones,
            )
            env_steps_done += args.num_envs

            # One update per collected transition, back-to-back so the GPU can
            # overlap kernel execution with launch overhead.
            if replay_buffer.size >= args.batch_size:
                for _ in range(args.num_envs):
                    metrics = policy.train(
                        replay_buffer, batch_size=args.batch_size
                    )
                    if args.log_freq > 0:
                        window_critic_losses.append(metrics["critic_loss"])
                        window_safety_losses.append(metrics["safety_loss"])

            if next_log is not None and env_steps_done >= next_log:
                next_log += args.log_freq
                elapsed = time.perf_counter() - start_time
                lambda_stats = policy.lambda_stats(observations)
                mean_h_pred = policy.policy_safety_value(observations)
                window_count = len(window_rewards)
                message = (
                    f"Step {env_steps_done}: replay_size={replay_buffer.size}, "
                    f"mean_lambda={lambda_stats['mean']:.4f}, "
                    f"max_lambda={lambda_stats['max']:.4f}, "
                    f"mean_h_pred={mean_h_pred:+.4f}"
                )
                if "cap_fraction" in lambda_stats:
                    message += (
                        f", lambda_at_cap={lambda_stats['cap_fraction']:.1%}, "
                        f"lambda_at_floor={lambda_stats['floor_fraction']:.1%}"
                    )
                message += f", {env_steps_done / elapsed:.0f} steps/s"
                if window_critic_losses:
                    message += (
                        f", critic_loss={np.mean(window_critic_losses):.4f}"
                        f", safety_loss={np.mean(window_safety_losses):.4f}"
                    )
                if window_count:
                    message += (
                        f" | episodes={window_count}, "
                        f"mean_reward={np.mean(window_rewards):.3f}, "
                        f"mean_length={np.mean(window_lengths):.1f}, "
                        f"collision_rate={window_collisions / window_count:.1%}, "
                        f"goal_rate={window_goals / window_count:.1%}, "
                        f"margin_violation_rate="
                        f"{window_margin_violations / window_count:.1%}, "
                        f"margin_step_fraction="
                        f"{window_violation_steps / max(window_steps, 1):.2%}"
                    )
                print(message)
                log_metrics = {
                    "replay_size": replay_buffer.size,
                    "mean_h_pred": mean_h_pred,
                    "steps_per_second": env_steps_done / elapsed,
                }
                log_metrics.update(lambda_metrics(lambda_stats))
                if window_critic_losses:
                    log_metrics["critic_loss"] = float(
                        np.mean(window_critic_losses)
                    )
                    log_metrics["safety_loss"] = float(
                        np.mean(window_safety_losses)
                    )
                if window_count:
                    log_metrics.update(
                        episodes=window_count,
                        mean_reward=float(np.mean(window_rewards)),
                        mean_length=float(np.mean(window_lengths)),
                        collision_rate=window_collisions / window_count,
                        goal_rate=window_goals / window_count,
                        margin_violation_rate=(
                            window_margin_violations / window_count
                        ),
                        margin_step_fraction=(
                            window_violation_steps / max(window_steps, 1)
                        ),
                    )
                training_log.log("progress", env_steps_done, **log_metrics)
                window_rewards.clear()
                window_lengths.clear()
                window_collisions = 0
                window_goals = 0
                window_margin_violations = 0
                window_violation_steps = 0
                window_steps = 0
                window_critic_losses.clear()
                window_safety_losses.clear()

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
    parser.add_argument("--safety-lr", type=float, default=1e-4)
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
        "--safety-lr-final",
        type=float,
        default=1e-6,
        help="Final safety-critic learning rate of the linear anneal "
             "(paper: 1e-6)",
    )
    parser.add_argument(
        "--lr-decay",
        choices=["linear", "none"],
        default="linear",
        help="'linear' anneals every learning rate from its initial to its "
             "final value over --max-timesteps, as in the paper (Table 1); "
             "'none' keeps the initial rates constant",
    )
    parser.add_argument(
        "--safety-updates",
        type=int,
        default=1,
        help="Safety-critic gradient steps per training update; any extra "
             "steps draw fresh replay batches (default: 1)",
    )
    parser.add_argument(
        "--safety-batch-size",
        type=int,
        default=None,
        help="Batch size for safety-critic steps (default: --batch-size); a "
             "different size also draws a separate safety batch",
    )
    parser.add_argument(
        "--safety-mix",
        type=float,
        default=0.0,
        help="Fraction of each safety-critic batch drawn from a sub-buffer "
             "of near-boundary transitions (h or next_h within "
             "--safety-mix-threshold of the constraint, violations "
             "included). The fraction ramps up as that sub-buffer fills "
             "past --safety-mix-warmup transitions. 0 (default) keeps "
             "plain uniform sampling",
    )
    parser.add_argument(
        "--safety-mix-threshold",
        type=float,
        default=0.3,
        help="Constraint value below which a transition counts as "
             "near-boundary for --safety-mix (h = vehicle_radius - "
             "min_lidar, so 0.3 keeps clearances within 0.3 m of the "
             "collision boundary plus every violation)",
    )
    parser.add_argument(
        "--safety-mix-warmup",
        type=int,
        default=10_000,
        help="Boundary-sub-buffer fill level at which --safety-mix reaches "
             "its full value; below it the mix ramps up linearly from zero",
    )
    parser.add_argument(
        "--safety-twin",
        dest="safety_twin",
        action="store_true",
        default=True,
        help="Train two safety critics and bootstrap the reachability "
             "target from the min of the two target heads, mirroring "
             "TD3's twin reward critics to damp overestimation; the "
             "actor, multiplier, and logging use head 1 (default). The "
             "setting is recorded in config.json and the checkpoint "
             "format changes",
    )
    parser.add_argument(
        "--no-safety-twin",
        dest="safety_twin",
        action="store_false",
        help="Train a single safety critic (the pre-twin behaviour)",
    )
    parser.add_argument(
        "--safety-aggregation",
        choices=["min", "mean"],
        default="min",
        help="How the twin safety-critic target heads are combined: min "
             "(TD3-style overestimation damping, default) or mean "
             "(average the two heads, the variant used by the earlier "
             "_avg/_v2 runs)",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global gradient norm for all networks; 0 disables clipping")
    parser.add_argument("--multiplier-lr", type=float, default=6e-7,
                        help="Statewise dual-ascent learning rate (network "
                             "mode). The paper's value is 6e-7 annealed to "
                             "1e-7; with the softplus multiplier a step moves "
                             "lambda by roughly that amount, so short runs "
                             "need a proportionally larger value")
    parser.add_argument("--multiplier-lr-final", type=float, default=1e-7,
                        help="Final multiplier learning rate of the linear "
                             "anneal (paper: 1e-7)")
    parser.add_argument("--multiplier-freq", type=int, default=12)
    parser.add_argument(
        "--multiplier",
        choices=["network", "scalar"],
        default="network",
        help="'network' uses a statewise lambda(s) network (RCRL); 'scalar' "
             "uses a single fixed Lagrange multiplier for all states, set by "
             "--initial-lambda and not updated during training",
    )
    parser.add_argument("--max-lambda", type=float, default=0.0,
                        help="Optional hard cap on lambda(s). The paper's "
                             "reference implementation leaves the softplus "
                             "multiplier unbounded and 0 (the default) "
                             "disables the cap. To cap it, scale the bound "
                             "with the reward divisor: 100 at "
                             "--reward-scale 1.0, 2.0 at --reward-scale 50")
    parser.add_argument("--initial-lambda", type=float, default=10.0,
                        help="Initial lambda. With the default "
                             "--reward-scale 1.0 this keeps the safety term "
                             "comparable to the reward term; at "
                             "--reward-scale 50 use 0.1 instead")
    parser.add_argument("--min-lambda", type=float, default=0.0,
                        help="Floor of the statewise lambda(s) network. The "
                             "dual drives lambda to zero on a safe-dominated "
                             "visited distribution, switching the constraint "
                             "off; a positive floor keeps a minimum safety "
                             "pressure. 0 reproduces the pure dual")
    parser.add_argument("--random-reset-margin", type=float, default=0.3,
                        help="Extra clearance (m) beyond the collision "
                             "boundary for sampled start and goal states")
    parser.add_argument("--random-reset-max-speed", type=float, default=1.0,
                        help="Maximum speed (m/s) of the sampled start velocity")
    parser.add_argument("--min-start-goal-distance", type=float, default=None,
                        help="Minimum start-goal distance (m) for sampled "
                             "episodes; default is 2 * goal_radius (1.0 m). "
                             "Distances close to the corridor length may be "
                             "infeasible on some layouts, in which case reset "
                             "raises an error")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Fixed reward divisor used by the environment; "
                             "1.0 uses the unscaled reward, while larger "
                             "values shrink returns to keep reward and safety "
                             "terms comparable in the actor loss")
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
    if args.safety_updates < 1:
        parser.error("--safety-updates must be at least 1")
    if args.safety_batch_size is not None and args.safety_batch_size < 1:
        parser.error("--safety-batch-size must be positive")
    if not 0.0 <= args.safety_mix <= 1.0:
        parser.error("--safety-mix must be between 0 and 1")
    if args.safety_mix_threshold < 0:
        parser.error("--safety-mix-threshold must be nonnegative")
    if args.safety_mix_warmup < 1:
        parser.error("--safety-mix-warmup must be positive")
    if args.policy_freq < 1:
        parser.error("--policy-freq must be positive")
    if args.multiplier_freq < args.policy_freq:
        parser.error("--multiplier-freq must be at least --policy-freq")
    if args.max_lambda < 0 or not 0 < args.initial_lambda:
        parser.error(
            "--max-lambda must be nonnegative (0 disables the cap) and "
            "--initial-lambda must be positive"
        )
    if args.min_lambda < 0:
        parser.error("--min-lambda must be nonnegative")
    if args.multiplier == "network" and not (
        args.min_lambda < args.initial_lambda
    ):
        parser.error("network mode requires --min-lambda < --initial-lambda")
    if (
        args.multiplier == "network"
        and args.max_lambda > 0
        and not args.initial_lambda < args.max_lambda
    ):
        parser.error("network mode requires --initial-lambda < --max-lambda")
    for name in ("critic_lr", "actor_lr", "safety_lr", "multiplier_lr",
                 "critic_lr_final", "actor_lr_final", "safety_lr_final",
                 "multiplier_lr_final"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.random_reset_margin < 0:
        parser.error("--random-reset-margin must be nonnegative")
    if args.random_reset_max_speed <= 0:
        parser.error("--random-reset-max-speed must be positive")
    if (args.min_start_goal_distance is not None
            and args.min_start_goal_distance < 0):
        parser.error("--min-start-goal-distance must be nonnegative")
    if args.reward_scale <= 0:
        parser.error("--reward-scale must be positive")
    if args.eval_episodes_per_layout < 1:
        parser.error("--eval-episodes-per-layout must be positive")
    if args.save_freq < 0:
        parser.error("--save-freq must be nonnegative")
    return args


if __name__ == "__main__":
    main(parse_args())
