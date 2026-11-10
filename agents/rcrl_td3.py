"""TD3 with the reachability constraint and statewise dual update from RCRL."""

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import LidarStateEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, encoder=None):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.input_dim if encoder is not None else state_dim
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def _encode_state(self, state):
        return state if self.encoder is None else self.encoder(state)

    def forward(self, state):
        action = F.elu(self.l1(self._encode_state(state)))
        action = F.elu(self.l2(action))
        return self.max_action * torch.tanh(self.l3(action))


class Critic(nn.Module):
    """Twin reward critic used by TD3."""

    def __init__(self, state_dim, action_dim, encoder=None):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.input_dim if encoder is not None else state_dim
        self.l1 = nn.Linear(input_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(input_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def _state_action(self, state, action):
        state = state if self.encoder is None else self.encoder(state)
        return torch.cat([state, action], dim=1)

    def forward(self, state, action):
        state_action = self._state_action(state, action)
        q1 = F.elu(self.l1(state_action))
        q1 = F.elu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.elu(self.l4(state_action))
        q2 = F.elu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        state_action = self._state_action(state, action)
        q1 = F.elu(self.l1(state_action))
        q1 = F.elu(self.l2(q1))
        return self.l3(q1)


class SafetyCritic(nn.Module):
    """Single RCRL safety-value critic, where positive values are unsafe."""

    def __init__(self, state_dim, action_dim, encoder=None):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.input_dim if encoder is not None else state_dim
        self.l1 = nn.Linear(input_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def _state_action(self, state, action):
        state = state if self.encoder is None else self.encoder(state)
        return torch.cat([state, action], dim=1)

    def forward(self, state, action):
        state_action = self._state_action(state, action)
        value = F.elu(self.l1(state_action))
        value = F.elu(self.l2(value))
        return self.l3(value)


class SafetyCriticTwin(nn.Module):
    """Twin RCRL safety critics, where positive values are unsafe.

    Both heads are trained against the same reachability target and the
    Bellman target bootstraps from the min of the two target heads,
    mirroring TD3's min over the twin reward critics. The observed failure
    mode of the single critic is overestimation (an over-conservative
    feasible set), which the min damps the way TD3 damps Q overestimation.
    Head 1 is the value used by the actor loss, the multiplier, and
    logging, matching how the reward side uses ``Critic.q1``.
    """

    def __init__(self, state_dim, action_dim, encoder=None):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.input_dim if encoder is not None else state_dim
        self.l1 = nn.Linear(input_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(input_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def _state_action(self, state, action):
        state = state if self.encoder is None else self.encoder(state)
        return torch.cat([state, action], dim=1)

    def forward(self, state, action):
        state_action = self._state_action(state, action)
        value1 = F.elu(self.l1(state_action))
        value1 = F.elu(self.l2(value1))
        value1 = self.l3(value1)
        value2 = F.elu(self.l4(state_action))
        value2 = F.elu(self.l5(value2))
        value2 = self.l6(value2)
        return value1, value2

    def h1(self, state, action):
        state_action = self._state_action(state, action)
        value1 = F.elu(self.l1(state_action))
        value1 = F.elu(self.l2(value1))
        return self.l3(value1)


class MultiplierNetwork(nn.Module):
    """A nonnegative statewise Lagrange multiplier lambda(s).

    The output is ``min_lambda + softplus(MLP(s))``, matching the softplus
    output layer of the paper's multiplier network (Appendix C, Table 1).
    Softplus has unit slope for every positive value, so a dual step of size
    ``multiplier_lr`` moves lambda by approximately that amount. A sigmoid
    bounded to ``[min_lambda, max_lambda]`` instead scales every step by the
    width of the bound and by ``sigmoid'(logit)``, which shrinks the
    effective multiplier learning rate and vanishes at both bounds.

    ``max_lambda`` optionally clips the output to the paper's upper bound
    lambda_max, as the reference implementation does. The clip is a hard
    ceiling (it zeroes the gradient of saturated states); ``None`` leaves the
    multiplier unbounded above, matching the reference default.

    ``min_lambda`` is a floor on the learned multiplier. The dual drives
    lambda(s) to zero wherever the safety critic predicts safe states, and on
    a corridor whose visited states are overwhelmingly safe that collapses
    the whole network and switches the constraint off. A positive floor keeps
    a minimum safety pressure everywhere while leaving the statewise
    variation above it intact. ``min_lambda = 0`` reproduces the pure dual.
    """

    def __init__(self, state_dim, max_lambda, initial_lambda, min_lambda=0.0,
                 encoder=None):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.input_dim if encoder is not None else state_dim
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        # Start with lambda(s) == initial_lambda for every state: the random
        # output weights are zeroed so only the bias sets the softplus input.
        # This is the inverse of softplus written without exp(), so it stays
        # finite for the large initial multipliers used with unscaled rewards.
        initial_value = initial_lambda - min_lambda
        initial_logit = initial_value + math.log1p(-math.exp(-initial_value))
        nn.init.zeros_(self.l3.weight)
        nn.init.constant_(self.l3.bias, initial_logit)

    def forward(self, state):
        if self.encoder is not None:
            state = self.encoder(state)
        multiplier = F.elu(self.l1(state))
        multiplier = F.elu(self.l2(multiplier))
        values = self.min_lambda + F.softplus(self.l3(multiplier))
        if self.max_lambda is not None:
            values = values.clamp(max=self.max_lambda)
        return values


class RCRLTD3:
    """TD3 reward learning with RCRL's reachability safety constraint.

    TD3 target-policy smoothing applies to reward targets only. RCRL safety
    targets evaluate the unperturbed target policy as required by its Bellman
    equation.

    The actor's safety term is unhinged, so safe states (negative predicted
    h) receive a clearance bonus that pulls the policy toward them. The
    multiplier uses the same unhinged dual objective, so lambda can fall
    again once predicted violations disappear.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.1,  # 0.2 was less stable in training
        noise_clip=0.5,
        policy_freq=4,
        critic_lr=1e-4,
        actor_lr=2e-5,
        safety_lr=1e-4,
        multiplier_lr=6e-7,
        critic_lr_final=1e-6,
        actor_lr_final=1e-6,
        safety_lr_final=1e-6,
        multiplier_lr_final=1e-7,
        anneal_steps=None,
        multiplier_freq=12,
        max_lambda=None,
        initial_lambda=5.0,
        min_lambda=0.0,
        multiplier_type="scalar",
        grad_clip=1.0,
        safety_updates=1,
        safety_batch_size=None,
        safety_mix=0.0,
        safety_mix_warmup=10_000,
        safety_twin=False,
        safety_aggregation="min",
        safety_inject_mix=0.0,
        lidar_encoder=False,
        lidar_embed_dim=16,
        num_lidar_rays=64,
    ):
        if policy_freq < 1:
            raise ValueError("policy_freq must be positive")
        if safety_updates < 1:
            raise ValueError("safety_updates must be at least 1")
        if safety_batch_size is not None and safety_batch_size < 1:
            raise ValueError("safety_batch_size must be positive")
        if not 0.0 <= safety_mix <= 1.0:
            raise ValueError("safety_mix must be between 0 and 1")
        if safety_mix_warmup < 1:
            raise ValueError("safety_mix_warmup must be positive")
        if multiplier_freq < policy_freq:
            raise ValueError("multiplier_freq must be at least policy_freq")
        if max_lambda is not None and max_lambda <= 0:
            raise ValueError("max_lambda must be positive or None")
        if initial_lambda <= 0:
            raise ValueError("initial_lambda must be positive")
        if min_lambda < 0:
            raise ValueError("min_lambda must be nonnegative")
        if grad_clip < 0:
            raise ValueError("grad_clip must be nonnegative")
        if multiplier_type not in ("scalar", "network"):
            raise ValueError("multiplier_type must be 'scalar' or 'network'")
        if safety_aggregation not in ("min", "mean"):
            raise ValueError(
                "safety_aggregation must be 'min' or 'mean'"
            )
        if not 0.0 <= safety_inject_mix <= 1.0:
            raise ValueError("safety_inject_mix must be between 0 and 1")
        if anneal_steps is not None and anneal_steps < 1:
            raise ValueError("anneal_steps must be positive or None")
        for name, lr in (
            ("critic_lr", critic_lr),
            ("actor_lr", actor_lr),
            ("safety_lr", safety_lr),
            ("multiplier_lr", multiplier_lr),
            ("critic_lr_final", critic_lr_final),
            ("actor_lr_final", actor_lr_final),
            ("safety_lr_final", safety_lr_final),
            ("multiplier_lr_final", multiplier_lr_final),
        ):
            if lr < 0:
                raise ValueError(f"{name} must be nonnegative")
        # The bounds apply to the learned network; a fixed scalar may exceed
        # max_lambda and the floor is irrelevant in that mode.
        if multiplier_type == "network" and not (
            min_lambda < initial_lambda
        ):
            raise ValueError(
                "network mode requires min_lambda < initial_lambda"
            )
        if multiplier_type == "network" and max_lambda is not None and not (
            initial_lambda < max_lambda
        ):
            raise ValueError(
                "network mode requires initial_lambda < max_lambda"
            )

        if lidar_embed_dim < 1:
            raise ValueError("lidar_embed_dim must be positive")
        if num_lidar_rays < 1:
            raise ValueError("num_lidar_rays must be positive")

        # The paper's Table 1 uses Adam with beta1 = 0.99, beta2 = 0.999.
        self.adam_betas = (0.99, 0.999)
        # Each network owns its encoder, so the lidar features are trained by
        # that network's own loss and never shared with the target copies of
        # another module.
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.lidar_encoder = bool(lidar_encoder)
        self.lidar_embed_dim = int(lidar_embed_dim)
        self.num_lidar_rays = int(num_lidar_rays)

        self.actor = Actor(
            state_dim, action_dim, max_action,
            encoder=self._make_state_encoder(),
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=self.adam_betas
        )

        self.critic = Critic(
            state_dim, action_dim, encoder=self._make_state_encoder()
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=self.adam_betas
        )

        self.safety_twin = bool(safety_twin)
        self.safety_aggregation = safety_aggregation
        if self.safety_twin:
            self.safety_critic = SafetyCriticTwin(
                state_dim, action_dim, encoder=self._make_state_encoder()
            ).to(device)
        else:
            self.safety_critic = SafetyCritic(
                state_dim, action_dim, encoder=self._make_state_encoder()
            ).to(device)
        self.safety_critic_target = copy.deepcopy(self.safety_critic)
        self.safety_critic_optimizer = torch.optim.Adam(
            self.safety_critic.parameters(), lr=safety_lr, betas=self.adam_betas
        )

        self.multiplier_type = multiplier_type
        if multiplier_type == "network":
            self.multiplier = MultiplierNetwork(
                state_dim, max_lambda, initial_lambda, min_lambda,
                encoder=self._make_state_encoder(),
            ).to(device)
            self.multiplier_optimizer = torch.optim.Adam(
                self.multiplier.parameters(), lr=multiplier_lr,
                betas=self.adam_betas,
            )
        else:
            # A fixed Lagrange multiplier shared by all states, set by the
            # user via --initial-lambda and never updated during training.
            self.fixed_lambda = float(initial_lambda)

        self.max_action = max_action
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        self.initial_lambda = initial_lambda
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.multiplier_freq = multiplier_freq
        self.grad_clip = grad_clip
        # Linear learning-rate annealing from the initial to the final value
        # over ``anneal_steps`` training updates, as in the paper (Table 1:
        # critic/safety 1e-4 -> 1e-6, actor 2e-5 -> 1e-6, multiplier
        # 6e-7 -> 1e-7). ``anneal_steps=None`` keeps the rates constant and is
        # what evaluation uses, where no further training happens.
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.safety_lr = safety_lr
        self.multiplier_lr = multiplier_lr
        self.critic_lr_final = critic_lr_final
        self.actor_lr_final = actor_lr_final
        self.safety_lr_final = safety_lr_final
        self.multiplier_lr_final = multiplier_lr_final
        self.anneal_steps = None if anneal_steps is None else int(anneal_steps)
        # Fraction of each safety batch drawn from the replay buffer's
        # near-boundary sub-buffer, and the sub-buffer fill level at which
        # the fraction ramps up to its full value. 0 keeps plain uniform
        # sampling and reproduces the original behaviour.
        self.safety_mix = float(safety_mix)
        self.safety_mix_warmup = int(safety_mix_warmup)
        self.safety_inject_mix = float(safety_inject_mix)

        # Extra safety-critic steps per training update test whether the
        # single safety critic is undertrained. Defaults reproduce the
        # original one-step-on-the-shared-batch behaviour exactly.
        self.safety_updates = int(safety_updates)
        self.safety_batch_size = (
            None if safety_batch_size is None else int(safety_batch_size)
        )
        self.total_it = 0

    def _make_state_encoder(self):
        """A fresh lidar encoder for one network, or ``None`` when disabled.

        Every network gets its own instance: sharing one module would let the
        actor, critic and safety losses train a single set of lidar features
        through different objectives, and the target copies would no longer be
        independent.
        """
        if not self.lidar_encoder:
            return None
        return LidarStateEncoder(
            state_dim=self.state_dim,
            num_lidar_rays=self.num_lidar_rays,
            embed_dim=self.lidar_embed_dim,
        )

    def multiplier_for(self, state):
        """Lagrange multiplier as a (batch, 1) tensor for this state."""
        if self.multiplier_type == "scalar":
            return torch.full((state.shape[0], 1), self.fixed_lambda, device=device)
        return self.multiplier(state)

    def mean_lambda(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.multiplier_for(state).mean().item()

    def lambda_stats(self, state):
        """Mean, max and cap fraction of lambda(s), for progress logging."""
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            values = self.multiplier_for(state)
            stats = {
                "mean": values.mean().item(),
                "max": values.max().item(),
            }
            if self.multiplier_type == "network":
                if self.max_lambda is None:
                    # No ceiling to saturate, so report only the floor rate.
                    floor_tolerance = 0.01 * max(self.initial_lambda, 1.0)
                else:
                    stats["cap_fraction"] = (
                        values >= 0.99 * self.max_lambda
                    ).float().mean().item()
                    floor_tolerance = 0.01 * (
                        self.max_lambda - self.min_lambda
                    )
                stats["floor_fraction"] = (
                    values <= self.min_lambda + floor_tolerance
                ).float().mean().item()
        return stats

    def policy_safety_value(self, state):
        """Mean safety-critic prediction h(s, pi(s)) for the current policy."""
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if self.safety_twin:
                return self.safety_critic.h1(
                    state, self.actor(state)
                ).mean().item()
            return self.safety_critic(state, self.actor(state)).mean().item()

    def select_action(self, state):
        state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().flatten()

    @staticmethod
    def _soft_update(source, target, tau):
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.copy_(tau * parameter.data + (1.0 - tau) * target_parameter.data)

    @staticmethod
    def _set_requires_grad(modules, requires_grad):
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(requires_grad)

    def _clip_grads(self, module):
        """Clip a module's gradients to a global norm of grad_clip (0 disables)."""
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)

    def _learning_rate(self, initial, final):
        """Linearly decay ``initial`` to ``final`` over ``anneal_steps``.

        The paper anneals every rate over the full training run (Table 1);
        with ``anneal_steps=None`` the rate is constant. Progress is measured
        in training updates, which equal collected transitions in both
        trainers, so the schedule spans the same horizon as the paper's.
        """
        if self.anneal_steps is None:
            return initial
        progress = min(1.0, self.total_it / self.anneal_steps)
        return initial + (final - initial) * progress

    def _apply_learning_rate(self, optimizer, initial, final):
        """Set ``optimizer`` to its scheduled rate for the current update."""
        learning_rate = self._learning_rate(initial, final)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def safety_target(self, next_state, h, next_h, not_done):
        """One-step reachability target for the safety critic.

        ``h`` is the constraint of the current state, ``next_h`` the
        constraint of the state reached by the action, and ``not_done`` is
        zero on terminal transitions. RCRL evaluates the unperturbed target
        policy and bootstraps from the current ``h``, which gives

            (1 - discount) * h + discount * max(h, continuation_h).

        With the twin critic the two target heads are combined by
        ``safety_aggregation`` before the max with h. ``"min"`` (the default)
        equals the min of the two full twin targets (max with the constant h
        distributes over min) and damps overestimation the way TD3's min over
        the reward critics does. ``"mean"`` averages the two heads instead;
        it cannot be moved outside the max, so it is applied to the
        continuation value directly.

        Kept as one method so the online update and the offline diagnostic
        cannot use different formulas.
        """
        with torch.no_grad():
            # RCRL evaluates the unperturbed target policy and the current h(s).
            safety_next_action = self.actor_target(next_state)
            if self.safety_twin:
                future_h1, future_h2 = self.safety_critic_target(
                    next_state, safety_next_action
                )
                if self.safety_aggregation == "mean":
                    future_h = 0.5 * (future_h1 + future_h2)
                else:
                    future_h = torch.minimum(future_h1, future_h2)
            else:
                future_h = self.safety_critic_target(next_state, safety_next_action)
            continuation_h = torch.where(not_done.bool(), future_h, next_h)
            return (1.0 - self.discount) * h + self.discount * torch.maximum(
                h, continuation_h
            )

    def _update_safety_critic(
        self, state, action, next_state, h, next_h, not_done
    ):
        """One safety-critic gradient step; returns (current_h, safety_loss)."""
        target_h = self.safety_target(next_state, h, next_h, not_done)

        if self.safety_twin:
            current_h1, current_h2 = self.safety_critic(state, action)
            safety_loss = F.mse_loss(current_h1, target_h) + F.mse_loss(
                current_h2, target_h
            )
            # Head 1 is the value reported and used downstream, matching
            # how the reward side reports q1.
            current_h = current_h1
        else:
            current_h = self.safety_critic(state, action)
            safety_loss = F.mse_loss(current_h, target_h)
        self.safety_critic_optimizer.zero_grad()
        safety_loss.backward()
        self._clip_grads(self.safety_critic)
        self._apply_learning_rate(
            self.safety_critic_optimizer, self.safety_lr, self.safety_lr_final
        )
        self.safety_critic_optimizer.step()
        return current_h, safety_loss

    def _draw_safety_batch(self, replay_buffer, batch_size,
                           injection_buffer=None):
        """Draw a batch for a safety-critic step.

        With ``safety_mix > 0`` the batch blends near-boundary transitions
        (the buffer's boundary sub-buffer) with uniform draws from the main
        buffer, ramping up as the sub-buffer fills. This puts safety
        gradient where the ``h(s)`` anchor of the Bellman target is weakest,
        i.e. states with small but negative constraint that the policy
        otherwise avoids. ``sample_mixed`` falls back to plain uniform
        sampling when the buffer has no boundary sub-buffer.

        With ``safety_inject_mix > 0`` and an ``injection_buffer``, a
        fraction of the batch is drawn from transitions generated by the
        safety data generator (states sampled in the danger shell, one step
        under the actor). The fraction ramps up linearly from zero as that
        buffer fills past ``safety_mix_warmup`` transitions.
        """
        if (
            self.safety_inject_mix > 0
            and injection_buffer is not None
            and injection_buffer.size > 0
        ):
            effective = self.safety_inject_mix * min(
                1.0,
                injection_buffer.size / float(max(self.safety_mix_warmup, 1)),
            )
            n_injected = min(
                int(round(batch_size * effective)), injection_buffer.size
            )
            if n_injected > 0:
                rest = self._draw_safety_batch(
                    replay_buffer, batch_size - n_injected
                )
                injected = injection_buffer.sample(n_injected)
                return tuple(
                    torch.cat([rest_part, inj_part])
                    for rest_part, inj_part in zip(rest, injected)
                )
        if self.safety_mix > 0:
            return replay_buffer.sample_mixed(
                batch_size, self.safety_mix, self.safety_mix_warmup
            )
        return replay_buffer.sample(batch_size)

    def train(self, replay_buffer, batch_size=256, injection_buffer=None):
        self.total_it += 1
        state, action, next_state, reward, h, next_h, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            reward_noise = torch.randn_like(action).mul_(self.policy_noise)
            reward_noise.clamp_(-self.noise_clip, self.noise_clip)
            reward_next_action = (
                self.actor_target(next_state) + reward_noise
            ).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, reward_next_action)
            target_q = reward + not_done * self.discount * torch.minimum(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self._clip_grads(self.critic)
        self._apply_learning_rate(
            self.critic_optimizer, self.critic_lr, self.critic_lr_final
        )
        self.critic_optimizer.step()

        # The defaults (one update, shared batch, no boundary mix) reproduce
        # the original single step. Extra updates, a separate safety batch
        # size, or a boundary mix draw a fresh batch for every safety step:
        # repeated passes over the same samples would overfit and understate
        # the loss, and the reward batch is uniform by construction.
        # An injection buffer also forces its own safety batch, so the
        # generated transitions reach the safety critic even when the
        # defaults would share the reward batch.
        if (
            self.safety_updates == 1
            and self.safety_batch_size is None
            and self.safety_mix <= 0
            and injection_buffer is None
        ):
            current_h, safety_loss = self._update_safety_critic(
                state, action, next_state, h, next_h, not_done
            )
            safety_loss_values = [safety_loss.item()]
        else:
            safety_batch_size = self.safety_batch_size or batch_size
            safety_loss_values = []
            for _ in range(self.safety_updates):
                (
                    safety_state,
                    safety_action,
                    safety_next_state,
                    _,
                    safety_h,
                    safety_next_h,
                    safety_not_done,
                ) = self._draw_safety_batch(
                    replay_buffer, safety_batch_size, injection_buffer
                )
                current_h, safety_loss = self._update_safety_critic(
                    safety_state,
                    safety_action,
                    safety_next_state,
                    safety_h,
                    safety_next_h,
                    safety_not_done,
                )
                safety_loss_values.append(safety_loss.item())
        mean_safety_loss = sum(safety_loss_values) / len(safety_loss_values)

        metrics = {
            "critic_loss": critic_loss.item(),
            "safety_loss": mean_safety_loss,
            "mean_constraint": h.mean().item(),
            "mean_safety_value": current_h.detach().mean().item(),
            "mean_multiplier": self.multiplier_for(state).detach().mean().item(),
        }

        if self.total_it % self.policy_freq == 0:
            self._set_requires_grad([self.critic, self.safety_critic], False)
            policy_action = self.actor(state)
            q_pi = self.critic.q1(state, policy_action)
            if self.safety_twin:
                qh_pi = self.safety_critic.h1(state, policy_action)
            else:
                qh_pi = self.safety_critic(state, policy_action)
            lambda_s = self.multiplier_for(state).detach()
            # Unhinged term: safe states (negative predicted h) earn a
            # clearance bonus that pulls the policy toward them.
            actor_loss = (-q_pi + lambda_s * qh_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self._clip_grads(self.actor)
            self._apply_learning_rate(
                self.actor_optimizer, self.actor_lr, self.actor_lr_final
            )
            self.actor_optimizer.step()
            self._set_requires_grad([self.critic, self.safety_critic], True)

            metrics["actor_loss"] = actor_loss.item()
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.safety_critic, self.safety_critic_target, self.tau)

        if self.multiplier_type == "network" and self.total_it % self.multiplier_freq == 0:
            with torch.no_grad():
                multiplier_action = self.actor(state)
                if self.safety_twin:
                    multiplier_safety_value = self.safety_critic.h1(
                        state, multiplier_action
                    )
                else:
                    multiplier_safety_value = self.safety_critic(
                        state, multiplier_action
                    )
            # Keep the unhinged dual: unlike the actor term, lambda must be
            # able to fall when predicted violations disappear.
            multiplier_loss = -(
                self.multiplier(state) * multiplier_safety_value
            ).mean()
            self.multiplier_optimizer.zero_grad()
            multiplier_loss.backward()
            self._clip_grads(self.multiplier)
            self._apply_learning_rate(
                self.multiplier_optimizer,
                self.multiplier_lr,
                self.multiplier_lr_final,
            )
            self.multiplier_optimizer.step()
            metrics["multiplier_loss"] = multiplier_loss.item()

        return metrics

    def save(self, filename):
        checkpoints = {
            "_critic.pth": self.critic.state_dict(),
            "_critic_optimizer.pth": self.critic_optimizer.state_dict(),
            "_actor.pth": self.actor.state_dict(),
            "_actor_optimizer.pth": self.actor_optimizer.state_dict(),
            "_safety_critic.pth": self.safety_critic.state_dict(),
            "_safety_critic_optimizer.pth": self.safety_critic_optimizer.state_dict(),
        }
        if self.multiplier_type == "network":
            checkpoints["_multiplier.pth"] = self.multiplier.state_dict()
            checkpoints["_multiplier_optimizer.pth"] = (
                self.multiplier_optimizer.state_dict()
            )
        # Write each artifact to a temp file, then move it into place, so a
        # crash mid-save cannot leave a mixed (partially updated) checkpoint.
        for suffix, state_dict in checkpoints.items():
            tmp_path = filename + suffix + ".tmp"
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, filename + suffix)

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer.pth", map_location=device)
        )
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer.pth", map_location=device)
        )

        self.safety_critic.load_state_dict(
            torch.load(filename + "_safety_critic.pth", map_location=device)
        )
        self.safety_critic_optimizer.load_state_dict(
            torch.load(filename + "_safety_critic_optimizer.pth", map_location=device)
        )
        if self.multiplier_type == "network":
            self.multiplier.load_state_dict(
                torch.load(filename + "_multiplier.pth", map_location=device)
            )
            self.multiplier_optimizer.load_state_dict(
                torch.load(filename + "_multiplier_optimizer.pth", map_location=device)
            )

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.safety_critic_target = copy.deepcopy(self.safety_critic)
