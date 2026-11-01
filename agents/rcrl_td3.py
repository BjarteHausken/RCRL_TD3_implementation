"""TD3 with the reachability constraint and statewise dual update from RCRL."""

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        action = F.relu(self.l1(state))
        action = F.relu(self.l2(action))
        return self.max_action * torch.tanh(self.l3(action))


class Critic(nn.Module):
    """Twin reward critic used by TD3."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(state_action))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(state_action))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(state_action))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


class SafetyCritic(nn.Module):
    """Single RCRL safety-value critic, where positive values are unsafe."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        value = F.relu(self.l1(state_action))
        value = F.relu(self.l2(value))
        return self.l3(value)


class MultiplierNetwork(nn.Module):
    """A bounded, nonnegative statewise Lagrange multiplier lambda(s).

    The output is ``min_lambda + (max_lambda - min_lambda) * sigmoid(...)``.
    The cap is a hard bound of the parameterisation rather than a clamp applied
    after the fact. A post-hoc clamp zeroes the gradient of every state at the
    cap and freezes the dual there; a straight-through clamp keeps the gradient
    but lets the pre-clamp value drift arbitrarily far above the cap, so the
    dual can no longer descend within a training run. Sigmoid keeps a small
    (for Adam, sufficient) gradient at both bounds and cannot overshoot them.

    ``min_lambda`` is a floor on the learned multiplier. The dual drives
    lambda(s) to zero wherever the safety critic predicts safe states, and on a
    corridor whose visited states are overwhelmingly safe that collapses the
    whole network and switches the constraint off. A positive floor keeps a
    minimum safety pressure everywhere while leaving the statewise variation
    above it intact. ``min_lambda = 0`` reproduces the pure dual.
    """

    def __init__(self, state_dim, max_lambda, initial_lambda, min_lambda=0.0):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        # Start with lambda(s) == initial_lambda for every state: the random
        # output weights are zeroed so only the bias sets the sigmoid logit.
        initial_fraction = (initial_lambda - min_lambda) / (
            max_lambda - min_lambda
        )
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        nn.init.zeros_(self.l3.weight)
        nn.init.constant_(self.l3.bias, initial_logit)

    def forward(self, state):
        multiplier = F.relu(self.l1(state))
        multiplier = F.relu(self.l2(multiplier))
        # Evaluate the sigmoid in float64: in float32 it saturates to exactly
        # 0 or 1 once the logit passes about +/-17, which zeroes the gradient
        # and would reproduce the clamp freeze for a long-running dual. The
        # result is cast back so the multiplier keeps the float32 dtype of
        # the rest of the actor loss.
        logit = self.l3(multiplier).double()
        bounded = self.min_lambda + (
            self.max_lambda - self.min_lambda
        ) * torch.sigmoid(logit)
        return bounded.to(torch.float32)


class RCRLTD3:
    """TD3 reward learning with RCRL's reachability safety constraint.

    TD3 target-policy smoothing applies to reward targets only. RCRL safety
    targets evaluate the unperturbed target policy as required by its Bellman
    equation.

    Both the actor's safety term and the multiplier's dual objective hinge on
    positive safety values. Safe states therefore receive no clearance bonus
    and cannot drag the multiplier down; only predicted constraint violations
    raise it.
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
        policy_freq=2,
        critic_lr=1e-4,
        actor_lr=2e-5,
        safety_lr=1e-4,
        multiplier_lr=1e-5,
        multiplier_freq=6,
        max_lambda=2.0,
        initial_lambda=0.1,
        min_lambda=0.0,
        multiplier_type="network",
        grad_clip=1.0,
    ):
        if policy_freq < 1:
            raise ValueError("policy_freq must be positive")
        if multiplier_freq < policy_freq:
            raise ValueError("multiplier_freq must be at least policy_freq")
        if max_lambda <= 0:
            raise ValueError("max_lambda must be positive")
        if initial_lambda <= 0:
            raise ValueError("initial_lambda must be positive")
        if min_lambda < 0:
            raise ValueError("min_lambda must be nonnegative")
        if grad_clip < 0:
            raise ValueError("grad_clip must be nonnegative")
        if multiplier_type not in ("scalar", "network"):
            raise ValueError("multiplier_type must be 'scalar' or 'network'")
        # The bounds apply to the learned network; a fixed scalar may exceed
        # max_lambda and the floor is irrelevant in that mode.
        if multiplier_type == "network" and not (
            min_lambda < initial_lambda < max_lambda
        ):
            raise ValueError(
                "network mode requires min_lambda < initial_lambda < max_lambda"
            )

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.safety_critic = SafetyCritic(state_dim, action_dim).to(device)
        self.safety_critic_target = copy.deepcopy(self.safety_critic)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critic.parameters(), lr=safety_lr)

        self.multiplier_type = multiplier_type
        if multiplier_type == "network":
            self.multiplier = MultiplierNetwork(
                state_dim, max_lambda, initial_lambda, min_lambda
            ).to(device)
            self.multiplier_optimizer = torch.optim.Adam(self.multiplier.parameters(), lr=multiplier_lr)
        else:
            # A fixed Lagrange multiplier shared by all states, set by the
            # user via --initial-lambda and never updated during training.
            self.fixed_lambda = float(initial_lambda)

        self.max_action = max_action
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.multiplier_freq = multiplier_freq
        self.grad_clip = grad_clip
        self.total_it = 0

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
                stats["cap_fraction"] = (
                    values >= 0.99 * self.max_lambda
                ).float().mean().item()
                floor_tolerance = 0.01 * (self.max_lambda - self.min_lambda)
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

    def _update_safety_critic(
        self, state, action, next_state, h, next_h, not_done
    ):
        """One safety-critic gradient step; returns (current_h, safety_loss)."""
        with torch.no_grad():
            # RCRL evaluates the unperturbed target policy and the current h(s).
            safety_next_action = self.actor_target(next_state)
            future_h = self.safety_critic_target(next_state, safety_next_action)
            continuation_h = torch.where(not_done.bool(), future_h, next_h)
            target_h = (1.0 - self.discount) * h + self.discount * torch.maximum(
                h, continuation_h
            )

        current_h = self.safety_critic(state, action)
        safety_loss = F.mse_loss(current_h, target_h)
        self.safety_critic_optimizer.zero_grad()
        safety_loss.backward()
        self._clip_grads(self.safety_critic)
        self.safety_critic_optimizer.step()
        return current_h, safety_loss

    def train(self, replay_buffer, batch_size=256):
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
        self.critic_optimizer.step()

        current_h, safety_loss = self._update_safety_critic(
            state, action, next_state, h, next_h, not_done
        )

        metrics = {
            "critic_loss": critic_loss.item(),
            "safety_loss": safety_loss.item(),
            "mean_constraint": h.mean().item(),
            "mean_safety_value": current_h.detach().mean().item(),
            "mean_multiplier": self.multiplier_for(state).detach().mean().item(),
        }

        if self.total_it % self.policy_freq == 0:
            self._set_requires_grad([self.critic, self.safety_critic], False)
            policy_action = self.actor(state)
            q_pi = self.critic.q1(state, policy_action)
            qh_pi = self.safety_critic(state, policy_action)
            lambda_s = self.multiplier_for(state).detach()
            # Hinge on predicted violations only: safe states must not earn a
            # clearance bonus that pushes the policy away from usable gaps.
            actor_loss = (-q_pi + lambda_s * torch.relu(qh_pi)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self._clip_grads(self.actor)
            self.actor_optimizer.step()
            self._set_requires_grad([self.critic, self.safety_critic], True)

            metrics["actor_loss"] = actor_loss.item()
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.safety_critic, self.safety_critic_target, self.tau)

        if self.multiplier_type == "network" and self.total_it % self.multiplier_freq == 0:
            with torch.no_grad():
                multiplier_action = self.actor(state)
                multiplier_safety_value = self.safety_critic(state, multiplier_action)
            # Same hinge in the dual: safe-dominated batches no longer drag
            # lambda down, only positive predicted violations raise it.
            multiplier_loss = -(
                self.multiplier(state) * torch.relu(multiplier_safety_value)
            ).mean()
            self.multiplier_optimizer.zero_grad()
            multiplier_loss.backward()
            self._clip_grads(self.multiplier)
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
