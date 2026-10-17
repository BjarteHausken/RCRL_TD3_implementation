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
    """A bounded, nonnegative statewise Lagrange multiplier lambda(s)."""

    def __init__(self, state_dim, max_lambda, initial_lambda):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.max_lambda = max_lambda
        initial_logit = math.log(math.expm1(initial_lambda))
        nn.init.constant_(self.l3.bias, initial_logit)

    def forward(self, state):
        multiplier = F.relu(self.l1(state))
        multiplier = F.relu(self.l2(multiplier))
        multiplier = F.softplus(self.l3(multiplier))
        return multiplier.clamp(max=self.max_lambda)


class RCRLTD3:
    """TD3 reward learning with RCRL's reachability safety constraint.

    TD3 target-policy smoothing applies to reward targets only. RCRL safety
    targets evaluate the unperturbed target policy as required by its Bellman
    equation.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        critic_lr=1e-4,
        actor_lr=2e-5,
        safety_lr=1e-4,
        multiplier_lr=1e-6,
        multiplier_freq=6,
        max_lambda=100.0,
        initial_lambda=0.1,
    ):
        if policy_freq < 1:
            raise ValueError("policy_freq must be positive")
        if multiplier_freq < policy_freq:
            raise ValueError("multiplier_freq must be at least policy_freq")
        if max_lambda <= 0:
            raise ValueError("max_lambda must be positive")
        if not 0 < initial_lambda <= max_lambda:
            raise ValueError("initial_lambda must be in (0, max_lambda]")

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.safety_critic = SafetyCritic(state_dim, action_dim).to(device)
        self.safety_critic_target = copy.deepcopy(self.safety_critic)
        self.safety_critic_optimizer = torch.optim.Adam(
            self.safety_critic.parameters(), lr=safety_lr
        )

        self.multiplier = MultiplierNetwork(
            state_dim, max_lambda, initial_lambda
        ).to(device)
        self.multiplier_optimizer = torch.optim.Adam(
            self.multiplier.parameters(), lr=multiplier_lr
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.multiplier_freq = multiplier_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().flatten()

    @staticmethod
    def _soft_update(source, target, tau):
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.copy_(
                tau * parameter.data + (1.0 - tau) * target_parameter.data
            )

    @staticmethod
    def _set_requires_grad(modules, requires_grad):
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(requires_grad)

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
        self.safety_critic_optimizer.step()
        return current_h, safety_loss

    def train_warmup(self, replay_buffer, batch_size=256):
        """Critic warmup for the random-exploration phase.

        Trains the reward critic and the safety critic (and their targets) on
        the broad random state coverage, but leaves the actor and multiplier
        untouched so the policy cannot exploit an inaccurate constraint
        estimate. The RCRL safety value is learned over the feasible set
        before the reward-driven actor starts.
        """
        self.total_it += 1
        state, action, next_state, reward, h, next_h, not_done = replay_buffer.sample(
            batch_size
        )

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
        self.critic_optimizer.step()

        current_h, safety_loss = self._update_safety_critic(
            state, action, next_state, h, next_h, not_done
        )

        self._soft_update(self.critic, self.critic_target, self.tau)
        self._soft_update(self.safety_critic, self.safety_critic_target, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "safety_loss": safety_loss.item(),
            "mean_constraint": h.mean().item(),
            "mean_safety_value": current_h.detach().mean().item(),
            }

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
        self.critic_optimizer.step()

        current_h, safety_loss = self._update_safety_critic(
            state, action, next_state, h, next_h, not_done
        )

        metrics = {
            "critic_loss": critic_loss.item(),
            "safety_loss": safety_loss.item(),
            "mean_constraint": h.mean().item(),
            "mean_safety_value": current_h.detach().mean().item(),
            "mean_multiplier": self.multiplier(state).detach().mean().item(),
        }

        if self.total_it % self.policy_freq == 0:
            self._set_requires_grad([self.critic, self.safety_critic], False)
            policy_action = self.actor(state)
            q_pi = self.critic.q1(state, policy_action)
            qh_pi = self.safety_critic(state, policy_action)
            lambda_s = self.multiplier(state).detach()
            actor_loss = (-q_pi + lambda_s * qh_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._set_requires_grad([self.critic, self.safety_critic], True)

            metrics["actor_loss"] = actor_loss.item()
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)
            self._soft_update(self.safety_critic, self.safety_critic_target, self.tau)

        if self.total_it % self.multiplier_freq == 0:
            with torch.no_grad():
                multiplier_action = self.actor(state)
                multiplier_safety_value = self.safety_critic(state, multiplier_action)
            multiplier_loss = -(
                self.multiplier(state) * multiplier_safety_value
            ).mean()
            self.multiplier_optimizer.zero_grad()
            multiplier_loss.backward()
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
            "_multiplier.pth": self.multiplier.state_dict(),
            "_multiplier_optimizer.pth": self.multiplier_optimizer.state_dict(),
        }
        for suffix, state_dict in checkpoints.items():
            torch.save(state_dict, filename + suffix)

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer.pth", map_location=device)
        )
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer.pth", map_location=device)
        )

        # Actor-only evaluation remains compatible with earlier checkpoints.
        safety_path = filename + "_safety_critic.pth"
        if os.path.exists(safety_path):
            self.safety_critic.load_state_dict(torch.load(safety_path, map_location=device))
            optimizer_path = filename + "_safety_critic_optimizer.pth"
            if os.path.exists(optimizer_path):
                self.safety_critic_optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device)
                )

        multiplier_path = filename + "_multiplier.pth"
        if os.path.exists(multiplier_path):
            self.multiplier.load_state_dict(torch.load(multiplier_path, map_location=device))
            optimizer_path = filename + "_multiplier_optimizer.pth"
            if os.path.exists(optimizer_path):
                self.multiplier_optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=device)
                )

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.safety_critic_target = copy.deepcopy(self.safety_critic)


# Backwards-compatible spelling used by the evaluation scripts and old callers.
RCRL_TD3 = RCRLTD3
