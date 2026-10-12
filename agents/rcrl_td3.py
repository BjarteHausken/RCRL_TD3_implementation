import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.td3 import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SafetyCritic(nn.Module):
    """
    Action-conditioned reachability safety critic.

    Output interpretation:
        Q_h(s, a) <= 0 -> predicted safe
        Q_h(s, a) > 0  -> predicted future constraint violation

    Assumes the environment constraint uses:
        h(s) <= 0 -> safe
        h(s) > 0  -> unsafe
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class RCRL_TD3(object):
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
        safety_lr=3e-4,
    ):

        # Normal TD3 actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4
        )

        # Normal TD3 twin reward critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4
        )

        # RCRL safety critic
        self.safety_critic = SafetyCritic(state_dim, action_dim).to(device)
        self.safety_critic_target = copy.deepcopy(self.safety_critic)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critic.parameters(), lr=safety_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, next_constraint, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(
                next_state, next_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = (
            F.mse_loss(current_Q1, target_Q)
            + F.mse_loss(current_Q2, target_Q)
        )

        # RCRL critic update
        safety_loss = self.update_safety_critic(state, action, next_state, next_constraint, not_done)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(state)

            # Keep gradients through the critics with respect to the action,
            # but do not accumulate gradients for critic parameters.
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.safety_critic_optimizer.zero_grad(set_to_none=True)
            critic_parameters = list(self.critic.parameters())
            safety_parameters = list(self.safety_critic.parameters())
            for parameter in critic_parameters + safety_parameters:
                parameter.requires_grad_(False)

            reward_value = self.critic.Q1(state, actor_action)
            safety_value = self.safety_critic(state, actor_action)

            safety_penalty = F.relu(safety_value)

            # Compute actor loss
            actor_loss = -reward_value.mean() + 1.0 * safety_penalty.mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for parameter in critic_parameters + safety_parameters:
                parameter.requires_grad_(True)

            # Update the frozen target models
            self.soft_update(self.critic, self.critic_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.safety_critic, self.safety_critic_target, self.tau)

        return {"critic_loss": critic_loss.item(), "safety_loss": safety_loss, "mean_constraint": next_constraint.mean().item(),}

    def soft_update(self, source, target, tau):
        for parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.copy_(tau * parameter.data + (1.0 - tau) * target_parameter.data)

    
    def compute_safety_target(self, next_state, next_constraint, not_done):
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            future_safety = self.safety_critic_target(next_state, next_action)
            continued_target = torch.maximum(next_constraint, future_safety)

            # If terminal, the final observed constraint is the target
            safety_target = torch.where(not_done.bool(), continued_target, next_constraint)

        return safety_target
    
    def update_safety_critic(self, state, action, next_state, next_constraint, not_done):
        safety_target = self.compute_safety_target(next_state, next_constraint, not_done)
        current_safety = self.safety_critic(state, action)

        safety_critic_loss = F.mse_loss(current_safety, safety_target)

        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        
        # Optional
        torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), max_norm=10.0)

        self.safety_critic_optimizer.step()

        return safety_critic_loss.item()


    def save(self, filename):
        torch.save(
            self.critic.state_dict(),
            filename + "_critic.pth",
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            filename + "_critic_optimizer.pth",
        )

        torch.save(
            self.actor.state_dict(),
            filename + "_actor.pth",
        )
        torch.save(
            self.actor_optimizer.state_dict(),
            filename + "_actor_optimizer.pth",
        )

        torch.save(self.safety_critic.state_dict(), filename + "_safety_critic.pth")
        torch.save(
            self.safety_critic_optimizer.state_dict(),
            filename + "_safety_critic_optimizer.pth",
        )

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer.pth", map_location=device)
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(
            torch.load(filename + "_actor.pth", map_location=device)
        )
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer.pth", map_location=device)
        )
        self.actor_target = copy.deepcopy(self.actor)

        self.safety_critic.load_state_dict(
            torch.load(filename + "_safety_critic.pth", map_location=device)
        )
        self.safety_critic_optimizer.load_state_dict(
            torch.load(filename + "_safety_critic_optimizer.pth", map_location=device)
        )
        self.safety_critic_target = copy.deepcopy(self.safety_critic)
