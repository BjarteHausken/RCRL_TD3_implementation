import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import LidarStateEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Adapted from Scott Fujimoto's reference implementation
# (https://github.com/sfujim/TD3), MIT License.
# Networks, optimizer and schedule follow the RCRL paper's Table 1 (ELU
# hidden layers, Adam with betas (0.99, 0.999), linear learning-rate anneal,
# policy update interval 4) so the plain baseline differs from RCRL only in
# the safety machinery.


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, encoder=None):
        super(Actor, self).__init__()

        self.encoder = encoder
        input_dim = encoder.input_dim if encoder is not None else state_dim
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def _encode_state(self, state):
        return state if self.encoder is None else self.encoder(state)

    def forward(self, state):
        a = F.elu(self.l1(self._encode_state(state)))
        a = F.elu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, encoder=None):
        super(Critic, self).__init__()

        self.encoder = encoder
        input_dim = encoder.input_dim if encoder is not None else state_dim

        # Q1 architecture
        self.l1 = nn.Linear(input_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(input_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def _state_action(self, state, action):
        state = state if self.encoder is None else self.encoder(state)
        return torch.cat([state, action], 1)

    def forward(self, state, action):
        sa = self._state_action(state, action)

        q1 = F.elu(self.l1(sa))
        q1 = F.elu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.elu(self.l4(sa))
        q2 = F.elu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = self._state_action(state, action)

        q1 = F.elu(self.l1(sa))
        q1 = F.elu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.5,
        policy_freq=4,
        critic_lr=1e-4,
        actor_lr=2e-5,
        critic_lr_final=1e-6,
        actor_lr_final=1e-6,
        anneal_steps=None,
        grad_clip=1.0,
        lidar_encoder=False,
        lidar_embed_dim=16,
        num_lidar_rays=64,
        actor_encoder_lr=None,
        actor_encoder_lr_final=None,
        lidar_encoder_kind="conv",
        lidar_encoder_min=False,
        actor_encoder_every_step=False,
        lidar_ray_channels=16,
        lidar_ray_exact_min=True,
        lidar_ray_directions=None,
    ):
        if grad_clip < 0:
            raise ValueError("grad_clip must be nonnegative")
        if lidar_embed_dim < 1:
            raise ValueError("lidar_embed_dim must be positive")
        if num_lidar_rays < 1:
            raise ValueError("num_lidar_rays must be positive")
        if actor_encoder_lr is not None and actor_encoder_lr <= 0:
            raise ValueError("actor_encoder_lr must be positive or None")
        if actor_encoder_lr_final is not None:
            if actor_encoder_lr is None:
                raise ValueError(
                    "actor_encoder_lr_final requires actor_encoder_lr"
                )
            if actor_encoder_lr_final <= 0:
                raise ValueError(
                    "actor_encoder_lr_final must be positive or None"
                )
        if (
            actor_encoder_lr is not None
            or actor_encoder_lr_final is not None
        ) and not lidar_encoder:
            raise ValueError(
                "actor encoder learning rates require the lidar encoder"
            )
        if lidar_encoder_kind not in ("conv", "linear", "ray"):
            raise ValueError(f"unknown lidar encoder kind {lidar_encoder_kind!r}")
        if lidar_ray_channels < 2:
            raise ValueError("lidar_ray_channels must be at least 2")
        if (lidar_encoder_kind != "conv" or lidar_encoder_min
                or actor_encoder_every_step) and not lidar_encoder:
            raise ValueError(
                "lidar encoder kind, min features and every-step updates "
                "require the lidar encoder"
            )
        for name, lr in (
            ("critic_lr", critic_lr),
            ("actor_lr", actor_lr),
            ("critic_lr_final", critic_lr_final),
            ("actor_lr_final", actor_lr_final),
        ):
            if lr < 0:
                raise ValueError(f"{name} must be nonnegative")
        if anneal_steps is not None and anneal_steps < 1:
            raise ValueError("anneal_steps must be positive or None")

        # Adam betas from the paper's Table 1.
        self.adam_betas = (0.99, 0.999)
        # Each network owns its encoder, so its lidar features are trained by
        # that network's own loss and its target copy stays independent.
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.lidar_encoder = bool(lidar_encoder)
        self.lidar_embed_dim = int(lidar_embed_dim)
        self.num_lidar_rays = int(num_lidar_rays)
        # ``conv`` is the circular CNN, ``linear`` the dense bottleneck
        # control. ``lidar_encoder_min`` appends the exact minimum range and
        # its bearing, which the conv's angular average cannot represent.
        self.lidar_encoder_kind = lidar_encoder_kind
        self.lidar_encoder_min = bool(lidar_encoder_min)
        # Update the actor encoder on every training update (encoder-only
        # gradient on non-policy steps) instead of once per ``policy_freq``.
        self.actor_encoder_every_step = bool(actor_encoder_every_step)
        # ``ray`` is the dimension-agnostic per-ray MLP: the two settings
        # shape it, and ``lidar_ray_directions`` supplies the ray geometry
        # when the spatial dimension is not 2.
        self.lidar_ray_channels = int(lidar_ray_channels)
        self.lidar_ray_exact_min = bool(lidar_ray_exact_min)
        self.lidar_ray_directions = lidar_ray_directions
        # Optional separate learning-rate schedule for the actor's lidar
        # encoder, which trains once per ``policy_freq`` steps at
        # ``actor_lr`` while the critic encoder trains every step at the
        # (larger) critic rate. ``None`` keeps a single param group.
        self.actor_encoder_lr = (
            None if actor_encoder_lr is None else float(actor_encoder_lr)
        )
        self.actor_encoder_lr_final = (
            None if self.actor_encoder_lr is None else float(
                actor_lr_final if actor_encoder_lr_final is None
                else actor_encoder_lr_final
            )
        )

        self.actor = Actor(
            state_dim, action_dim, max_action,
            encoder=self._make_state_encoder(),
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        if self.actor_encoder_lr is None:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr, betas=self.adam_betas
            )
            self.actor_lr_schedule = [(actor_lr, actor_lr_final)]
        else:
            # ``named_parameters`` lists the encoder first, so filtering on
            # the prefix leaves the head in its construction order.
            head_parameters = [
                parameter
                for name, parameter in self.actor.named_parameters()
                if not name.startswith("encoder.")
            ]
            self.actor_optimizer = torch.optim.Adam(
                [
                    {"params": head_parameters, "lr": actor_lr},
                    {
                        "params": list(self.actor.encoder.parameters()),
                        "lr": self.actor_encoder_lr,
                    },
                ],
                lr=actor_lr,
                betas=self.adam_betas,
            )
            self.actor_lr_schedule = [
                (actor_lr, actor_lr_final),
                (self.actor_encoder_lr, self.actor_encoder_lr_final),
            ]

        self.critic = Critic(
            state_dim, action_dim, encoder=self._make_state_encoder()
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=self.adam_betas
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.grad_clip = grad_clip

        # Linear learning-rate annealing from the initial to the final value
        # over ``anneal_steps`` updates, as in the paper. ``None`` keeps the
        # rates constant, which is what evaluation uses.
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.critic_lr_final = critic_lr_final
        self.actor_lr_final = actor_lr_final
        self.anneal_steps = None if anneal_steps is None else int(anneal_steps)

        self.total_it = 0

    def _make_state_encoder(self):
        """A fresh lidar encoder for one network, or ``None`` when disabled."""
        if not self.lidar_encoder:
            return None
        return LidarStateEncoder(
            state_dim=self.state_dim,
            num_lidar_rays=self.num_lidar_rays,
            embed_dim=self.lidar_embed_dim,
            kind=self.lidar_encoder_kind,
            min_features=self.lidar_encoder_min,
            ray_channels=self.lidar_ray_channels,
            ray_exact_min=self.lidar_ray_exact_min,
            ray_directions=self.lidar_ray_directions,
        )

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            return self.actor(state).cpu().data.numpy().flatten()

    def _learning_rate(self, initial, final):
        """Linearly decay ``initial`` to ``final`` over ``anneal_steps``."""
        if self.anneal_steps is None:
            return initial
        progress = min(1.0, self.total_it / self.anneal_steps)
        return initial + (final - initial) * progress

    def _apply_learning_rate(self, optimizer, initial, final=None):
        """Set ``optimizer`` to its scheduled rate(s) for the current update.

        ``initial`` is either one starting rate for every parameter group or
        a sequence of ``(initial, final)`` pairs with one pair per group, as
        the actor uses when its encoder follows its own schedule.
        """
        if final is None:
            schedules = list(initial)
        else:
            schedules = [(initial, final)] * len(optimizer.param_groups)
        for group, (start, end) in zip(optimizer.param_groups, schedules):
            group["lr"] = self._learning_rate(start, end)

    def warm_start_actor_encoder(self):
        """Copy the critic's lidar encoder into the actor and its target.

        The critic encoder trains every step at the critic rate while the
        actor encoder trains once per ``policy_freq`` steps at the actor
        rate, so the critic's lidar features are usable much earlier. A
        one-time copy tests whether the actor's own features otherwise
        arrive too late to shape the policy. Returns ``False`` when the
        encoder is disabled.
        """
        if self.actor.encoder is None:
            return False
        encoder_state = self.critic.encoder.state_dict()
        self.actor.encoder.load_state_dict(encoder_state)
        self.actor_target.encoder.load_state_dict(encoder_state)
        return True

    def train(self, replay_buffer, batch_size=512):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self._apply_learning_rate(
            self.critic_optimizer, self.critic_lr, self.critic_lr_final
        )
        self.critic_optimizer.step()
        # Delayed policy updates
        policy_step = self.total_it % self.policy_freq == 0
        if policy_step:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self._apply_learning_rate(
                self.actor_optimizer, self.actor_lr_schedule
            )
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        elif self.actor_encoder_every_step and self.actor.encoder is not None:
            # Non-policy update: train only the actor's lidar encoder from the
            # actor loss. ``zero_grad(set_to_none=True)`` clears the head's
            # gradients so Adam skips those parameters and they keep the
            # values from the last policy step.
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward(inputs=list(self.actor.encoder.parameters()))
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.encoder.parameters(), self.grad_clip
                )
            self._apply_learning_rate(
                self.actor_optimizer, self.actor_lr_schedule
            )
            self.actor_optimizer.step()
            # The target policy uses this encoder, so it has to track it.
            for param, target_param in zip(
                self.actor.encoder.parameters(),
                self.actor_target.encoder.parameters(),
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename):
        checkpoints = {
            "_critic.pth": self.critic.state_dict(),
            "_critic_optimizer.pth": self.critic_optimizer.state_dict(),
            "_actor.pth": self.actor.state_dict(),
            "_actor_optimizer.pth": self.actor_optimizer.state_dict(),
        }
        # Write each artifact to a temp file, then move it into place, so a
        # crash mid-save cannot leave a mixed (partially updated) checkpoint.
        for suffix, state_dict in checkpoints.items():
            tmp_path = filename + suffix + ".tmp"
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, filename + suffix)

    def load(self, filename):
        self.critic.load_state_dict(
            torch.load(filename + "_critic.pth", map_location=device)
        )
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
