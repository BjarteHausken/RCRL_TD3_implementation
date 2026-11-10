"""Learned encoders that summarise the lidar ring before the MLP networks.

The observation is ``[vx, vy, lidar..., goal_dx, goal_dy]``. The lidar block is
a sampled full 360 degree ring, so a 1-D convolution with circular padding can
use its wrap-around structure instead of treating every ray as an independent
input. ``LidarEncoder`` replaces the rays with a small number of angular bins;
``LidarStateEncoder`` returns the full encoded observation with the velocity
and goal-relative pairs left untouched.
"""

import torch
import torch.nn as nn


class LidarEncoder(nn.Module):
    """Two-layer circular 1-D CNN from ``num_rays`` lidar rays to ``embed_dim``.

    The first convolution expands the single range channel to ``channels``
    features and the second collapses it back to one value per bin. Input
    positions are strided by ``num_rays // embed_dim``, so bin centres are one
    stride apart and neighbouring bins share part of their kernel window.
    With the default 64 rays and 16 bins the centres are 4 rays (22.5 degrees)
    apart and each output depends on a 9-ray receptive field.
    """

    def __init__(self, num_rays=64, embed_dim=16, channels=16, kernel_size=5):
        super().__init__()
        if num_rays < 1 or embed_dim < 1:
            raise ValueError("num_rays and embed_dim must be positive")
        if num_rays % embed_dim:
            raise ValueError(
                f"num_rays={num_rays} is not divisible by embed_dim={embed_dim}"
            )
        self.num_rays = int(num_rays)
        self.embed_dim = int(embed_dim)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.stride = self.num_rays // self.embed_dim
        padding = self.kernel_size // 2
        output_length = (
            (self.num_rays + 2 * padding - self.kernel_size) // self.stride + 1
        )
        if output_length != self.embed_dim:
            raise ValueError(
                f"conv geometry yields {output_length} bins for num_rays="
                f"{self.num_rays}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}; expected embed_dim={self.embed_dim}"
            )
        self.net = nn.Sequential(
            nn.Conv1d(
                1,
                self.channels,
                self.kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            nn.ELU(),
            nn.Conv1d(
                self.channels,
                1,
                self.kernel_size,
                stride=self.stride,
                padding=padding,
                padding_mode="circular",
            ),
            nn.ELU(),
        )

    def forward(self, lidar):
        leading = lidar.shape[:-1]
        values = self.net(lidar.reshape(-1, self.num_rays).unsqueeze(1))
        return values.reshape(*leading, self.embed_dim)


class LidarStateEncoder(nn.Module):
    """Replace the lidar block of an observation with its angular embedding.

    ``[vx, vy, lidar..., goal_dx, goal_dy]`` becomes
    ``[vx, vy, z, goal_dx, goal_dy]``, where ``z`` has ``embed_dim`` entries.
    ``input_dim`` is the size of the encoded state and is the input size the
    MLP networks should use.
    """

    def __init__(
        self,
        state_dim=68,
        num_lidar_rays=64,
        embed_dim=16,
        channels=16,
        kernel_size=5,
    ):
        super().__init__()
        if state_dim != num_lidar_rays + 4:
            raise ValueError(
                f"state_dim={state_dim} does not match the observation layout "
                f"[vx, vy, lidar({num_lidar_rays}), goal_dx, goal_dy]"
            )
        self.num_rays = int(num_lidar_rays)
        self.embed_dim = int(embed_dim)
        self.input_dim = 2 + self.embed_dim + 2
        self.lidar = LidarEncoder(
            self.num_rays, self.embed_dim, channels, kernel_size
        )

    def forward(self, state):
        leading = state.shape[:-1]
        state = state.reshape(-1, state.shape[-1])
        encoded = torch.cat(
            (
                state[:, :2],
                self.lidar(state[:, 2:2 + self.num_rays]),
                state[:, -2:],
            ),
            dim=-1,
        )
        return encoded.reshape(*leading, self.input_dim)
