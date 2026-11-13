"""Learned encoders that summarise the lidar ring before the MLP networks.

The observation is ``[velocity(d), lidar(N), goal(d)]`` (``d = 2`` today:
``[vx, vy, lidar..., goal_dx, goal_dy]``). The lidar block is a set of range
measurements with known ray directions, so it can be summarised without any
reference to a 2-D ring. ``LidarEncoder`` replaces the rays with a small
number of angular bins (circular 1-D CNN or a dense layer); ``RayEncoder``
applies one shared MLP to every range and pools the results invariantly and
equivariantly, which works for any number of rays and any spatial dimension.
``LidarStateEncoder`` returns the full encoded observation with the velocity
and goal-relative blocks left untouched.
"""

import math

import torch
import torch.nn as nn


class LidarEncoder(nn.Module):
    """Compress ``num_rays`` lidar rays to ``embed_dim`` angular features.

    Two structures are available through ``kind``:

    ``"conv"`` (default) is the two-layer circular 1-D CNN. The first
    convolution expands the single range channel to ``channels`` features and
    the second collapses it back to one value per bin. Input positions are
    strided by ``num_rays // embed_dim``, so bin centres are one stride apart
    and neighbouring bins share part of their kernel window. With the default
    64 rays and 16 bins the centres are 4 rays (22.5 degrees) apart and each
    output depends on a 9-ray receptive field.

    ``"linear"`` is a single dense layer over all rays. It keeps the same
    bottleneck but drops the weight sharing and the fixed angular window, so
    every bin can weight every ray. Comparing the two separates the cost of
    the circular-convolution prior from the cost of the compression itself.
    """

    def __init__(self, num_rays=64, embed_dim=16, channels=16, kernel_size=5,
                 kind="conv"):
        super().__init__()
        if num_rays < 1 or embed_dim < 1:
            raise ValueError("num_rays and embed_dim must be positive")
        if kind not in ("conv", "linear"):
            raise ValueError(f"unknown lidar encoder kind {kind!r}")
        self.num_rays = int(num_rays)
        self.embed_dim = int(embed_dim)
        self.kind = kind
        if kind == "linear":
            self.net = nn.Sequential(
                nn.Linear(self.num_rays, self.embed_dim),
                nn.ELU(),
            )
            return
        if num_rays % embed_dim:
            raise ValueError(
                f"num_rays={num_rays} is not divisible by embed_dim={embed_dim}"
            )
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
        values = lidar.reshape(-1, self.num_rays)
        if self.kind == "conv":
            values = values.unsqueeze(1)
        values = self.net(values)
        return values.reshape(*leading, self.embed_dim)


class RayEncoder(nn.Module):
    """Dimension-agnostic learned summary of a set of range measurements.

    Every ray ``i`` has a range ``r_i`` (rotation-invariant) and a unit
    direction ``u_i in R^d`` fixed in the sensor frame. A single MLP
    ``phi: R -> R^C`` is applied to every range with shared weights, and its
    outputs are pooled in two ways:

    - the first ``C // 2`` channels are averaged over rays, giving
      ``mean_i phi(r_i)[:, :inv]`` - invariant to ray order and rotation;
    - the remaining ``C - C // 2`` channels act as weights on the direction
      vectors, ``vec_p = (1/N) sum_i phi(r_i)[:, inv+p] * u_i``. Under a
      rotation ``Q`` of the world every ``vec_p`` becomes ``Q vec_p``, so the
      information is preserved equivariantly while its magnitude
      ``||vec_p||`` stays invariant.

    This replaces the circular convolution's fixed angular window and stride
    pooling with weight sharing across rays and symmetry-correct aggregation:
    no angular ordering is ever used, so the module runs unchanged for a 2-D
    ring, a 3-D sensor, or any ``d``-dimensional range sensor.

    With ``exact_min`` the exact minimum range and the direction of its
    argmin ray are appended. The constraint ``h = vehicle_radius - min(lidar)``
    is built on that statistic, and a learned pool cannot be relied on to
    reproduce it exactly; the skip costs ``1 + d`` inputs and uses only
    ``min``/``argmin`` plus a buffer lookup, so it is dimension-agnostic too.
    """

    def __init__(self, num_rays=64, spatial_dim=2, channels=16,
                 exact_min=True, directions=None):
        super().__init__()
        if num_rays < 1:
            raise ValueError("num_rays must be positive")
        if channels < 2:
            raise ValueError(
                "channels must be at least 2 so that the invariant and "
                "equivariant blocks are both non-empty"
            )
        self.num_rays = int(num_rays)
        self.spatial_dim = int(spatial_dim)
        if self.spatial_dim < 1:
            raise ValueError("spatial_dim must be positive")
        self.channels = int(channels)
        self.exact_min = bool(exact_min)
        self.invariant_dim = self.channels // 2
        self.vector_dim = self.channels - self.invariant_dim

        if directions is None:
            if self.spatial_dim != 2:
                raise ValueError(
                    "directions must be supplied for spatial_dim != 2; the "
                    "built-in ring is a 2-D construction"
                )
            # Same ring as the environment's ``np.linspace(0, 2*pi,
            # num_rays, endpoint=False)``.
            angles = (
                torch.arange(self.num_rays, dtype=torch.float32)
                * (2.0 * math.pi / self.num_rays)
            )
            directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        else:
            directions = torch.as_tensor(directions, dtype=torch.float32)
            if directions.shape != (self.num_rays, self.spatial_dim):
                raise ValueError(
                    f"directions must have shape ({self.num_rays}, "
                    f"{self.spatial_dim}), got {tuple(directions.shape)}"
                )
            if not torch.isfinite(directions).all():
                raise ValueError("directions must be finite")
            norms = directions.norm(dim=-1)
            if (norms == 0).any():
                raise ValueError("directions must be nonzero")
            directions = directions / norms.unsqueeze(-1)
        self.register_buffer("directions", directions)

        self.per_ray = nn.Sequential(
            nn.Linear(1, self.channels),
            nn.ELU(),
            nn.Linear(self.channels, self.channels),
            nn.ELU(),
        )
        self.output_dim = (
            self.invariant_dim
            + self.vector_dim * self.spatial_dim
            + self.vector_dim
            + ((1 + self.spatial_dim) if self.exact_min else 0)
        )

    def forward(self, lidar):
        leading = lidar.shape[:-1]
        values = lidar.reshape(-1, self.num_rays)
        hidden = self.per_ray(values.unsqueeze(-1))

        invariant = hidden[..., :self.invariant_dim].mean(dim=-2)
        weights = hidden[..., self.invariant_dim:]
        vectors = torch.einsum("bnp,nd->bpd", weights, self.directions)
        vectors = vectors / self.num_rays
        magnitudes = vectors.norm(dim=-1)
        parts = [
            invariant,
            vectors.reshape(-1, self.vector_dim * self.spatial_dim),
            magnitudes,
        ]
        if self.exact_min:
            minimum, index = values.min(dim=-1)
            parts.append(minimum.unsqueeze(-1))
            parts.append(self.directions[index])
        encoded = torch.cat(parts, dim=-1)
        return encoded.reshape(*leading, self.output_dim)


class LidarStateEncoder(nn.Module):
    """Replace the lidar block of an observation with its learned embedding.

    ``[velocity(d), lidar..., goal(d)]`` becomes
    ``[velocity(d), z, goal(d)]``, where ``z`` is the encoder output
    (``embed_dim`` for the angular kinds, ``RayEncoder.output_dim`` for
    ``kind="ray"``). ``input_dim`` is the size of the encoded state and is the
    input size the MLP networks should use.

    With ``kind="conv"`` or ``"linear"`` the layout is the 2-D one,
    ``[vx, vy, lidar..., goal_dx, goal_dy]``; ``min_features`` then inserts
    the exact minimum range and the unit bearing of the closest ray after ``z``:

    ``[vx, vy, z, min_range, cos(bearing), sin(bearing), goal_dx, goal_dy]``

    That adds three inputs (``input_dim`` becomes ``embed_dim + 7``). The
    constraint is ``h = vehicle_radius - min(lidar)``, a statistic an angular
    average can smear away, so the flag tests whether the encoder loses
    information the policy needs.

    ``kind="ray"`` is the dimension-agnostic ``RayEncoder``: the number of
    spatial dimensions ``d`` is inferred from
    ``state_dim == num_lidar_rays + 2 * d``, and ``min_features`` is ignored
    because the ray encoder already contains the exact minimum and the
    direction of its argmin ray.
    """

    def __init__(
        self,
        state_dim=68,
        num_lidar_rays=64,
        embed_dim=16,
        channels=16,
        kernel_size=5,
        kind="conv",
        min_features=False,
        spatial_dim=None,
        ray_channels=16,
        ray_exact_min=True,
        ray_directions=None,
    ):
        super().__init__()
        if kind not in ("conv", "linear", "ray"):
            raise ValueError(f"unknown lidar encoder kind {kind!r}")
        self.num_rays = int(num_lidar_rays)
        self.kind = kind
        self.is_ray = kind == "ray"
        if self.is_ray:
            if spatial_dim is None:
                spatial_dim = (state_dim - self.num_rays) // 2
            self.spatial_dim = int(spatial_dim)
            if (
                self.spatial_dim < 1
                or state_dim != self.num_rays + 2 * self.spatial_dim
            ):
                raise ValueError(
                    f"state_dim={state_dim} does not match the observation "
                    f"layout [velocity({self.spatial_dim}), "
                    f"lidar({self.num_rays}), goal({self.spatial_dim})]"
                )
        else:
            self.spatial_dim = 2
            if state_dim != self.num_rays + 4:
                raise ValueError(
                    f"state_dim={state_dim} does not match the observation "
                    f"layout [vx, vy, lidar({self.num_rays}), goal_dx, "
                    f"goal_dy]"
                )
        self.embed_dim = int(embed_dim)
        self.min_features = bool(min_features) and not self.is_ray
        if self.is_ray:
            self.lidar = RayEncoder(
                self.num_rays,
                self.spatial_dim,
                ray_channels,
                exact_min=ray_exact_min,
                directions=ray_directions,
            )
            self.input_dim = 2 * self.spatial_dim + self.lidar.output_dim
            return
        self.input_dim = 2 + self.embed_dim + 2 + (3 if self.min_features else 0)
        self.lidar = LidarEncoder(
            self.num_rays, self.embed_dim, channels, kernel_size, kind=kind
        )
        if self.min_features:
            # Registered only when used: a buffer key would otherwise break
            # strict loading of every existing encoder checkpoint.
            # Same ring as the environment's ``np.linspace(0, 2*pi,
            # num_rays, endpoint=False)``: linspace's default endpoint would
            # shift every bearing by one ray spacing.
            self.register_buffer(
                "ray_angles",
                torch.arange(self.num_rays, dtype=torch.float32)
                * (2.0 * math.pi / self.num_rays),
            )

    def forward(self, state):
        leading = state.shape[:-1]
        state = state.reshape(-1, state.shape[-1])
        d = self.spatial_dim
        lidar = state[:, d:d + self.num_rays]
        if self.is_ray:
            parts = [state[:, :d], self.lidar(lidar), state[:, -d:]]
            encoded = torch.cat(parts, dim=-1)
            return encoded.reshape(*leading, self.input_dim)
        parts = [state[:, :2], self.lidar(lidar)]
        if self.min_features:
            minimum, index = lidar.min(dim=-1)
            bearing = self.ray_angles[index]
            parts.extend(
                (
                    minimum.unsqueeze(-1),
                    bearing.cos().unsqueeze(-1),
                    bearing.sin().unsqueeze(-1),
                )
            )
        parts.append(state[:, -2:])
        encoded = torch.cat(parts, dim=-1)
        return encoded.reshape(*leading, self.input_dim)
