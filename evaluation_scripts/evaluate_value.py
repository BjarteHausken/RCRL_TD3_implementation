"""Interactive reachability-value explorer for an RCRL-TD3 checkpoint.

The RCRL safety critic learns the action-conditioned value ``Q_h(s, a)``.
For a policy ``pi`` this script plots the policy-conditioned reachability
value

    V_h(s) = Q_h(s, pi(s))

over the corridor for a fixed velocity slice (speed and heading).  Positive
values predict a future constraint violation (unsafe); nonpositive values
predict a persistently safe trajectory.

The state observation matches ``CorridorEnvironment._get_observation``
exactly: velocity, lidar distances, and goal-relative coordinates, scaled by
``env.obs_scale`` when the checkpoint was trained with ``--normalize-obs``.

Usage:
    python evaluation_scripts/evaluate_value.py --name my_rcrl_run
    python evaluation_scripts/evaluate_value.py --name my_rcrl_run --layout-index 3 --speed 1.5
    python evaluation_scripts/evaluate_value.py --name my_rcrl_run --save value.png

Hover the map to read the value at the cursor; left click pins the state
marker whose value, raw constraint, and policy action are shown in the
upper-left corner.
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"
SAFE = "#006300"
UNSAFE = "#d03b3b"
VELOCITY = "#2a78d6"
PILLAR = "#c3c2b7"
GOAL = "#0ca30c"

# Diverging value ramp: six blue steps (safe side), a neutral gray midpoint,
# and six red steps (unsafe side), so both arms carry the same number of
# steps and zero reads as "nothing".
VALUE_RAMP = [
    "#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb",
    "#f0efec",
    "#f7d3d2", "#efabaa", "#e77e7c", "#e34948", "#b62b2b", "#7f1c1c",
]


def make_value_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("rcrl_reachability", VALUE_RAMP, N=256)


def make_grid(env, resolution):
    """Square-cell grid of feasible sensor positions on the current layout."""
    margin = env.vehicle_radius + 0.05
    xs = np.linspace(margin, env.length - margin, max(2, resolution))
    cell = xs[1] - xs[0]
    ny = max(2, int(round((env.width - 2.0 * margin) / cell)) + 1)
    ys = np.linspace(-env.width / 2.0 + margin, env.width / 2.0 - margin, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, positions


def observation_batch(env, positions, velocity, goal, lidar=None):
    """Observations for arbitrary positions, matching _get_observation.

    ``lidar`` can supply precomputed ``env.lidar_at(positions)`` distances;
    they depend only on the layout, so the map grid caches them and avoids a
    fresh cast every time the velocity or goal changes.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    velocity = np.asarray(velocity, dtype=np.float32).reshape(2)
    relative_goal = goal[None, :] - positions
    if lidar is None:
        lidar = env.lidar_at(positions)
    observations = np.concatenate(
        (
            np.broadcast_to(velocity, (positions.shape[0], 2)),
            lidar.astype(np.float32),
            relative_goal.astype(np.float32),
        ),
        axis=1,
    )
    if env.normalize_obs:
        observations = observations / env.obs_scale
    return observations


def evaluate_states(policy, observations, chunk_size=8192):
    """Policy action and reachability value for each observation row."""
    device = next(policy.actor.parameters()).device
    num_states = observations.shape[0]
    values = np.empty(num_states, dtype=np.float64)
    actions = np.empty(
        (num_states, policy.actor.l3.out_features), dtype=np.float64
    )
    for start in range(0, num_states, chunk_size):
        end = min(start + chunk_size, num_states)
        state = torch.as_tensor(
            observations[start:end], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            action = policy.actor(state)
            if policy.safety_twin:
                value = policy.safety_critic.h1(state, action)
            else:
                value = policy.safety_critic(state, action)
        values[start:end] = value.cpu().numpy().ravel()
        actions[start:end] = action.cpu().numpy()
    return values, actions


def collision_mask(env, positions):
    """True where the vehicle footprint overlaps a wall or pillar."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    x = positions[:, 0]
    y = positions[:, 1]
    radius = env.vehicle_radius
    mask = (
        (y + radius >= env.width / 2.0)
        | (y - radius <= -env.width / 2.0)
        | (x - radius <= 0.0)
        | (x + radius >= env.length)
    )
    if len(env.pillars) > 0:
        pillar_x, pillar_y, pillar_radius = env.pillars.T
        distance = np.hypot(
            x[:, None] - pillar_x[None, :], y[:, None] - pillar_y[None, :]
        )
        mask |= (distance <= (radius + pillar_radius)[None, :]).any(axis=1)
    return mask


class ValueExplorer:
    """Matplotlib view of V_h(s) over corridor positions for one velocity."""

    def __init__(
        self,
        policy,
        env,
        run_name,
        config,
        layout_type="test",
        layout_index=0,
        resolution=240,
        state_position=None,
        speed=None,
        heading=None,
        goal=None,
        vmax=None,
        hover=True,
    ):
        import matplotlib.pyplot as plt

        self.policy = policy
        self.env = env
        self.run_name = run_name
        self.layout_type = layout_type
        self.layout_index = layout_index
        self.resolution = resolution
        self.override_goal = None if goal is None else np.asarray(goal, dtype=float)
        # A goal the user chose (via CLI or sliders) survives layout changes;
        # an untouched goal is resampled with the new layout.
        self.goal_locked = goal is not None
        self.override_vmax = vmax
        self.hover_enabled = hover
        self.reset_margin = config.get("random_reset_margin", 0.3)
        self.reset_max_speed = config.get("random_reset_max_speed", 1.0)
        self.min_start_goal_distance = config.get("min_start_goal_distance")
        if self.min_start_goal_distance is None:
            self.min_start_goal_distance = 2.0 * env.goal_radius
        self._suspend = False
        self._last_hover_time = 0.0

        self.layouts = {
            "train": env.train_layouts,
            "test": env.test_layouts,
            "val": env.val_layouts,
        }[layout_type]
        if not 0 <= layout_index < len(self.layouts):
            raise ValueError(
                f"layout index must be in [0, {len(self.layouts) - 1}] "
                f"for the {layout_type!r} set"
            )

        # A sampled feasible start supplies the initial marker and velocity,
        # and a sampled goal (unless overridden) matches the training regime.
        state = self._sample_episode(layout_index)
        sampled_velocity = np.array([state[1], state[3]], dtype=float)
        sampled_speed = float(np.hypot(*sampled_velocity))
        sampled_heading = (
            float(np.degrees(np.arctan2(sampled_velocity[1], sampled_velocity[0])))
            % 360.0
            if sampled_speed > 1e-6
            else 0.0
        )
        self.speed = sampled_speed if speed is None else float(speed)
        self.heading = sampled_heading if heading is None else float(heading) % 360.0
        self.marker = (
            np.array([state[0], state[2]], dtype=float)
            if state_position is None
            else np.asarray(state_position, dtype=float)
        )
        self.goal = (
            self.env.goal.copy() if goal is None else self.override_goal.copy()
        )
        self.env.goal = self.goal.copy()

        self.grid_x, self.grid_y, self.positions = make_grid(env, resolution)
        self.grid_lidar = env.lidar_at(self.positions)
        self.cmap = make_value_colormap()
        self._build_figure(plt)
        self._refresh_slice()

    def _sample_episode(self, layout_index):
        self.env.reset(
            layout_type=self.layout_type,
            layout_index=layout_index,
            seed=layout_index,
            sample_start=True,
            start_margin=self.reset_margin,
            start_max_speed=self.reset_max_speed,
            goal_margin=self.reset_margin,
            min_start_goal_distance=self.min_start_goal_distance,
        )
        # reset() returns the observation; the sampled dynamics state is the
        # (x, x_dot, y, y_dot) vector stored on the environment.
        return self.env.dynamics.get_state().copy()

    def _velocity(self):
        angle = np.radians(self.heading)
        return self.speed * np.array([np.cos(angle), np.sin(angle)])

    def _slice_values(self):
        values, _ = evaluate_states(
            self.policy,
            observation_batch(
                self.env,
                self.positions,
                self._velocity(),
                self.goal,
                lidar=self.grid_lidar,
            ),
        )
        values[collision_mask(self.env, self.positions)] = np.nan
        return values

    def _value_bound(self, values):
        if self.override_vmax is not None:
            return float(self.override_vmax)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 1.0
        bound = float(np.max(np.abs(finite)))
        return bound * 1.05 if bound > 0.0 else 1.0

    def _build_figure(self, plt):
        from matplotlib.patches import Circle, FancyArrowPatch
        from matplotlib.widgets import Slider

        self.fig = plt.figure(figsize=(15, 6.8), facecolor=SURFACE)
        self.ax = self.fig.add_axes([0.045, 0.46, 0.80, 0.44])
        self.ax.set_facecolor(SURFACE)

        # Corridor walls.
        wall_color = AXIS
        for xs, ys in (
            ([0.0, self.env.length], [self.env.width / 2.0] * 2),
            ([0.0, self.env.length], [-self.env.width / 2.0] * 2),
            ([0.0, 0.0], [-self.env.width / 2.0, self.env.width / 2.0]),
            ([self.env.length] * 2, [-self.env.width / 2.0, self.env.width / 2.0]),
        ):
            self.ax.plot(xs, ys, color=wall_color, lw=1.5, zorder=3)

        self.pillar_patches = []
        self._draw_pillars()

        self.goal_patch = Circle(
            tuple(self.goal),
            self.env.goal_radius,
            facecolor=GOAL,
            edgecolor=GOAL,
            alpha=0.35,
            lw=1.5,
            zorder=4,
        )
        self.ax.add_patch(self.goal_patch)
        self.goal_label = self.ax.text(
            self.goal[0],
            self.goal[1] + self.env.goal_radius + 0.18,
            "goal",
            color=SAFE,
            fontsize=9,
            ha="center",
            va="bottom",
            zorder=5,
        )

        # Heatmap. The array is replaced in place on every velocity change.
        initial_values = self._slice_values()
        from matplotlib.colors import Normalize

        bound = self._value_bound(initial_values)
        self.mesh = self.ax.pcolormesh(
            self.grid_x,
            self.grid_y,
            np.ma.masked_invalid(initial_values.reshape(self.grid_x.shape)),
            shading="auto",
            cmap=self.cmap,
            norm=Normalize(vmin=-bound, vmax=bound),
            zorder=1,
            rasterized=True,
        )
        self.cbar = self.fig.colorbar(
            self.mesh, ax=self.ax, fraction=0.03, pad=0.012
        )
        self.cbar.set_label(
            "V_h(s) = Q_h(s, π(s))   (positive = unsafe)",
            color=INK_SECONDARY,
            fontsize=9,
        )
        self.cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
        self.cbar.outline.set_edgecolor(AXIS)

        # Selected state: vehicle ring plus its state velocity.
        self.vehicle_patch = Circle(
            tuple(self.marker),
            self.env.vehicle_radius,
            facecolor="none",
            edgecolor=INK,
            lw=1.8,
            zorder=6,
        )
        self.ax.add_patch(self.vehicle_patch)
        self.velocity_arrow = FancyArrowPatch(
            tuple(self.marker),
            tuple(self.marker + self._velocity()),
            arrowstyle="-|>",
            mutation_scale=13,
            color=VELOCITY,
            lw=2.0,
            zorder=7,
        )
        self.ax.add_patch(self.velocity_arrow)
        self.velocity_label = self.ax.text(
            self.marker[0],
            self.marker[1],
            "v",
            color=VELOCITY,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=7,
        )

        box = dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor="none",
                   alpha=0.88)
        self.value_text = self.ax.text(
            0.012, 0.975, "", transform=self.ax.transAxes, color=INK,
            fontsize=10.5, va="top", ha="left", zorder=8, bbox=box,
        )
        self.status_text = self.ax.text(
            0.012, 0.865, "", transform=self.ax.transAxes, color=INK,
            fontsize=10.5, va="top", ha="left", zorder=8, bbox=box,
        )
        self.detail_text = self.ax.text(
            0.012, 0.755, "", transform=self.ax.transAxes, color=INK_MUTED,
            fontsize=9, va="top", ha="left", zorder=8, bbox=box,
        )
        self.hover_text = self.ax.text(
            0.988, 0.02, "", transform=self.ax.transAxes, color=INK_MUTED,
            fontsize=9, va="bottom", ha="right", zorder=8, bbox=box,
        )

        self.ax.set_xlim(-0.5, self.env.length + 0.5)
        self.ax.set_ylim(-self.env.width / 2.0 - 1.0, self.env.width / 2.0 + 1.0)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x [m]", color=INK_MUTED)
        self.ax.set_ylabel("y [m]", color=INK_MUTED)
        self.ax.tick_params(colors=INK_MUTED, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(AXIS)
        self.ax.grid(False)
        self.ax.set_title("", color=INK, fontsize=12, loc="left", pad=10)

        # Sliders: velocity slice, goal, layout.
        slider_left, slider_width = 0.10, 0.55
        slider_top, slider_step, slider_height = 0.36, 0.060, 0.026

        def slider_axes(index):
            position = [
                slider_left,
                slider_top - index * slider_step,
                slider_width,
                slider_height,
            ]
            axes = self.fig.add_axes(position, facecolor=SURFACE)
            axes.tick_params(colors=INK_MUTED, labelsize=8)
            for spine in axes.spines.values():
                spine.set_color(AXIS)
            return axes

        def style_slider(slider):
            # Place labels above the track, left-aligned: the default label
            # sits left of the track and long labels run off the figure.
            slider.label.set_position((0.0, 1.08))
            slider.label.set_horizontalalignment("left")
            slider.label.set_verticalalignment("bottom")
            slider.label.set_color(INK_SECONDARY)
            slider.label.set_fontsize(9)
            slider.valtext.set_color(INK_SECONDARY)
            slider.valtext.set_fontsize(9)
            return slider

        max_speed = float(self.env.dynamics.max_velocity)
        self.speed_slider = style_slider(Slider(
            slider_axes(0), "speed |v| [m/s]", 0.0, max_speed,
            valinit=self.speed, valstep=0.01, valfmt="%.2f", color=VELOCITY,
        ))
        self.heading_slider = style_slider(Slider(
            slider_axes(1), "heading [deg from +x]", 0.0, 360.0,
            valinit=self.heading, valstep=1.0, valfmt="%.0f", color=VELOCITY,
        ))
        self.goal_x_slider = style_slider(Slider(
            slider_axes(2), "goal x [m]",
            self.env.vehicle_radius + 0.05,
            self.env.length - self.env.vehicle_radius - 0.05,
            valinit=self.goal[0], valstep=0.05, valfmt="%.2f", color=GOAL,
        ))
        self.goal_y_slider = style_slider(Slider(
            slider_axes(3), "goal y [m]",
            -self.env.width / 2.0 + self.env.vehicle_radius + 0.05,
            self.env.width / 2.0 - self.env.vehicle_radius - 0.05,
            valinit=self.goal[1], valstep=0.05, valfmt="%.2f", color=GOAL,
        ))
        self.layout_slider = style_slider(Slider(
            slider_axes(4), f"layout ({self.layout_type})", 0,
            len(self.layouts) - 1, valinit=self.layout_index,
            valstep=1.0, valfmt="%d", color=AXIS,
        ))

        self.fig.text(
            0.70, 0.345,
            "left click: pin state\nhover: read value\nlayout slider: switch pillars",
            color=INK_MUTED, fontsize=9, va="top", ha="left",
        )

        self.speed_slider.on_changed(self._on_velocity_changed)
        self.heading_slider.on_changed(self._on_velocity_changed)
        self.goal_x_slider.on_changed(self._on_goal_changed)
        self.goal_y_slider.on_changed(self._on_goal_changed)
        self.layout_slider.on_changed(self._on_layout_changed)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)

    def _draw_pillars(self):
        from matplotlib.patches import Circle

        for patch in self.pillar_patches:
            patch.remove()
        self.pillar_patches = []
        for pillar_x, pillar_y, pillar_radius in self.env.pillars:
            patch = Circle(
                (pillar_x, pillar_y),
                pillar_radius,
                facecolor=PILLAR,
                edgecolor=AXIS,
                lw=0.8,
                zorder=3,
            )
            self.ax.add_patch(patch)
            self.pillar_patches.append(patch)

    def _on_velocity_changed(self, _value):
        if self._suspend:
            return
        self.speed = float(self.speed_slider.val)
        self.heading = float(self.heading_slider.val) % 360.0
        self._refresh_slice()

    def _on_goal_changed(self, _value):
        if self._suspend:
            return
        self.goal = np.array(
            [float(self.goal_x_slider.val), float(self.goal_y_slider.val)]
        )
        self.goal_locked = True
        self.env.goal = self.goal.copy()
        self.goal_patch.center = tuple(self.goal)
        self.goal_label.set_position(
            (self.goal[0], self.goal[1] + self.env.goal_radius + 0.18)
        )
        self._refresh_slice()

    def _on_layout_changed(self, value):
        if self._suspend:
            return
        layout_index = int(round(value))
        if layout_index == self.layout_index:
            return
        self.layout_index = layout_index

        state = self._sample_episode(layout_index)
        self.grid_lidar = self.env.lidar_at(self.positions)
        if not self.goal_locked:
            self.goal = self.env.goal.copy()
        self.env.goal = self.goal.copy()
        self.marker = np.array([state[0], state[2]], dtype=float)

        self._suspend = True
        self.goal_x_slider.set_val(self.goal[0])
        self.goal_y_slider.set_val(self.goal[1])
        self._suspend = False

        self._draw_pillars()
        self.goal_patch.center = tuple(self.goal)
        self.goal_label.set_position(
            (self.goal[0], self.goal[1] + self.env.goal_radius + 0.18)
        )
        self._refresh_slice()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.marker = np.array([event.xdata, event.ydata], dtype=float)
        self._update_marker()
        self.fig.canvas.draw_idle()

    def _on_move(self, event):
        if not self.hover_enabled or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        now = time.perf_counter()
        if now - self._last_hover_time < 0.03:
            return
        self._last_hover_time = now

        position = np.array([[event.xdata, event.ydata]], dtype=float)
        if collision_mask(self.env, position)[0]:
            self.hover_text.set_text("cursor: collision state")
        else:
            value, _ = evaluate_states(
                self.policy,
                observation_batch(
                    self.env, position, self._velocity(), self.goal
                ),
            )
            self.hover_text.set_text(f"cursor: V_h = {value[0]:+.3f}")
        self.fig.canvas.draw_idle()

    def _refresh_slice(self):
        values = self._slice_values()
        self.mesh.set_array(np.ma.masked_invalid(values).ravel())
        bound = self._value_bound(values)
        self.mesh.set_clim(-bound, bound)
        self.cbar.update_normal(self.mesh)

        title = (
            f"{self.run_name}  ·  V_h(s) = Q_h(s, π(s))  ·  "
            f"|v| = {self.speed:.2f} m/s, heading = {self.heading:.0f}°  ·  "
            f"{self.layout_type} layout {self.layout_index}"
        )
        self.ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
        self._update_marker()
        self.fig.canvas.draw_idle()

    def _update_marker(self):
        position = self.marker.reshape(1, 2)
        velocity = self._velocity()
        speed = float(np.hypot(*velocity))

        self.vehicle_patch.center = (self.marker[0], self.marker[1])
        self.velocity_arrow.set_positions(
            self.marker, self.marker + velocity
        )
        self.velocity_arrow.set_visible(speed > 1e-6)
        self.velocity_label.set_position(
            (self.marker[0] + velocity[0], self.marker[1] + velocity[1])
        )
        self.velocity_label.set_text("v" if speed > 1e-6 else "")

        lidar = self.env.lidar_at(position)
        raw_constraint = self.env.vehicle_radius - float(np.min(lidar))
        value, action = evaluate_states(
            self.policy,
            observation_batch(self.env, position, velocity, self.goal),
        )
        value = float(value[0])
        action = action[0]

        self.value_text.set_text(
            f"V_h(s) = {value:+.3f}    at "
            f"({self.marker[0]:.2f}, {self.marker[1]:.2f}) m"
        )
        if collision_mask(self.env, position)[0]:
            self.status_text.set_text("● collision state (not a valid start)")
            self.status_text.set_color(INK_MUTED)
        elif value > 0.0:
            self.status_text.set_text("● predicted unsafe")
            self.status_text.set_color(UNSAFE)
        else:
            self.status_text.set_text("● predicted safe")
            self.status_text.set_color(SAFE)
        self.detail_text.set_text(
            f"π(s) = ({action[0]:+.2f}, {action[1]:+.2f})    "
            f"raw h(s) = {raw_constraint:+.3f}"
        )

    def save(self, path):
        self.fig.savefig(path, dpi=160, facecolor=self.fig.get_facecolor())
        value, _ = evaluate_states(
            self.policy,
            observation_batch(
                self.env, self.marker.reshape(1, 2), self._velocity(), self.goal
            ),
        )
        print(f"Saved {path}")
        print(
            f"velocity=({self._velocity()[0]:+.3f}, {self._velocity()[1]:+.3f}) m/s "
            f"layout={self.layout_index} "
            f"goal=({self.goal[0]:.2f}, {self.goal[1]:.2f})"
        )
        print(
            f"state=({self.marker[0]:.2f}, {self.marker[1]:.2f}) "
            f"V_h(s)={value[0]:+.4f}"
        )


def main(
    run_dir,
    layout_set="one",
    normalize_obs=True,
    layout_type="test",
    layout_index=0,
    resolution=240,
    state_position=None,
    speed=None,
    heading=None,
    goal=None,
    vmax=None,
    save=None,
    hover=True,
):
    # Headless saving needs Agg before environment imports matplotlib.pyplot.
    if save is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from agents.rcrl_td3 import RCRLTD3
    from environment import create_env
    from evaluation_scripts.evaluate import agent_type_from_config, load_config

    config = load_config(run_dir)
    agent_type = agent_type_from_config(config, run_dir)
    if agent_type != "rcrl_td3":
        raise SystemExit(
            f"{os.path.join(run_dir, 'config.json')} records agent "
            f"{agent_type!r}; only RCRL-TD3 runs learn a reachability value"
        )
    model_path = os.path.join(run_dir, agent_type)

    env = create_env(
        layout_set,
        normalize_obs=normalize_obs,
        reward_scale=config.get("reward_scale", 1.0),
        max_steps=config.get("max_steps", 150),
    )

    layouts = {
        "train": env.train_layouts,
        "test": env.test_layouts,
        "val": env.val_layouts,
    }[layout_type]
    if not 0 <= layout_index < len(layouts):
        raise SystemExit(
            f"--layout-index must be in [0, {len(layouts) - 1}] for "
            f"--layout-type {layout_type}"
        )

    # Layouts may carry their own width, so validate against the selected
    # layout rather than the environment's constructor default.
    layout = layouts[layout_index]
    layout_width = (
        float(layout["width"]) if isinstance(layout, dict) else env.width
    )

    goal_margin = env.vehicle_radius + 0.05
    if goal is not None and not (
        goal_margin <= goal[0] <= env.length - goal_margin
        and -layout_width / 2.0 + goal_margin
        <= goal[1]
        <= layout_width / 2.0 - goal_margin
    ):
        raise SystemExit("--goal-x/--goal-y must lie inside the corridor")
    if state_position is not None and not (
        0.0 <= state_position[0] <= env.length
        and -layout_width / 2.0 <= state_position[1] <= layout_width / 2.0
    ):
        raise SystemExit("--state-x/--state-y must lie inside the corridor")
    if speed is not None and speed > env.dynamics.max_velocity:
        raise SystemExit(
            f"--speed must not exceed the environment limit "
            f"({env.dynamics.max_velocity} m/s)"
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = RCRLTD3(
        state_dim,
        action_dim,
        max_action,
        multiplier_type=config.get("multiplier", "scalar"),
        max_lambda=config.get("max_lambda", 100.0),
        initial_lambda=config.get("initial_lambda", 10.0),
        min_lambda=config.get("min_lambda", 0.0),
        safety_twin=config.get("safety_twin", False),
        safety_aggregation=config.get("safety_aggregation", "min"),
    )
    policy.load(model_path)
    policy.actor.eval()
    policy.safety_critic.eval()
    print("Loaded model:", model_path)

    explorer = ValueExplorer(
        policy,
        env,
        config.get("name", os.path.basename(run_dir)),
        config,
        layout_type=layout_type,
        layout_index=layout_index,
        resolution=resolution,
        state_position=state_position,
        speed=speed,
        heading=heading,
        goal=goal,
        vmax=vmax,
        hover=hover and save is None,
    )

    if save is not None:
        explorer.save(save)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive reachability-value explorer for RCRL-TD3"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run directory under models/; its config.json records the agent type",
    )
    parser.add_argument(
        "--layout-set",
        default="one",
        help="Layout set name; loads layouts/<name>_{train,test,validation}.npy",
    )
    parser.add_argument(
        "--layout-type",
        default="test",
        choices=("train", "test", "val"),
        help="Layout split to browse (default: test)",
    )
    parser.add_argument(
        "--layout-index",
        type=int,
        default=0,
        help="Initial layout index within the chosen split (default: 0)",
    )
    parser.add_argument(
        "--normalize-obs",
        dest="normalize_obs",
        action="store_true",
        default=True,
        help="Scale observation components by their known ranges (default); "
             "checkpoints trained with --no-normalize-obs must be explored "
             "with --no-normalize-obs",
    )
    parser.add_argument(
        "--no-normalize-obs",
        dest="normalize_obs",
        action="store_false",
        help="Disable observation scaling",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=240,
        help="Grid cells along the corridor length (default: 240)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Initial speed |v| in m/s (default: speed of the sampled start)",
    )
    parser.add_argument(
        "--heading",
        type=float,
        default=None,
        help="Initial heading in degrees from +x (default: sampled start "
             "velocity direction)",
    )
    parser.add_argument(
        "--state-x", type=float, default=None,
        help="Initial marker x position [m] (default: sampled start x)",
    )
    parser.add_argument(
        "--state-y", type=float, default=None,
        help="Initial marker y position [m] (default: sampled start y)",
    )
    parser.add_argument(
        "--goal-x", type=float, default=None,
        help="Initial goal x position [m] (default: sampled goal)",
    )
    parser.add_argument(
        "--goal-y", type=float, default=None,
        help="Initial goal y position [m] (default: sampled goal)",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Fix the symmetric color scale to +/- VMAX instead of rescaling "
             "on every velocity change",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save a single headless figure to PATH instead of opening a window",
    )

    args = parser.parse_args()

    if args.resolution < 2:
        parser.error("--resolution must be at least 2")
    if args.speed is not None and args.speed < 0.0:
        parser.error("--speed must be nonnegative")
    if args.vmax is not None and args.vmax <= 0.0:
        parser.error("--vmax must be positive")
    if (args.state_x is None) != (args.state_y is None):
        parser.error("--state-x and --state-y must be given together")
    if (args.goal_x is None) != (args.goal_y is None):
        parser.error("--goal-x and --goal-y must be given together")

    state_position = (
        None
        if args.state_x is None
        else np.array([args.state_x, args.state_y], dtype=float)
    )
    goal = (
        None
        if args.goal_x is None
        else np.array([args.goal_x, args.goal_y], dtype=float)
    )

    main(
        os.path.join("models", args.name),
        args.layout_set,
        args.normalize_obs,
        args.layout_type,
        args.layout_index,
        args.resolution,
        state_position,
        args.speed,
        args.heading,
        goal,
        args.vmax,
        args.save,
    )
