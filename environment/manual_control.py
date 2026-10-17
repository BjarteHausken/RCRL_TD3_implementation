import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from environment.dynamics import DoubleIntegratorExactDiscretization
from environment.environment import CorridorEnvironment

# Disable s as save button
plt.rcParams["keymap.save"] = []

dynamics = DoubleIntegratorExactDiscretization(dt=0.1, max_velocity=5.0)
env = CorridorEnvironment(dynamics=dynamics, length=30.0, width=5.0, max_acceleration=1.0)

pressed_keys = set()

def on_key_press(event):
    pressed_keys.add(event.key)

    if event.key == "r":
        env.reset()

def on_key_release(event):
    pressed_keys.discard(event.key)

def get_action():
    action = np.zeros(2, dtype=np.float32)

    if "d" in pressed_keys:
        action[0] += 1.0
    if "a" in pressed_keys:
        action[0] -= 1.0
    if "w" in pressed_keys:
        action[1] += 1.0
    if "s" in pressed_keys:
        action[1] -= 1.0
    
    norm = np.linalg.norm(action)

    if norm > 1.0:
        action /= norm

    return action

def draw_environment():
    ax.clear()

    x, x_dot, y, y_dot = env.dynamics.get_state()

    # Corridor walls
    ax.plot(
        [0, env.length],
        [env.width / 2, env.width / 2],
        color="black",
    )

    ax.plot(
        [0, env.length],
        [-env.width / 2, -env.width / 2],
        color="black",
    )

    # Front wall
    ax.plot(
        [0, 0],
        [-env.width/2, env.width/2],
        color="black",
    )
    # Pillars
    for pillar_x, pillar_y, pillar_radius in env.pillars:
        pillar = Circle((pillar_x, pillar_y), pillar_radius, color="gray")
        ax.add_patch(pillar)

    # LiDAR
    lidar = env.get_lidar()
    
    angles = np.linspace(0, 2 * np.pi, env.num_lidar_rays, endpoint=False)

    for angle, distance in zip(angles, lidar):
        end_x = x + distance * np.cos(angle)
        end_y = y + distance * np.sin(angle)

        if distance < env.lidar_range:
            color = "red"
            ax.scatter(end_x, end_y, color="red", s=10, zorder=4)
        else:
            color = "green"

        ax.plot([x, end_x], [y, end_y], color=color, alpha=0.25, linewidth=0.7)

    # Vehicle
    vehicle = Circle((x,y), env.vehicle_radius, color="blue", zorder=5)
    ax.add_patch(vehicle)

    # Velocity vector
    ax.arrow(
        x, y, 
        x_dot, y_dot,
        color="orange",
        width=0.02,
        head_width=0.15,
        length_includes_head=True,
        zorder=6,
    )

    # Information
    speed = np.sqrt(x_dot**2 + y_dot**2)
    constraints = env.get_lidar_constraint()

    ax.set_title(
        f"Speed: {speed:.2f} m/s | "
        f"vx: {x_dot:.2f} | "
        f"vy: {y_dot:.2f} | "
        f"h(x): {constraints:.2f} | "
    )
    ax.set_xlim(-1, env.length+1)

    ax.set_ylim(-env.width / 2 - 0.5, env.width / 2 + 0.5)

    ax.set_aspect("equal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.canvas.draw_idle()

fig, ax = plt.subplots(figsize=(12, 4))

fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)

plt.ion()
plt.show(block=False)

while plt.fignum_exists(fig.number):
    action = get_action()
    
    observation, reward, terminated, truncated, info = env.step(action)

    draw_environment()

    # Collision
    if info["collision"]:
        ax.set_title("Collision!", color="red")
        plt.pause(1.0)
        env.reset()

    # Goal
    elif info["goal_reached"]:
        ax.set_title("Goal reached!", color="green")
        plt.pause(1.0)
        env.reset()

    plt.pause(env.dynamics.dt)

plt.ioff()


