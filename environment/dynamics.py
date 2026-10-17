import numpy as np

class DoubleIntegrator:
    def __init__(self, dt, max_velocity=3.0):
        self.dt = dt
        self.state = np.array([1,0,1,0])
        self.max_velocity = max_velocity

    def update(self, u):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_state(self):
        return self.state
    
    def reset(self, state=None):
        if state is None:
            self.state = np.array([1.0, 0.0, 1.0, 0.0])
        else:
            self.state = state

        return self.state

class DoubleIntegratorEuler(DoubleIntegrator):
    def update(self, u):
        x, x_dot, y, y_dot = self.state

        # Position update
        x += x_dot * self.dt
        y += y_dot * self.dt

        # Velocity update
        new_velocity = np.array([x_dot + u[0] * self.dt, y_dot + u[1] * self.dt])

        speed = np.linalg.norm(new_velocity)

        if speed > self.max_velocity:
            new_velocity = (new_velocity / speed) * self.max_velocity

        self.state = np.array([x, new_velocity[0], y, new_velocity[1]])

class DoubleIntegratorExactDiscretization(DoubleIntegrator):
    def update(self, u):
        x, x_dot, y, y_dot = self.state

        # Velocity update
        new_velocity = np.array([x_dot + u[0] * self.dt, y_dot + u[1] * self.dt])

        speed = np.linalg.norm(new_velocity)

        if speed > self.max_velocity:
            new_velocity = (new_velocity / speed) * self.max_velocity

        new_x_dot = new_velocity[0]
        new_y_dot = new_velocity[1]

        # Position update
        x += 0.5 * (x_dot + new_x_dot) * self.dt
        y += 0.5 * (y_dot + new_y_dot) * self.dt

        self.state = np.array([x, new_x_dot, y, new_y_dot])

 
