import numpy as np
import torch

# Replay buffer adapted from Scott Fujimoto's reference implementation
# (https://github.com/sfujim/TD3), MIT License.
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_constraint = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

class ReplayBufferRCRLTD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        boundary_threshold=None,
        boundary_max_size=None,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.h = np.zeros((max_size, 1), dtype=np.float32)
        self.next_h = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Optional sub-buffer of near-boundary transitions, used to
        # oversample the safety-critic batches. A transition is
        # near-boundary when h or next_h is above -boundary_threshold, i.e.
        # the clearance is within boundary_threshold of the collision
        # boundary (h = vehicle_radius - min_lidar) or already violating it.
        self.boundary_threshold = boundary_threshold
        self.boundary_ptr = 0
        self.boundary_size = 0
        if boundary_threshold is not None:
            self.boundary_max_size = boundary_max_size or max(max_size // 5, 1)
            self.boundary_state = np.zeros(
                (self.boundary_max_size, state_dim), dtype=np.float32
            )
            self.boundary_action = np.zeros(
                (self.boundary_max_size, action_dim), dtype=np.float32
            )
            self.boundary_next_state = np.zeros(
                (self.boundary_max_size, state_dim), dtype=np.float32
            )
            self.boundary_reward = np.zeros(
                (self.boundary_max_size, 1), dtype=np.float32
            )
            self.boundary_h = np.zeros(
                (self.boundary_max_size, 1), dtype=np.float32
            )
            self.boundary_next_h = np.zeros(
                (self.boundary_max_size, 1), dtype=np.float32
            )
            self.boundary_not_done = np.zeros(
                (self.boundary_max_size, 1), dtype=np.float32
            )

    def add(
        self,
        state,
        action,
        next_state,
        reward,
        constraint,
        next_constraint,
        done,
    ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.h[self.ptr] = constraint
        self.next_h[self.ptr] = next_constraint
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.boundary_threshold is not None and (
            constraint > -self.boundary_threshold
            or next_constraint > -self.boundary_threshold
        ):
            self._add_boundary(
                state,
                action,
                next_state,
                reward,
                constraint,
                next_constraint,
                done,
            )

    def _add_boundary(
        self,
        state,
        action,
        next_state,
        reward,
        constraint,
        next_constraint,
        done,
    ):
        self.boundary_state[self.boundary_ptr] = state
        self.boundary_action[self.boundary_ptr] = action
        self.boundary_next_state[self.boundary_ptr] = next_state
        self.boundary_reward[self.boundary_ptr] = reward
        self.boundary_h[self.boundary_ptr] = constraint
        self.boundary_next_h[self.boundary_ptr] = next_constraint
        self.boundary_not_done[self.boundary_ptr] = 1.0 - done

        self.boundary_ptr = (self.boundary_ptr + 1) % self.boundary_max_size
        self.boundary_size = min(
            self.boundary_size + 1, self.boundary_max_size
        )

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.h[ind]).to(self.device),
            torch.FloatTensor(self.next_h[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    def sample_mixed(self, batch_size, mix, warmup=10_000):
        """Draw a safety batch with a ``mix`` fraction from the boundary
        sub-buffer and the remainder drawn uniformly from the main buffer.

        The effective fraction ramps up linearly from zero to ``mix`` as the
        sub-buffer fills past ``warmup`` transitions, so early training is
        not dominated by a handful of near-boundary rows. Falls back to
        plain uniform sampling when no boundary sub-buffer is allocated.
        """
        if (
            self.boundary_threshold is None
            or mix <= 0.0
            or self.boundary_size == 0
        ):
            return self.sample(batch_size)
        effective = mix * min(1.0, self.boundary_size / float(max(warmup, 1)))
        n_boundary = min(int(round(batch_size * effective)), self.boundary_size)
        if n_boundary <= 0:
            return self.sample(batch_size)
        n_main = batch_size - n_boundary
        ind_b = np.random.randint(0, self.boundary_size, size=n_boundary)
        ind_m = np.random.randint(0, self.size, size=n_main)

        return tuple(
            torch.FloatTensor(
                np.concatenate([boundary[ind_b], main[ind_m]])
            ).to(self.device)
            for main, boundary in (
                (self.state, self.boundary_state),
                (self.action, self.boundary_action),
                (self.next_state, self.boundary_next_state),
                (self.reward, self.boundary_reward),
                (self.h, self.boundary_h),
                (self.next_h, self.boundary_next_h),
                (self.not_done, self.boundary_not_done),
            )
        )

class ReplayBufferTD3Torch(object):
    """Device-resident replay buffer for vectorized plain TD3 training.

    Transitions are appended in batches with ``add_batch`` and stored as
    tensors on the training device (CUDA when available, CPU otherwise), so
    ``sample`` gathers batches entirely on-device without a host round trip.
    The stored fields and ``done`` semantics match ``ReplayBuffer``; only the
    storage layout differs.
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim), device=self.device)
        self.action = torch.zeros((max_size, action_dim), device=self.device)
        self.next_state = torch.zeros((max_size, state_dim), device=self.device)
        self.reward = torch.zeros((max_size, 1), device=self.device)
        self.not_done = torch.zeros((max_size, 1), device=self.device)

    def _write(self, store, values):
        """Write a (batch, dim) batch at ``self.ptr``, wrapping the ring."""
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if values.dim() == 1:
            values = values.unsqueeze(1)
        n = values.shape[0]
        end = self.ptr + n
        if end <= self.max_size:
            store[self.ptr:end] = values
        else:
            first = self.max_size - self.ptr
            store[self.ptr:] = values[:first]
            store[: n - first] = values[first:]

    def add_batch(self, state, action, next_state, reward, done):
        """Append a batch of transitions, each field shaped (batch, dim).

        ``done`` is 1.0 for terminations (collision or goal) and 0.0 for
        time-limit truncations, matching ``ReplayBuffer.add``.
        """
        batch = np.asarray(reward).shape[0]
        if batch == 0:
            return
        self._write(self.state, state)
        self._write(self.action, action)
        self._write(self.next_state, next_state)
        self._write(self.reward, reward)
        self._write(self.not_done, 1.0 - np.asarray(done, dtype=np.float32))
        self.ptr = (self.ptr + batch) % self.max_size
        self.size = min(self.size + batch, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind],
        )


class ReplayBufferRCRLTD3Torch(object):
    """Device-resident replay buffer for vectorized RCRL-TD3 training.

    Transitions are appended in batches with ``add_batch`` and stored as
    tensors on the training device (CUDA when available, CPU otherwise), so
    ``sample`` gathers batches entirely on-device without a host round trip.
    The stored fields and ``done`` semantics match ``ReplayBufferRCRLTD3``;
    only the storage layout differs.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        device=None,
        boundary_threshold=None,
        boundary_max_size=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros((max_size, state_dim), device=self.device)
        self.action = torch.zeros((max_size, action_dim), device=self.device)
        self.next_state = torch.zeros((max_size, state_dim), device=self.device)
        self.reward = torch.zeros((max_size, 1), device=self.device)
        self.h = torch.zeros((max_size, 1), device=self.device)
        self.next_h = torch.zeros((max_size, 1), device=self.device)
        self.not_done = torch.zeros((max_size, 1), device=self.device)

        # Optional sub-buffer of near-boundary transitions, used to
        # oversample the safety-critic batches. A transition is
        # near-boundary when h or next_h is above -boundary_threshold, i.e.
        # the clearance is within boundary_threshold of the collision
        # boundary (h = vehicle_radius - min_lidar) or already violating it.
        self.boundary_threshold = boundary_threshold
        self.boundary_ptr = 0
        self.boundary_size = 0
        if boundary_threshold is not None:
            self.boundary_max_size = boundary_max_size or max(max_size // 5, 1)
            self.boundary_state = torch.zeros(
                (self.boundary_max_size, state_dim), device=self.device
            )
            self.boundary_action = torch.zeros(
                (self.boundary_max_size, action_dim), device=self.device
            )
            self.boundary_next_state = torch.zeros(
                (self.boundary_max_size, state_dim), device=self.device
            )
            self.boundary_reward = torch.zeros(
                (self.boundary_max_size, 1), device=self.device
            )
            self.boundary_h = torch.zeros(
                (self.boundary_max_size, 1), device=self.device
            )
            self.boundary_next_h = torch.zeros(
                (self.boundary_max_size, 1), device=self.device
            )
            self.boundary_not_done = torch.zeros(
                (self.boundary_max_size, 1), device=self.device
            )

    def _write(self, store, values, ptr, max_size):
        """Write a (batch, dim) batch at ``ptr``, wrapping at ``max_size``."""
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if values.dim() == 1:
            values = values.unsqueeze(1)
        n = values.shape[0]
        end = ptr + n
        if end <= max_size:
            store[ptr:end] = values
        else:
            first = max_size - ptr
            store[ptr:] = values[:first]
            store[: n - first] = values[first:]

    def add_batch(self, state, action, next_state, reward, h, next_h, done):
        """Append a batch of transitions, each field shaped (batch, dim).

        ``done`` is 1.0 for terminations (collision or goal) and 0.0 for
        time-limit truncations, matching ``ReplayBufferRCRLTD3.add``. When a
        boundary sub-buffer is allocated, near-boundary transitions are also
        mirrored into it.
        """
        batch = np.asarray(reward).shape[0]
        if batch == 0:
            return
        self._write(self.state, state, self.ptr, self.max_size)
        self._write(self.action, action, self.ptr, self.max_size)
        self._write(self.next_state, next_state, self.ptr, self.max_size)
        self._write(self.reward, reward, self.ptr, self.max_size)
        self._write(self.h, h, self.ptr, self.max_size)
        self._write(self.next_h, next_h, self.ptr, self.max_size)
        self._write(
            self.not_done, 1.0 - np.asarray(done, dtype=np.float32),
            self.ptr, self.max_size,
        )
        self.ptr = (self.ptr + batch) % self.max_size
        self.size = min(self.size + batch, self.max_size)

        if self.boundary_threshold is not None:
            self._add_boundary_batch(
                state, action, next_state, reward, h, next_h, done
            )

    def _add_boundary_batch(
        self, state, action, next_state, reward, h, next_h, done
    ):
        h_t = torch.as_tensor(h, dtype=torch.float32, device=self.device)
        next_h_t = torch.as_tensor(next_h, dtype=torch.float32, device=self.device)
        mask = (h_t.squeeze(-1) > -self.boundary_threshold) | (
            next_h_t.squeeze(-1) > -self.boundary_threshold
        )
        kept_idx = mask.nonzero().squeeze(-1)
        kept = int(kept_idx.numel())
        if kept == 0:
            return
        if kept > self.boundary_max_size:
            # Writing more rows than the ring holds leaves only the most
            # recent rows behind, so skip straight to them.
            kept_idx = kept_idx[kept - self.boundary_max_size:]
            kept = int(kept_idx.numel())
        for store, values in (
            (self.boundary_state, state),
            (self.boundary_action, action),
            (self.boundary_next_state, next_state),
            (self.boundary_reward, reward),
            (self.boundary_h, h),
            (self.boundary_next_h, next_h),
            (self.boundary_not_done, 1.0 - np.asarray(done, dtype=np.float32)),
        ):
            rows = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            if rows.dim() == 1:
                rows = rows.unsqueeze(1)
            self._write(
                store, rows[kept_idx], self.boundary_ptr, self.boundary_max_size
            )
        self.boundary_ptr = (self.boundary_ptr + kept) % self.boundary_max_size
        self.boundary_size = min(self.boundary_size + kept, self.boundary_max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.h[ind],
            self.next_h[ind],
            self.not_done[ind],
        )

    def sample_mixed(self, batch_size, mix, warmup=10_000):
        """Draw a safety batch with a ``mix`` fraction from the boundary
        sub-buffer and the remainder drawn uniformly from the main buffer.

        The effective fraction ramps up linearly from zero to ``mix`` as the
        sub-buffer fills past ``warmup`` transitions, so early training is
        not dominated by a handful of near-boundary rows. Falls back to
        plain uniform sampling when no boundary sub-buffer is allocated.
        """
        if (
            self.boundary_threshold is None
            or mix <= 0.0
            or self.boundary_size == 0
        ):
            return self.sample(batch_size)
        effective = mix * min(1.0, self.boundary_size / float(max(warmup, 1)))
        n_boundary = min(int(round(batch_size * effective)), self.boundary_size)
        if n_boundary <= 0:
            return self.sample(batch_size)
        n_main = batch_size - n_boundary
        ind_b = torch.randint(
            0, self.boundary_size, (n_boundary,), device=self.device
        )
        ind_m = torch.randint(0, self.size, (n_main,), device=self.device)

        return tuple(
            torch.cat([boundary[ind_b], main[ind_m]])
            for main, boundary in (
                (self.state, self.boundary_state),
                (self.action, self.boundary_action),
                (self.next_state, self.boundary_next_state),
                (self.reward, self.boundary_reward),
                (self.h, self.boundary_h),
                (self.next_h, self.boundary_next_h),
                (self.not_done, self.boundary_not_done),
            )
        )
