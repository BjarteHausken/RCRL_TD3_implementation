import numpy as np
import torch

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
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
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
        self.h = torch.zeros((max_size, 1), device=self.device)
        self.next_h = torch.zeros((max_size, 1), device=self.device)
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

    def add_batch(self, state, action, next_state, reward, h, next_h, done):
        """Append a batch of transitions, each field shaped (batch, dim).

        ``done`` is 1.0 for terminations (collision or goal) and 0.0 for
        time-limit truncations, matching ``ReplayBufferRCRLTD3.add``.
        """
        batch = np.asarray(reward).shape[0]
        if batch == 0:
            return
        self._write(self.state, state)
        self._write(self.action, action)
        self._write(self.next_state, next_state)
        self._write(self.reward, reward)
        self._write(self.h, h)
        self._write(self.next_h, next_h)
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
            self.h[ind],
            self.next_h[ind],
            self.not_done[ind],
        )
