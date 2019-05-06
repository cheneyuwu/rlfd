import threading

import numpy as np

from yw.tool import logger


class ReplayBuffer:
    @staticmethod
    def default_sampler(episode_batch, batch_size):
        """Sample transitions of size batch_size randomly from episode_batch.

        Args:
            episode_batch - {key: array(buffer_size x T x dim_key)}
            batch_size    - batch size in transitions

        Return:
            transitions
        """

        T = episode_batch["u"].shape[1]
        total_batch_size = episode_batch["u"].shape[0]

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, total_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert transitions["u"].shape[0] == batch_size

        return transitions

    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions=None):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = ReplayBuffer.default_sampler if sample_transitions is None else sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape]) for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][: self.current_size]

        buffers["o_2"] = buffers["o"][:, 1:, :]
        buffers["ag_2"] = buffers["ag"][:, 1:, :]  # change from T+1 to T so that you can have a real transition

        transitions = self.sample_transitions(buffers, batch_size)

        for key in ["r", "o_2", "ag_2"] + list(self.buffers.keys()):
            assert key in transitions, "key %s missing from transitions" % key
        # Note: each transition is {'o' 'u' 'g' 'ag' 'ag_2' 'o_2'}

        return transitions

    def get(self, num_transition=None):
        """get all transitions, this could be useful for pipelining
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][: self.current_size]

        buffers["o_2"] = buffers["o"][:, 1:, :]
        buffers["ag_2"] = buffers["ag"][:, 1:, :]  # change from T+1 to T so that you can have a real transition
        buffers["o"] = buffers["o"][:, :-1, :]
        buffers["ag"] = buffers["ag"][:, :-1, :]
        buffers = {key: buffers[key].reshape((-1,) + buffers[key].shape[2:]) for key in buffers.keys()}
        if num_transition:
            assert num_transition <= self.current_size * self.T, "No enough demonstration data!"
        return {key: buffers[key][:num_transition] for key in buffers.keys()}

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1  # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
        return idx
