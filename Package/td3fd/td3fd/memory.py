"""Partially adopted from OpenAI baseline code base
"""
import numpy as np


class ReplayBufferBase:
    def __init__(self, size):
        """ Creates a replay buffer.

        Args:
            size (int) - the size of the buffer, measured in transitions
        """
        # memory management
        self.size = size
        self.current_size = 0

    def sample_all(self):
        """ Returns all the transitions currently stored in the replay buffer.
        """
        raise NotImplementedError

    def sample(self, batch_size):
        """ Returns a dict {key: array(batch_size x shapes[key])}
        """
        raise NotImplementedError

    def load_from_file(self, data_file, num_demo=None):
        raise NotImplementedError

    def dump_to_file(self, path):
        if self.current_size == 0:
            return
        buffers = {k: v[: self.current_size] for k, v in self.buffers.items()}
        np.savez_compressed(path, **buffers)  # save the file

    def store_episode(self, episode_batch):
        """ API for storing episodes. Including memory management.
            episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        idxs = self._get_storage_idx(batch_size)

        # load inputs into buffers
        for key in self.buffers.keys():
            self.buffers[key][idxs] = episode_batch[key]

        # memory management
        self.current_size = min(self.size, self.current_size + batch_size)

    @property
    def full(self):
        return self.current_size == self.size

    def get_current_size(self):
        return self.current_size

    def clear_buffer(self):
        self.current_size = 0
        self._clear_buffer()

    def _clear_buffer(self):
        """Overwrite this for further cleanup"""
        pass


class RingReplayBuffer(ReplayBufferBase):
    def __init__(self, buffer_shapes, size_in_transitions):
        """ Creates a replay buffer.

        Args:
            buffer_shapes       (dict of float) - the shape for all buffers that are used in the replay buffer
            size_in_transitions (int)           - the size of the buffer, measured in transitions
        """
        super().__init__(size=size_in_transitions)

        # contains {key: array(transitions x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape], dtype=np.float32) for key, shape in buffer_shapes.items()}
        self.pointer = 0

    def load_from_file(self, data_file, num_demo=None):
        episode_batch = {**np.load(data_file)}
        assert "done" in episode_batch.keys()
        dones = np.nonzero(episode_batch["done"])[0]
        assert num_demo is None or num_demo <= len(dones)
        last_idx = dones[num_demo - 1] if num_demo is not None else dones[-1]
        for key in episode_batch.keys():
            assert len(episode_batch[key].shape) == 2
            episode_batch[key] = episode_batch[key][: last_idx + 1]
        self.store_episode(episode_batch)
        return episode_batch

    def sample_all(self):
        """ Returns all the transitions currently stored in the replay buffer.
        """
        transitions = {}  # make a copy of the stored data in case it gets changed
        assert self.current_size > 0
        for key in self.buffers.keys():
            transitions[key] = self.buffers[key][: self.current_size].copy()

        assert all([transitions[k].shape[0:1] == transitions["u"].shape[0:1] for k in transitions.keys()])

        return transitions

    def sample(self, batch_size):
        """ Returns a dict {key: array(batch_size x shapes[key])}
        """
        # select which episodes and time steps to use.
        idxs = np.random.randint(0, self.current_size, batch_size)
        transitions = {key: self.buffers[key][idxs].copy() for key in self.buffers.keys()}
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}
        assert all([transitions[k].shape[0] == batch_size for k in transitions.keys()])

        return transitions

    def _get_storage_idx(self, inc):
        assert inc <= self.size, "batch committed to replay is too large!"
        assert inc > 0, "invalid increment"
        # go consecutively until you hit the end, and restart from the beginning.
        if self.pointer + inc <= self.size:
            idx = np.arange(self.pointer, self.pointer + inc)
            self.pointer += inc
        else:
            overflow = inc - (self.size - self.pointer)
            idx_a = np.arange(self.pointer, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.pointer = overflow

        if inc == 1:
            idx = idx[0]
        return idx

    def _clear_buffer(self):
        self.pointer = 0


class UniformReplayBuffer(ReplayBufferBase):
    def __init__(self, buffer_shapes, size_in_transitions, T):
        """ Creates a replay buffer.

        Args:
            buffer_shapes       (dict of int) - the shape for all buffers that are used in the replay buffer
            size_in_transitions (int)         - the size of the buffer, measured in transitions
            T                   (int)         - the time horizon for episodes
        """

        super().__init__(size=size_in_transitions // T)

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape], dtype=np.float32) for key, shape in buffer_shapes.items()}
        self.T = T

    def load_from_file(self, data_file, num_demo=None):
        episode_batch = {**np.load(data_file)}
        for key in episode_batch.keys():
            assert len(episode_batch[key].shape) == 3  # (eps x T x dim)
            assert num_demo is None or num_demo <= episode_batch[key].shape[0], "No enough demonstration data!"
            episode_batch[key] = episode_batch[key][:num_demo]
        self.store_episode(episode_batch)
        return episode_batch

    def sample_all(self):
        """ Returns all the transitions currently stored in the replay buffer.
        """
        buffers = {}
        assert self.current_size > 0
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][: self.current_size].copy()

        # Split obs and goal (i.e. for (s, a, s'), put s and s' to two entries)
        buffers["o_2"] = buffers["o"][:, 1:, :]
        buffers["o"] = buffers["o"][:, :-1, :]
        if "ag" in buffers.keys():
            buffers["ag_2"] = buffers["ag"][:, 1:, :]
            buffers["ag"] = buffers["ag"][:, :-1, :]
        if "g" in buffers.keys():
            buffers["g_2"] = buffers["g"][:, :, :]

        # Merge the first and second dimension (dimensions are : episode x T or T+1 x dim)
        transitions = {k: buffers[k].reshape([-1] + list(buffers[k].shape[2:])) for k in buffers.keys()}

        assert all([transitions[k].shape[0:1] == transitions["u"].shape[0:1] for k in transitions.keys()])

        return transitions

    def sample(self, batch_size):
        """ Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        assert self.current_size > 0
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][: self.current_size]

        buffers["o_2"] = buffers["o"][:, 1:, :]
        if "ag" in buffers.keys():
            buffers["ag_2"] = buffers["ag"][:, 1:, :]
        if "g" in buffers.keys():
            buffers["g_2"] = buffers["g"][:, :, :]

        transitions = self.sample_transitions(buffers, batch_size)

        assert all([key in transitions for key in list(self.buffers.keys())]), "key missing from transitions"

        return transitions

    def sample_transitions(self, buffers, batch_size):
        """Sample transitions of size batch_size randomly from episode_batch.

        Args:
            episode_batch - {key: array(buffer_size x T x dim_key)}
            batch_size    - batch size in transitions

        Return:
            transitions
        """
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, buffers["u"].shape[0], batch_size)
        t_samples = np.random.randint(self.T, size=batch_size)
        transitions = {key: buffers[key][episode_idxs, t_samples].copy() for key in buffers.keys()}

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert transitions["u"].shape[0] == batch_size

        return transitions

    def get_current_size_episode(self):
        return self.current_size

    def get_current_size_transiton(self):
        return self.current_size * self.T

    def _get_storage_idx(self, inc):
        assert inc <= self.size, "batch committed to replay is too large!"
        assert inc > 0, "invalid increment"
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

        if inc == 1:
            idx = idx[0]
        return idx
