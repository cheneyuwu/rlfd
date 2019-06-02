import threading

import numpy as np

from yw.tool import logger


class ReplayBuffer:

    def __init__(self, buffer_shapes, size_in_transitions, T):
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

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape], dtype=np.float32) for key, shape in buffer_shapes.items()}

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
        if "ag" in buffers.keys():
            buffers["ag_2"] = buffers["ag"][:, 1:, :]
        if "g" in buffers.keys():
            buffers["g_2"] = buffers["g"][:, :, :]

        transitions = self._sample_transitions(buffers, batch_size)

        assert all([key in transitions for key in list(self.buffers.keys())]), "key missing from transitions"

        return transitions

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

    def _sample_transitions(self, buffers, batch_size):
        return NotImplementedError

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


class UniformReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_shapes, size_in_transitions, T):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        super().__init__(buffer_shapes, size_in_transitions, T)

    def _sample_transitions(self, buffers, batch_size):
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

class HERReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_shapes, size_in_transitions, T, k, reward_fun):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        """Creates a sample function that can be used for HER experience replay.

        Args:
            strategy (str) - set to "future" to use the HER replay strategy; if set to 'none', regular DDPG experience replay is used
            k        (int) - the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times as many HER replays as regular replays are used)
            reward_fun (function): function to re-compute the reward with substituted goals
        """
        super().__init__(buffer_shapes, size_in_transitions, T)
        self.future_p = 1 - (1.0 / (1 + k))
        self.reward_fun = reward_fun

    def _sample_transitions(self, buffers, batch_size):
        """
        buffers is {key: array(buffer_size x T x dim_key)}
        """

        # Select which episodes and time steps to use.

        episode_idxs = np.random.randint(0, buffers["u"].shape[0], batch_size)
        t_samples = np.random.randint(self.T, size=batch_size)
        transitions = {key: buffers[key][episode_idxs, t_samples].copy() for key in buffers.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (self.T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = buffers["ag"][episode_idxs[her_indexes], future_t]
        transitions["g"][her_indexes] = future_ag
        transitions["g_2"][her_indexes] = future_ag

        if "q" in transitions.keys():
            transitions["q"][her_indexes] = -100 * np.ones(transitions["q"][her_indexes].shape)

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith("info_"):
                info[key.replace("info_", "")] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ["ag_2", "g_2"]}
        reward_params["info"] = info
        transitions["r"] = self.reward_fun(**reward_params).reshape(-1, 1)  # reshape to be consistent with default reward

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert transitions["u"].shape[0] == batch_size

        return transitions

class NStepReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_shapes, size_in_transitions, T, gamma, n):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        super().__init__(buffer_shapes, size_in_transitions, T)
        self.gamma = gamma
        self.n = n

    """Creates a sample function that can be used for n step return

    Args:
        gamma (str) - discount rate
        n        (int) - the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times as many HER replays as regular replays are used)
    """
    def _sample_transitions(self, buffers, batch_size):

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, buffers["u"].shape[0], batch_size)
        t_samples = np.random.randint(self.T, size=batch_size)

        # Get the transitions
        transitions = {}
        for k in ["o", "u", "g", "ag", "q", "mask", "info_is_success"]:
            transitions[k] = buffers[k][episode_idxs, t_samples].copy()
        # calculate n step return
        cum_reward = np.zeros_like(buffers["r"][episode_idxs, t_samples])
        cum_discount = np.zeros_like(buffers["n"][episode_idxs, t_samples])
        assert self.n>= 1
        for step in range(self.n):
            cum_reward += np.where(((t_samples + step) < self.T).reshape(cum_reward.shape), buffers["r"][episode_idxs, np.minimum(self.T-1, t_samples + step)] * np.power(self.gamma, step), 0)
            cum_discount += np.where(((t_samples + step) < self.T).reshape(cum_discount.shape), 1, 0)
        transitions["r"] = cum_reward
        transitions["n"] = cum_discount
        # change the state it goes to
        n_step_t_samples = np.minimum(t_samples + self.n - 1, self.T-1)
        for k in ["o_2", "ag_2", "g_2"]:
            transitions[k] = buffers[k][episode_idxs, n_step_t_samples].copy()
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        assert transitions.keys() == buffers.keys()
        assert transitions["u"].shape[0] == batch_size

        return transitions
