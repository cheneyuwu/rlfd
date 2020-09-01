import abc
import numpy as np


def iterbatches(arrays,
                *,
                num_batches=None,
                batch_size=None,
                shuffle=True,
                include_final_partial_batch=False):
  assert (num_batches is None) != (
      batch_size is None), "Provide num_batches or batch_size, but not both"
  arrays = tuple(map(np.asarray, arrays))
  n = arrays[0].shape[0]
  assert all(a.shape[0] == n for a in arrays[1:])
  inds = np.arange(n)
  if shuffle:
    np.random.shuffle(inds)
  sections = np.arange(0, n,
                       batch_size)[1:] if num_batches is None else num_batches
  for batch_inds in np.array_split(inds, sections):
    if include_final_partial_batch or len(batch_inds) == batch_size:
      yield tuple(a[batch_inds] for a in arrays)


class SampleIterator(object):

  def __init__(self, sample_fn, batch_size):
    self._sample_fn = sample_fn
    self._batch_size = batch_size

  def __call__(self, batch_size):
    self._batch_size = batch_size
    return self

  def __iter__(self):
    return self

  def __next__(self):
    batch = self._sample_fn(self._batch_size)
    if batch is None:
      raise StopIteration
    return batch


class ReplayBuffer(object, metaclass=abc.ABCMeta):

  def __init__(self, buffer_shapes, size):
    """ Create a replay buffer.
    Args:
        size (int) - the size of the buffer, measured in transitions
    """
    # memory management
    self._size = size
    self._current_size = 0

    # buffer
    self.buffers = {
        key: np.empty([self._size, *shape], dtype=np.float32)
        for key, shape in buffer_shapes.items()
    }

  def load_from_file(self, data_file):
    episode_batch = dict(np.load(data_file))
    self.store(episode_batch)
    return episode_batch

  def dump_to_file(self, path):
    if self._current_size == 0:
      return
    buffers = {k: v[:self._current_size] for k, v in self.buffers.items()}
    np.savez_compressed(path, **buffers)  # save the file

  def sample(self,
             batch_size=None,
             return_iterator=False,
             shuffle=False,
             include_partial_batch=False,
             repeat=False):
    """ Returns a dict {key: array(batch_size x shapes[key])}
    If batch_size is -1, this function should return the entire buffer (i.e. all stored transitions)
    """
    assert self._current_size > 0, "Replay buffer is empty."
    if return_iterator:
      return self._sample_iterator(batch_size, shuffle, include_partial_batch,
                                   repeat)
    else:
      assert batch_size != None, "Must provide batch size to sample randomly."
      return self._sample_random(batch_size)

  def store(self, data):
    """ Store data into the replay buffer.
        data.shape[0] should be the size of increment
    """
    batch_sizes = [len(v) for v in data.values()]
    assert np.all(np.array(batch_sizes) == batch_sizes[0])
    batch_size = batch_sizes[0]

    self._store(data)

    # memory management
    self._current_size = min(self._size, self._current_size + batch_size)

  def clear_buffer(self):
    self._clear_buffer()
    self._current_size = 0

  @property
  def full(self):
    return self._current_size == self._size

  @property
  def capacity(self):
    return self._size

  @property
  def current_size(self):
    return self._current_size

  @property
  @abc.abstractmethod
  def stored_steps(self):
    """current number of environment steps stored in the replay buffer"""

  @property
  @abc.abstractmethod
  def stored_episodes(self):
    """current number of environment episodes stored in the replay buffer"""

  @abc.abstractmethod
  def _store(self, data):
    """store batch of data into the replay buffer"""

  @abc.abstractmethod
  def _sample_random(self, batch_size):
    """Sample a batch of sizee batch_size randomly from the replay buffer"""

  @abc.abstractmethod
  def _sample_iterator(self, batch_size, shuffle, include_partial_batch,
                       repeat):
    """return a iterator from sampler"""

  @abc.abstractmethod
  def _clear_buffer(self):
    """Overwrite this for further cleanup"""


class StepBaseReplayBuffer(ReplayBuffer):

  def __init__(self, buffer_shapes, size_in_transitions):
    """ Creates a replay buffer.

        Args:
            buffer_shapes       (dict of float) - the shape for all buffers that are used in the replay buffer
            size_in_transitions (int)           - the size of the buffer, measured in transitions
        """
    super().__init__(size=size_in_transitions, buffer_shapes=buffer_shapes)

    # contains {key: array(transitions x dim_key)}
    self._pointer = 0

  @staticmethod
  def construct_from_file(data_file):
    experiences = dict(np.load(data_file))
    buffer_shapes = {k: v.shape[1:] for k, v in experiences.items()}
    buffer_size = None
    for v in experiences.values():
      if buffer_size != None:
        assert buffer_size == v.shape[0], "Inconsistent batch size."
      else:
        buffer_size = v.shape[0]
    replay_buffer = StepBaseReplayBuffer(buffer_shapes, buffer_size)
    replay_buffer.store(experiences)
    return replay_buffer

  @property
  def stored_steps(self):
    """current number of environment steps stored in the replay buffer"""
    return self._current_size

  @property
  def stored_episodes(self):
    """current number of environment episodes stored in the replay buffer"""
    raise ValueError("Number of episodes unknown in step based replay buffer.")

  def _sample_random(self, batch_size):
    """Sample a batch of sizee batch_size randomly from the replay buffer"""
    inds = np.random.randint(0, self.stored_steps, batch_size)
    transitions = {
        key: self.buffers[key][inds].copy() for key in self.buffers.keys()
    }

    return transitions

  def _sample_iterator(self, batch_size, shuffle, include_partial_batch,
                       repeat):
    """return a iterator from sampler"""
    inds = np.arange(self.stored_steps)
    if shuffle:
      np.random.shuffle(inds)

    transitions = {
        key: self.buffers[key][inds].copy() for key in self.buffers.keys()
    }

    if batch_size == None:
      batch_size = self.stored_steps

    self._num_sampled = 0

    def _sample(batch_size):
      inds = None
      if repeat:
        inds = []
        while len(inds) < batch_size:
          remaining_size = batch_size - len(inds)
          last_idx = min(
              [self._num_sampled + remaining_size, self.stored_steps])
          inds += list(range(self._num_sampled, last_idx))
          self._num_sampled = last_idx % self.stored_steps
      else:
        if self._num_sampled >= self.stored_steps:
          return None
        if (self._num_sampled + batch_size > self.stored_steps):
          if not include_partial_batch:
            return None
          else:
            batch_size = self.stored_steps - self._num_sampled
        inds = list(range(self._num_sampled, self._num_sampled + batch_size))
        self._num_sampled += batch_size

      batch = {k: v[inds] for k, v in transitions.items()}
      return batch

    return SampleIterator(_sample, batch_size)

  def _store(self, data):

    batch_sizes = [len(v) for v in data.values()]
    assert np.all(np.array(batch_sizes) == batch_sizes[0])
    batch_size = batch_sizes[0]
    idxs = self._get_storage_idx(batch_size)

    # load inputs into buffers
    for key in self.buffers.keys():
      self.buffers[key][idxs] = data[key]

  def _clear_buffer(self):
    self._pointer = 0

  def _get_storage_idx(self, inc):
    assert inc <= self._size, "batch committed to replay is too large!"
    assert inc > 0, "invalid increment"
    # go consecutively until you hit the end, and restart from the beginning.
    if self._pointer + inc <= self._size:
      idx = np.arange(self._pointer, self._pointer + inc)
      self._pointer += inc
    else:
      overflow = inc - (self._size - self._pointer)
      idx_a = np.arange(self._pointer, self._size)
      idx_b = np.arange(0, overflow)
      idx = np.concatenate([idx_a, idx_b])
      self._pointer = overflow

    if inc == 1:
      idx = idx[0]
    return idx


class EpisodeBaseReplayBuffer(ReplayBuffer):

  def __init__(self, buffer_shapes, size_in_transitions, T):
    """ Creates a replay buffer.

        Args:
            buffer_shapes       (dict of int) - the shape for all buffers that are used in the replay buffer
            size_in_transitions (int)         - the size of the buffer, measured in transitions
            T                   (int)         - the time horizon for episodes
        """

    super().__init__(size=size_in_transitions // T, buffer_shapes=buffer_shapes)
    # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
    self.T = T

  @staticmethod
  def construct_from_file(data_file):
    experiences = dict(np.load(data_file))
    buffer_shapes = {k: v.shape[1:] for k, v in experiences.items()}
    buffer_size = None
    T = None
    for v in experiences.values():
      if buffer_size != None:
        assert buffer_size == v.shape[0], "Inconsistent batch size."
        assert T == v.shape[1], "Inconsistent episode length."
      else:
        buffer_size = v.shape[0]
        T = v.shape[1]
    replay_buffer = EpisodeBaseReplayBuffer(buffer_shapes, buffer_size * T, T)
    replay_buffer.store(experiences)
    return replay_buffer

  @property
  def stored_steps(self):
    """current number of environment steps stored in the replay buffer"""
    return self._current_size * self.T

  @property
  def stored_episodes(self):
    """current number of environment episodes stored in the replay buffer"""
    return self._current_size

  def _sample_random(self, batch_size):
    """ Returns a dict {key: array(batch_size x shapes[key])}
    """

    buffers = {}
    for key in self.buffers.keys():
      buffers[key] = self.buffers[key][:self._current_size]

    episode_idxs = np.random.randint(self._current_size, size=batch_size)
    step_idxs = np.random.randint(self.T, size=batch_size)

    transitions = {
        key: buffers[key][episode_idxs, step_idxs].copy()
        for key in buffers.keys()
    }

    return transitions

  def _sample_iterator(self, batch_size, shuffle, include_partial_batch,
                       repeat):

    buffers = {}
    for key in self.buffers.keys():
      buffers[key] = self.buffers[key][:self._current_size]

    episode_idxs = np.repeat(np.arange(self._current_size), self.T)
    step_idxs = np.tile(np.arange(self.T), self._current_size)

    transitions = {
        key: buffers[key][episode_idxs, step_idxs].copy()
        for key in buffers.keys()
    }

    inds = np.arange(self.stored_steps)
    if shuffle:
      np.random.shuffle(inds)

    transitions = {
        key: transitions[key][inds].copy() for key in transitions.keys()
    }

    if batch_size == None:
      batch_size = self.stored_steps

    self._num_sampled = 0

    def _sample(batch_size):
      inds = None
      if repeat:
        inds = []
        while len(inds) < batch_size:
          remaining_size = batch_size - len(inds)
          last_idx = min(
              [self._num_sampled + remaining_size, self.stored_steps])
          inds += list(range(self._num_sampled, last_idx))
          self._num_sampled = last_idx % self.stored_steps
      else:
        if self._num_sampled >= self.stored_steps:
          return None
        if (self._num_sampled + batch_size > self.stored_steps):
          if not include_partial_batch:
            return None
          else:
            batch_size = self.stored_steps - self._num_sampled
        inds = list(range(self._num_sampled, self._num_sampled + batch_size))
        self._num_sampled += batch_size

      batch = {k: v[inds] for k, v in transitions.items()}
      return batch

    return SampleIterator(_sample, batch_size)

  def _store(self, data):

    batch_sizes = [len(v) for v in data.values()]
    assert np.all(np.array(batch_sizes) == batch_sizes[0])
    batch_size = batch_sizes[0]
    idxs = self._get_storage_idx(batch_size)

    # load inputs into buffers
    for key in self.buffers.keys():
      self.buffers[key][idxs] = data[key]

  def _clear_buffer(self):
    pass

  def _get_storage_idx(self, inc):
    assert inc <= self._size, "batch committed to replay is too large!"
    assert inc > 0, "invalid increment"
    # go consecutively until you hit the end, and then go randomly.
    if self._current_size + inc <= self._size:
      idx = np.arange(self._current_size, self._current_size + inc)
    elif self._current_size < self._size:
      overflow = inc - (self._size - self._current_size)
      idx_a = np.arange(self._current_size, self._size)
      idx_b = np.random.randint(0, self._current_size, overflow)
      idx = np.concatenate([idx_a, idx_b])
    else:
      idx = np.random.randint(0, self._size, inc)

    if inc == 1:
      idx = idx[0]
    return idx


if __name__ == "__main__":
  # EpisodeBaseReplayBuffer Usage
  o = np.linspace(0.0, 15.0, 16).reshape((2, 4, 2))  # Batch x Time x Dim
  r = np.linspace(0.0, 5.0, 6).reshape((2, 3, 1))  # Batch x Time x Dim
  replay_buffer = EpisodeBaseReplayBuffer({
      "o": o.shape[1:],
      "r": r.shape[1:]
  }, 6, 3)
  #
  print(replay_buffer.capacity)
  print(replay_buffer.current_size)
  print(replay_buffer.full)
  #
  replay_buffer.store({"o": o, "r": r})
  #
  print(replay_buffer.capacity)
  print(replay_buffer.current_size)
  print(replay_buffer.full)
  #
  batch = replay_buffer.sample(10)
  print(batch["r"].shape)
  print(batch["r"])
  #
  iterator = replay_buffer.sample(1, return_iterator=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  iterator = replay_buffer.sample(1, return_iterator=True, shuffle=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  iterator = replay_buffer.sample(4,
                                  return_iterator=True,
                                  shuffle=True,
                                  include_partial_batch=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  iterator = replay_buffer.sample(return_iterator=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  iterator = replay_buffer.sample(2, return_iterator=True)
  batch = next(iterator)
  print(batch["r"].shape)
  print(batch["r"])
  iterator(4)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])

  # StepBaseReplayBuffer Usage
  o = np.linspace(0.0, 17.0, 8).reshape((4, 2))  # Batch x Time x Dim
  r = np.linspace(0.0, 3.0, 4).reshape((4, 1))  # Batch x Time x Dim
  replay_buffer = StepBaseReplayBuffer({"o": o.shape[1:], "r": r.shape[1:]}, 6)
  #
  print(replay_buffer.capacity)
  print(replay_buffer.current_size)
  print(replay_buffer.full)
  #
  replay_buffer.store({"o": o, "r": r})
  replay_buffer.store({"o": o, "r": r})
  #
  print(replay_buffer.capacity)
  print(replay_buffer.current_size)
  print(replay_buffer.full)
  #
  print("##############")
  batch = replay_buffer.sample(10)
  print(batch["r"].shape)
  print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(1, return_iterator=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(1, return_iterator=True, shuffle=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(4,
                                  return_iterator=True,
                                  shuffle=True,
                                  include_partial_batch=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(return_iterator=True)
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(return_iterator=True)
  iterator(1)  # set batch size to be 1
  for batch in iterator:
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  batch = replay_buffer.sample(10, repeat=True)
  print(batch["r"].shape)
  print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(1, return_iterator=True, repeat=True)
  for _ in range(10):
    batch = next(iterator)
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(1,
                                  return_iterator=True,
                                  shuffle=True,
                                  repeat=True)
  for _ in range(10):
    batch = next(iterator)
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(4,
                                  return_iterator=True,
                                  shuffle=True,
                                  include_partial_batch=True,
                                  repeat=True)
  for _ in range(10):
    batch = next(iterator)
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(return_iterator=True, repeat=True)
  for _ in range(10):
    batch = next(iterator)
    print(batch["r"].shape)
    print(batch["r"])
  #
  print("##############")
  iterator = replay_buffer.sample(return_iterator=True, repeat=True)
  iterator(1)  # set batch size to be 1
  for _ in range(10):
    batch = next(iterator)
    print(batch["r"].shape)
    print(batch["r"])
