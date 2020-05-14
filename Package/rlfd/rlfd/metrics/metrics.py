"""Base class for driver metrics."""
import abc
import numpy as np
import tensorflow as tf


class NumpyDeque(object):
  """Deque implementation using a numpy array as a circular buffer."""

  def __init__(self, maxlen, dtype):
    """Deque using a numpy array as a circular buffer, with FIFO evictions.

    Args:
      maxlen: Maximum length of the deque before beginning to evict the oldest
        entries. If np.inf, deque size is unlimited and the array will grow
        automatically.
      dtype: Data type of deque elements.
    """
    self._start_index = np.int64(0)
    self._len = np.int64(0)
    self._maxlen = np.array(maxlen)
    initial_len = 10 if np.isinf(self._maxlen) else self._maxlen
    self._buffer = np.zeros(shape=(initial_len,), dtype=dtype)

  def clear(self):
    self._start_index = np.int64(0)
    self._len = np.int64(0)

  def add(self, value):
    insert_idx = int((self._start_index + self._len) % self._maxlen)

    # Increase buffer size if necessary.
    if np.isinf(self._maxlen) and insert_idx >= self._buffer.shape[0]:
      self._buffer.resize((self._buffer.shape[0] * 2,))

    self._buffer[insert_idx] = value
    if self._len < self._maxlen:
      self._len += 1
    else:
      self._start_index = np.mod(self._start_index + 1, self._maxlen)

  def extend(self, values):
    for value in values:
      self.add(value)

  def __len__(self):
    return self._len

  def mean(self, dtype=None):
    if self._len == self._buffer.shape[0]:
      return np.mean(self._buffer, dtype=dtype)

    assert self._start_index == 0
    return np.mean(self._buffer[:self._len], dtype=dtype)


class StepMetric(object, metaclass=abc.ABCMeta):

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    return self._name

  def __call__(self, *args, **kwargs):
    """update metric given a step
    """
    return self.call(*args, **kwargs)

  @abc.abstractmethod
  def call(self, *args, **kwargs):
    """Implement step metric update"""

  @abc.abstractmethod
  def result(self):
    """Computes and returns a final value for the metric."""

  @abc.abstractmethod
  def reset(self, *args, **kwargs):
    """Implement step metric update"""

  def summarize(self, step=None, step_metrics=()):
    """Generates summaries against train_step and all step_metrics.

    Args:
      train_step: (Optional) Step counter for training iterations. If None, no
        metric is generated against the global step.
      step_metrics: (Optional) Iterable of step metrics to generate summaries
        against.

    Returns:
      A list of summaries.
    """
    summaries = []
    result = self.result()
    if step is not None:
      tf.summary.scalar(name=self.name, data=result, step=step)
    for step_metric in step_metrics:
      # Skip plotting the metrics against itself.
      if self.name == step_metric.name:
        continue
      step_tag = '{} vs {}'.format(self.name, step_metric.name)
      tf.summary.scalar(name=step_tag,
                        data=result,
                        step=int(step_metric.result()))


class StreamingMetric(StepMetric, metaclass=abc.ABCMeta):
  """Abstract base class for streaming metrics.

  Streaming metrics keep track of the last (upto) K values of the metric in a
  Deque buffer of size K. Calling result() will return the average value of the
  items in the buffer.
  """

  def __init__(self, name='StreamingMetric', buffer_size=10):
    super(StreamingMetric, self).__init__(name)
    self._buffer = NumpyDeque(maxlen=buffer_size, dtype=np.float64)
    self.reset()

  def reset(self):
    self._buffer.clear()

  def add_to_buffer(self, values):
    """Appends new values to the buffer."""
    self._buffer.extend(values)

  def result(self):
    """Returns the value of this metric."""
    if self._buffer:
      return self._buffer.mean(dtype=np.float32)
    return np.array(0.0, dtype=np.float32)


class AverageReturnMetric(StreamingMetric):
  """Computes the average undiscounted reward."""

  def __init__(self, name='AverageReturn', buffer_size=10):
    """Creates an AverageReturnMetric."""
    super(AverageReturnMetric, self).__init__(name, buffer_size=buffer_size)
    self._episode_return = 0.0
    self.reset()

  def call(self, **transition):
    r = transition["r"]
    self._episode_return += r
    done = transition["reset"]
    if done:
      self.add_to_buffer([self._episode_return])
      self._episode_return = 0.0


class AverageEpisodeLengthMetric(StreamingMetric):
  """Computes the average episode length."""

  def __init__(self, name='AverageEpisodeLength', buffer_size=10):
    """Creates an AverageEpisodeLengthMetric."""
    super(AverageEpisodeLengthMetric, self).__init__(name,
                                                     buffer_size=buffer_size)
    self._episode_length = 0
    self.reset()

  def call(self, **transition):
    self._episode_length += 1
    done = transition["reset"]
    if done:
      self.add_to_buffer([self._episode_length])
      self._episode_length = 0


class EnvironmentSteps(StepMetric):
  """Counts the number of steps taken in the environment."""

  def __init__(self, name='EnvironmentSteps'):
    super(EnvironmentSteps, self).__init__(name)
    self.environment_steps = 0
    self.reset()

  def reset(self):
    self.environment_steps = 0

  def result(self):
    return self.environment_steps

  def call(self, **transition):
    self.environment_steps += 1


class NumberOfEpisodes(StepMetric):
  """Counts the number of episodes in the environment."""

  def __init__(self, name='NumberOfEpisodes'):
    super(NumberOfEpisodes, self).__init__(name)
    self.num_episodes = 0
    self.reset()

  def reset(self):
    self.num_episodes = 0

  def result(self):
    return self.num_episodes

  def call(self, **transition):
    done = transition["reset"]
    if done:
      self.num_episodes += 1