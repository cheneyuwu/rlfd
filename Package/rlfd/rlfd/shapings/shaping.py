import abc
import os
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from rlfd import logger, memory

SHAPINGS = {}


class EnsembleShaping(object):

  def __init__(self, shaping_type, num_ensembles, num_epochs, batch_size, fix_T,
               *args, **kwargs):
    self.shapings = [
        SHAPINGS[shaping_type](*args, **kwargs) for _ in range(num_ensembles)
    ]
    self._fix_T = fix_T
    self.shaping_type = shaping_type
    self.num_epochs = num_epochs
    self.batch_size = batch_size

  def before_training_hook(self, data_dir, env):
    self._data_dir = data_dir
    self._env = env
    # D4RL
    experiences = env.get_dataset()
    if experiences:  # T not fixed by default
      buffer_shapes = {k: v.shape[1:] for k, v in experiences.items()}
      buffer_size = experiences["o"].shape[0]
      self._dataset = memory.StepBaseReplayBuffer(buffer_shapes, buffer_size)
      self._dataset.store(experiences)
    else:
      # Ours
      demo_file = osp.join(data_dir, "demo_data.npz")
      assert osp.isfile(demo_file), "Demostrations not available."
      if self._fix_T:
        self._dataset = memory.EpisodeBaseReplayBuffer.construct_from_file(
            data_file=demo_file)
      else:
        self._dataset = memory.StepBaseReplayBuffer.construct_from_file(
            data_file=demo_file)

  def train(self):
    for i, shaping in enumerate(self.shapings):
      dataset_iter = self._dataset.sample(return_iterator=True,
                                          shuffle=False,
                                          include_partial_batch=True)
      shaping.before_training_hook(data_dir=self._data_dir,
                                   batch=next(dataset_iter))

      self.training_step = self.shapings[i].training_step
      with tf.summary.record_if(lambda: self.training_step % 200 == 0):
        for epoch in range(self.num_epochs):
          dataset_iter = self._dataset.sample(return_iterator=True,
                                              shuffle=True,
                                              include_partial_batch=True)
          dataset_iter(self.batch_size)
          for batch in dataset_iter:
            shaping.train(**batch, name="model_" + str(i))
          # TODO: should be done on validation set
          shaping.evaluate(**batch, name="model_" + str(i))

      shaping.after_training_hook()

  def after_training_hook(self, *args, **kwargs):
    pass

  @tf.function
  def potential(self, o, g, u):
    potential = tf.reduce_mean([x.potential(o, g, u) for x in self.shapings],
                               axis=0)
    return potential


class Shaping(object, metaclass=abc.ABCMeta):

  @classmethod
  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    SHAPINGS[cls.__name__] = cls

  def __init__(self):
    """Implement the state, action based potential"""
    self.training_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  def train(self, *args, **kwargs):
    """train the shaping potential"""
    result = self._train(*args, **kwargs)
    self.training_step.assign_add(1)
    return result

  def evaluate(self, *args, **kwargs):
    """evaluate the shaping potential"""
    return self._evaluate(*args, **kwargs)

  def before_training_hook(self, *args, **kwargs):
    pass

  def after_training_hook(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def potential(self, o, g, u):
    """return the shaping potential, has to be a tf.function"""

  def _train(self, *args, **kwargs):
    """train the shaping potential (implementation)"""

  def _evaluate(self, *args, **kwargs):
    """evaluate the shaping potential (implementation)"""
