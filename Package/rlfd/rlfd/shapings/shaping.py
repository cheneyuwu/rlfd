import abc

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from rlfd import logger, memory

SHAPINGS = {}


class EnsembleShaping(object):

  def __init__(self, shaping_type, num_ensembles, num_epochs, batch_size, *args,
               **kwargs):
    self.shapings = [
        SHAPINGS[shaping_type](*args, **kwargs) for _ in range(num_ensembles)
    ]

    self.shaping_type = shaping_type
    self.num_epochs = num_epochs
    self.batch_size = batch_size

  def train(self, demo_data):
    dataset = self._construct_dataset(demo_data)
    for i, shaping in enumerate(self.shapings):
      logger.log("Training shaping function #{}...".format(i))

      shaping.before_training_hook(demo_data)

      self.training_step = self.shapings[i].training_step
      with tf.summary.record_if(lambda: self.training_step % 200 == 0):
        for epoch in range(self.num_epochs):
          dataset_iter = dataset.sample(return_iterator=True,
                                        shuffle=True,
                                        include_partial_batch=True)
          dataset_iter(self.batch_size)
          for batch in dataset_iter:
            shaping.train(**batch, name="model_" + str(i))
          # TODO: should be done on validation set
          shaping.evaluate(**batch, name="model_" + str(i))

      shaping.after_training_hook(demo_data)

  @tf.function
  def potential(self, o, g, u):
    potential = tf.reduce_mean([x.potential(o, g, u) for x in self.shapings],
                               axis=0)
    return potential

  def _construct_dataset(self, demo_data):
    buffer_shapes = {k: tuple(v.shape[1:]) for k, v in demo_data.items()}
    dataset = memory.StepBaseReplayBuffer(buffer_shapes, int(1e6))
    dataset.store(demo_data)
    return dataset


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
