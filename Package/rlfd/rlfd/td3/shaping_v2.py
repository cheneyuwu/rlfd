import abc

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rlfd.td3.gan_network import Discriminator, Generator
from rlfd.td3.normalizer import Normalizer
from rlfd.td3.normalizing_flows import create_maf

tfd = tfp.distributions


class EnsembleShaping(object):

  def __init__(self, shaping_cls, num_ensembles, *args, **kwargs):
    self.shapings = [shaping_cls(*args, **kwargs) for _ in range(num_ensembles)]

  @property
  def training_step(self):
    return self.shapings[0].training_step

  # @tf.function
  # def train_graph(self, o, g, u):
  #   random_idx = tf.random.uniform(
  #       shape=[len(self.shapings), o.shape[0]],
  #       minval=0,
  #       maxval=o.shape[0],
  #       dtype=tf.dtypes.int32,
  #   )
  #   o = tf.gather(o, random_idx)
  #   g = tf.gather(u, random_idx)
  #   u = tf.gather(u, random_idx)
  #   for i, shaping in enumerate(self.shapings):
  #     loss = shaping.train_graph(o[i], g[i], u[i])

  def train(self, o, g, u, name=""):

    # o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    # g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    # u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    # self.train_graph(o_tf, g_tf, u_tf)

    for i, shaping in enumerate(self.shapings):
      idxs = np.random.randint(o.shape[0], size=o.shape[0])
      shaping.train(o[idxs], g[idxs], u[idxs], name="model_" + str(i))

  def evaluate(self, o, g, u, name=""):
    for i, shaping in enumerate(self.shapings):
      shaping.evaluate(o, g, u, name="model_" + str(i))

  @tf.function
  def potential(self, o, g, u):
    potential = tf.reduce_mean([x.potential(o, g, u) for x in self.shapings],
                               axis=0)
    return potential

  def training_before_hook(self, *args, **kwargs):
    for shaping in self.shapings:
      shaping.training_before_hook(*args, **kwargs)

  def training_after_hook(self, *args, **kwargs):
    for shaping in self.shapings:
      shaping.training_after_hook(*args, **kwargs)
