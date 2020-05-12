"""Adopted from Eric Jang's normalizing flow tutorial: https://blog.evjang.com/2018/01/nf1.html
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class ClippedAutoregressiveNetwork(tf.Module):

  def __init__(self, layer_sizes):
    self.autoregressive_network = tfb.AutoregressiveNetwork(
        params=2,
        hidden_units=layer_sizes,
        kernel_initializer="glorot_normal",
        bias_initializer="zeros",
        activation="relu",
        dtype=tf.float64,
    )

  def __call__(self, x):
    x = self.autoregressive_network(x)
    shift, log_scale = tf.unstack(x, num=2, axis=-1)

    clip_log_scale = tf.clip_by_value(log_scale, -5.0, 3.0)
    log_scale = log_scale + tf.stop_gradient(clip_log_scale - log_scale)

    return shift, log_scale


def create_maf(dim, num_bijectors=6, layer_sizes=[512, 512]):
  # build layers
  bijectors = []
  for _ in range(num_bijectors):
    bijectors.append(
        tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=ClippedAutoregressiveNetwork(layer_sizes)))
    # bijectors.append(
    #     tfb.MaskedAutoregressiveFlow(
    #         shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
    #             params=2,
    #             hidden_units=layer_sizes,
    #             kernel_initializer="glorot_normal",
    #             bias_initializer="zeros",
    #             activation="relu",
    #             dtype=tf.float64,
    #         )
    #     )
    # )
    bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))
  # discard the last Permute layer.
  chained_bijectors = tfb.Chain(list(reversed(bijectors[:-1])))
  trans_dist = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(
          loc=tf.zeros([dim], dtype=tf.float64)),
      bijector=chained_bijectors)
  return trans_dist
