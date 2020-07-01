import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfl = tfk.layers
tfd = tfp.distributions


class Actor(tfk.Model):
  LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
  LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
  EPS = 1e-6

  def __init__(self, dimo, dimg, dimu, max_u, layer_sizes, name="pi"):
    super().__init__(name=name)

    self._dimo = dimo
    self._dimg = dimg
    self._dimu = dimu
    self._max_u = max_u
    self._dist = tfd.MultivariateNormalDiag

    self._mlp_layers = []
    for size in layer_sizes:
      layer = tfl.Dense(units=size, activation="relu")
      self._mlp_layers.append(layer)
    self._mean_output_layer = tfl.Dense(units=self._dimu[0])
    self._logstd_output_layer = tfl.Dense(units=self._dimu[0])
    # Create weights
    self([tf.zeros([0, *self._dimo]), tf.zeros([0, *self._dimg])])

  @tf.function
  def call(self, inputs, sample=True):
    mean, logstd = self._compute_dist(inputs)
    mean_pi = self._dist(mean, logstd).sample() if sample else mean
    logprob_pi = self._dist(mean, logstd).log_prob(mean_pi)

    squashed_mean_pi = tf.tanh(mean_pi)
    squashed_logprob_pi = self._squash_correction(logprob_pi, squashed_mean_pi)

    return squashed_mean_pi * self._max_u, squashed_logprob_pi

  @tf.function
  def compute_log_prob(self, inputs):
    o, g, u = inputs
    u /= self._max_u
    mean, logstd = self._compute_dist([o, g])
    logprob_pi = self._dist(mean, logstd).log_prob(u)
    squashed_logprob_pi = self._squash_correction(logprob_pi, u)
    return squashed_logprob_pi

  @tf.function
  def compute_entropy(self, inputs):
    mean, logstd = self._compute_dist(inputs)
    return self._dist(mean, logstd).entropy()

  def _compute_dist(self, inputs):
    """Compute multivariate normal distribution."""
    o, g = inputs
    res = tf.concat([o, g], axis=1)
    for l in self._mlp_layers:
      res = l(res)
    mean = self._mean_output_layer(res)
    logstd = self._logstd_output_layer(res)
    logstd = tf.clip_by_value(logstd, self.LOG_SIG_CAP_MIN,
                              self.LOG_SIG_CAP_MAX)
    return mean, logstd

  def _squash_correction(self, logprob_pi, squashed_mean_pi):
    diff = tf.reduce_sum(tf.math.log(1. - squashed_mean_pi**2 + self.EPS),
                         axis=1)
    return logprob_pi - diff


class CriticV(tfk.Model):

  def __init__(self, dimo, dimg, layer_sizes, name="vf"):
    super().__init__(name=name)

    self._dimo = dimo
    self._dimg = dimg

    self._mlp_layers = []
    for size in layer_sizes:
      layer = tfl.Dense(units=size,
                        activation="relu",
                        kernel_initializer="glorot_normal")
      self._mlp_layers.append(layer)
    self._output_layer = tfl.Dense(units=1, kernel_initializer="glorot_normal")
    # Create weights
    self([tf.zeros([0, *self._dimo]), tf.zeros([0, *self._dimg])])

  @tf.function
  def call(self, inputs):
    o, g = inputs
    res = tf.concat([o, g], axis=1)
    for l in self._mlp_layers:
      res = l(res)
    res = self._output_layer(res)
    return res


class CriticQ(tfk.Model):

  def __init__(self, dimo, dimg, dimu, max_u, layer_sizes, name="qf"):
    super().__init__(name=name)

    self._dimo = dimo
    self._dimg = dimg
    self._dimu = dimu
    self._max_u = max_u

    self._mlp_layers = []
    for size in layer_sizes:
      layer = tfl.Dense(units=size,
                        activation="relu",
                        kernel_initializer="glorot_normal")
      self._mlp_layers.append(layer)
    self._output_layer = tfl.Dense(units=1, kernel_initializer="glorot_normal")
    # Create weights
    self([
        tf.zeros([0, *self._dimo]),
        tf.zeros([0, *self._dimg]),
        tf.zeros([0, *self._dimu])
    ])

  @tf.function
  def call(self, inputs):
    o, g, u = inputs
    res = tf.concat([o, g, u / self._max_u], axis=1)
    for l in self._mlp_layers:
      res = l(res)
    res = self._output_layer(res)
    return res
