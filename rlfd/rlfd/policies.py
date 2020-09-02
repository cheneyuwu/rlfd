import abc
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras


class Policy(object):
  """Abstract Policy class for Policies.
  """

  def __init__(self,
               dimo,
               dimu,
               get_action,
               process_observation=lambda o: o,
               process_action=lambda u: u):
    self._dimo = dimo
    self._dimu = dimu
    # TF graphables.
    self._get_action = get_action
    self._process_observation = process_observation
    self._process_action = process_action

  def __call__(self, o):
    batch_o = o.reshape((-1, *self._dimo))

    o_tf = tf.convert_to_tensor(batch_o, dtype=tf.float32)

    u = self._call_graph(o_tf).numpy()

    if len(batch_o.shape) != len(o.shape):
      assert len(batch_o.shape) - len(o.shape) == 1, "Batch dim must be 1."
      u = u[0]
    assert u.shape[-len(self._dimu):] == self._dimu

    return u

  @tf.function
  def _call_graph(self, o):
    """TF graph to compute the output."""
    o = self._process_observation(o)
    u = self._get_action(o)
    u = self._process_action(u)
    return u


class GaussianPolicy(Policy):

  def __init__(self,
               dimo,
               dimu,
               get_action,
               max_u,
               noise_eps,
               process_observation=lambda o: o,
               process_action=lambda u: u):
    super().__init__(dimo, dimu, self._get_action, process_observation,
                     process_action)

    self._raw_get_action = get_action
    self._max_u = max_u
    self._noise_eps = noise_eps

  def _get_action(self, o):
    noise = tf.random.normal((tf.shape(o)[0],) + self._dimu) * self._max_u
    u = self._raw_get_action(o)
    u = tf.clip_by_value(u + noise, -self._max_u, self._max_u)
    return u


class EpsilonGreedyPolicy(Policy):

  def __init__(self,
               dimo,
               dimu,
               get_action,
               max_u,
               random_prob,
               process_observation=lambda o: o,
               process_action=lambda u: u):
    super().__init__(dimo, dimu, self._get_action, process_observation,
                     process_action)
    self._max_u = max_u
    self._raw_get_action = get_action
    self._binomial_dist = tfd.Binomial(1, probs=random_prob)

  def _get_action(self, o):
    u = self._raw_get_action(o)
    mask = self._binomial_dist.sample(o.shape[0])
    u_rand = self._random_action(o)
    u = u + tf.reshape(mask, [-1] + [1] * len(self._dimu)) * (u_rand - u)
    return u

  def _random_action(self):
    return tf.random.uniform([tf.shape(o)[0]] + self._dimu, -1.0,
                             1.0) * self._max_u


class GaussianEpsilonGreedyPolicy(Policy):

  def __init__(self,
               dimo,
               dimu,
               get_action,
               max_u,
               noise_eps,
               random_prob,
               process_observation=lambda o: o,
               process_action=lambda u: u):
    super().__init__(dimo, dimu, self._get_action, process_observation,
                     process_action)
    self._max_u = max_u
    self._noise_eps = noise_eps
    self._raw_get_action = get_action
    self._binomial_dist = tfd.Binomial(1, probs=random_prob)

  def _get_action(self, o):
    # Add Gaussian noise
    noise = tf.random.normal((tf.shape(o)[0],) + self._dimu) * self._max_u
    u = self._raw_get_action(o)
    u = tf.clip_by_value(u + noise, -self._max_u, self._max_u)
    # Epsilon greedy
    mask = self._binomial_dist.sample(o.shape[0])
    u_rand = self._random_action(o)
    u = u + tf.reshape(mask, [-1] + [1] * len(self._dimu)) * (u_rand - u)
    return u

  def _random_action(self, o):
    return tf.random.uniform(
        (tf.shape(o)[0],) + self._dimu, -1.0, 1.0) * self._max_u


class RandomPolicy(Policy):

  def __init__(self, dimo, dimu, max_u):
    super().__init__(dimo, dimu, self._get_action)
    self._max_u = max_u

  def _get_action(self, o):
    return tf.random.uniform(
        (tf.shape(o)[0],) + self._dimu, -1.0, 1.0) * self._max_u
