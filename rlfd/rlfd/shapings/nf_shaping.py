"""Adopted from Eric Jang's normalizing flow tutorial: https://blog.evjang.com/2018/01/nf1.html
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from rlfd import normalizer
from rlfd.shapings import shaping


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
  # Build layers
  bijectors = []
  for _ in range(num_bijectors):
    bijectors.append(
        tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=ClippedAutoregressiveNetwork(layer_sizes)))
    bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))
  # Discard the last Permute layer.
  chained_bijectors = tfb.Chain(list(reversed(bijectors[:-1])))
  trans_dist = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(
          loc=tf.zeros([dim], dtype=tf.float64)),
      bijector=chained_bijectors)
  return trans_dist


class NFShaping(shaping.Shaping):

  def __init__(self, dims, max_u, num_bijectors, layer_sizes, num_masked,
               potential_weight, norm_obs, norm_eps, norm_clip, prm_loss_weight,
               reg_loss_weight, **kwargs):
    self.init_args = locals()

    super(NFShaping, self).__init__()

    # Prepare parameters
    self.dimo = dims["o"]
    self.dimu = dims["u"]
    self.max_u = max_u
    self.num_bijectors = num_bijectors
    self.layer_sizes = layer_sizes
    self.norm_obs = norm_obs
    self.norm_eps = norm_eps
    self.norm_clip = norm_clip
    self.prm_loss_weight = prm_loss_weight
    self.reg_loss_weight = reg_loss_weight
    self.potential_weight = tf.constant(potential_weight, dtype=tf.float64)

    #
    self.learning_rate = 2e-4
    self.scale = tf.constant(5.0, dtype=tf.float64)

    # normalizer for goal and observation.
    self.o_stats = normalizer.Normalizer(self.dimo, self.norm_eps,
                                         self.norm_clip)

    # normalizing flow
    state_dim = self.dimo[0] + self.dimu[0]
    self.nf = create_maf(dim=state_dim,
                         num_bijectors=num_bijectors,
                         layer_sizes=layer_sizes)
    # create weights
    self.nf.sample()
    # optimizers
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

  def _update_stats(self, batch):
    # add transitions to normalizer
    if not self.norm_obs:
      return
    self.o_stats.update(batch["o"])

  @tf.function
  def potential(self, o, u):
    o = self.o_stats.normalize(o)
    state_tf = tf.concat(axis=1, values=[o, u / self.max_u])

    state_tf = tf.cast(state_tf, tf.float64)

    potential = tf.reshape(self.nf.prob(state_tf), (-1, 1))
    potential = tf.math.log(potential + tf.exp(-self.scale))
    potential = potential + self.scale  # shift
    potential = self.potential_weight * potential / self.scale  # scale

    potential = tf.cast(potential, tf.float32)

    return potential

  def before_training_hook(self, batch, **kwargs):
    self._update_stats(batch)

  @tf.function
  def _train_graph(self, o, u):

    o = self.o_stats.normalize(o)
    state = tf.concat(axis=1, values=[o, u / self.max_u])

    state = tf.cast(state, tf.float64)

    with tf.GradientTape() as tape:
      with tf.GradientTape() as tape2:
        tape2.watch(state)
        # loss function that tries to maximize log prob
        # log probability
        neg_log_prob = tf.clip_by_value(-self.nf.log_prob(state), -1e5, 1e5)
        neg_log_prob = tf.reduce_mean(tf.reshape(neg_log_prob, (-1, 1)))
      # regularizer
      jacobian = tape2.gradient(neg_log_prob, state)
      regularizer = tf.norm(jacobian, ord=2)
      loss = self.prm_loss_weight * neg_log_prob + self.reg_loss_weight * regularizer
    grads = tape.gradient(loss, self.nf.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.nf.trainable_variables))

    loss = tf.cast(loss, tf.float32)

    return loss

  def _train(self, o, u, name="", **kwargs):

    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)

    loss_tf = self._train_graph(o_tf, u_tf)

    with tf.name_scope('NFShapingLosses'):
      tf.summary.scalar(name=name + ' loss vs training_step',
                        data=loss_tf,
                        step=self.training_step)

    return loss_tf.numpy()

  def _evaluate(self, o, u, name="", **kwargs):
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    potential = self.potential(o_tf, u_tf)
    potential = np.mean(potential.numpy())
    with tf.name_scope('NFShapingEvaluation'):
      tf.summary.scalar(name=name + ' mean potential vs training_step',
                        data=potential,
                        step=self.training_step)
    return potential

  def __getstate__(self):
    state = {
        k: v
        for k, v in self.init_args.items()
        if not k in ["self", "__class__"]
    }
    state["tf"] = {
        "o_stats": self.o_stats.get_weights(),
        "nf": list(map(lambda v: v.numpy(), self.nf.variables)),
    }
    return state

  def __setstate__(self, state):
    stored_vars = state.pop("tf")
    self.__init__(**state)
    self.o_stats.set_weights(stored_vars["o_stats"])
    list(
        map(lambda v: v[0].assign(v[1]),
            zip(self.nf.variables, stored_vars["nf"])))