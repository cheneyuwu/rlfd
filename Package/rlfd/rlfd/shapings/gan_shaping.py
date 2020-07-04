import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfl = tfk.layers

from rlfd import normalizer
from rlfd.shapings import shaping


class Generator(tfk.Model):

  def __init__(self, layer_sizes):
    super(Generator, self).__init__()

    self.network_layers = []
    for size in layer_sizes[:-1]:
      layer = tfl.Dense(units=size,
                        activation="relu",
                        kernel_initializer="glorot_normal")
      self.network_layers.append(layer)
    self.network_layers.append(
        tfl.Dense(units=layer_sizes[-1], kernel_initializer="glorot_normal"))

  def call(self, inputs):
    res = inputs
    for l in self.network_layers:
      res = l(res)
    return res


class Discriminator(tfk.Model):

  def __init__(self, layer_sizes):
    super(Discriminator, self).__init__()

    self.network_layers = []
    for size in layer_sizes[:-1]:
      layer = tfl.Dense(units=size,
                        activation="relu",
                        kernel_initializer="glorot_normal")
      self.network_layers.append(layer)
    self.network_layers.append(
        tfl.Dense(units=layer_sizes[-1], kernel_initializer="glorot_normal"))

  def call(self, inputs):
    res = inputs
    for l in self.network_layers:
      res = l(res)
    return res


class GANShaping(shaping.Shaping):

  def __init__(self, dimo, dimg, dimu, max_u, potential_weight, layer_sizes,
               latent_dim, gp_lambda, critic_iter, norm_obs, norm_eps,
               norm_clip, **kwargs):
    self.init_args = locals()

    super(GANShaping, self).__init__()

    self.dimo = dimo
    self.dimg = dimg
    self.dimu = dimu
    self.max_u = max_u
    self.latent_dim = latent_dim
    self.layer_sizes = layer_sizes
    self.norm_obs = norm_obs
    self.norm_eps = norm_eps
    self.norm_clip = norm_clip
    self.potential_weight = potential_weight
    self.critic_iter = critic_iter
    self.gp_lambda = gp_lambda

    # normalizer for goal and observation.
    self.o_stats = normalizer.Normalizer(self.dimo, self.norm_eps,
                                         self.norm_clip)
    self.g_stats = normalizer.Normalizer(self.dimg, self.norm_eps,
                                         self.norm_clip)

    # Generator & Discriminator
    state_dim = self.dimo[0] + self.dimg[0] + self.dimu[0]
    self.generator = Generator(layer_sizes=layer_sizes + [state_dim])
    self.discriminator = Discriminator(layer_sizes=layer_sizes + [1])
    # create weights
    self.generator(tf.zeros([0, latent_dim]))
    self.discriminator(tf.zeros([0, state_dim]))

    # Train
    self.train_gen = tf.Variable(0, trainable=False)  # counter
    self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,
                                                   beta_1=0.5,
                                                   beta_2=0.9)
    self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,
                                                  beta_1=0.5,
                                                  beta_2=0.9)

  def _update_stats(self, batch):
    # add transitions to normalizer
    if not self.norm_obs:
      return
    self.o_stats.update(batch["o"])
    self.g_stats.update(batch["g"])

  @tf.function
  def potential(self, o, g, u):
    """
    Use the output of the GAN's discriminator as potential.
    """
    o = self.o_stats.normalize(o)
    g = self.g_stats.normalize(g)
    state_tf = tf.concat(axis=1, values=[o, g, u / self.max_u])

    potential = self.discriminator(state_tf)
    potential = self.potential_weight * potential
    return potential

  def before_training_hook(self, batch):
    self._update_stats(batch)

  @tf.function
  def _train_graph(self, o, g, u):

    o = self.o_stats.normalize(o)
    g = self.g_stats.normalize(g)
    state = tf.concat(axis=1, values=[o, g, u / self.max_u])

    with tf.GradientTape(persistent=True) as tape:
      fake_data = self.generator(
          tf.random.uniform([tf.shape(state)[0], self.latent_dim]))
      disc_fake = self.discriminator(fake_data)
      disc_real = self.discriminator(state)
      # discriminator loss on generator (including gp loss)
      alpha = tf.random.uniform(shape=[tf.shape(state)[0]] + [1] *
                                (len(tf.shape(state)) - 1),
                                minval=0.0,
                                maxval=1.0)
      interpolates = alpha * state + (1.0 - alpha) * fake_data
      with tf.GradientTape() as tape2:
        tape2.watch(interpolates)
        disc_interpolates = self.discriminator(interpolates)
      gradients = tape2.gradient(disc_interpolates, interpolates)
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
      gradient_penalty = tf.reduce_mean((slopes - 1)**2)
      disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(
          disc_real) + self.gp_lambda * gradient_penalty
      # generator loss
      gen_loss = -tf.reduce_mean(disc_fake)
    disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
    gen_grads = tape.gradient(gen_loss, self.generator.trainable_weights)

    self.disc_optimizer.apply_gradients(
        zip(disc_grads, self.discriminator.trainable_weights))
    if self.train_gen % self.critic_iter == 0:
      self.gen_optimizer.apply_gradients(
          zip(gen_grads, self.generator.trainable_weights))
    self.train_gen.assign_add(1)

    return disc_loss

  def _train(self, o, g, u, name="", **kwargs):

    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)

    disc_loss_tf = self._train_graph(o_tf, g_tf, u_tf)

    with tf.name_scope('GANShapingLosses'):
      tf.summary.scalar(name=name + ' loss vs training_step',
                        data=disc_loss_tf,
                        step=self.training_step)

    return disc_loss_tf.numpy()

  def _evaluate(self, o, g, u, name="", **kwargs):
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    potential = self.potential(o_tf, g_tf, u_tf)
    potential = np.mean(potential.numpy())
    with tf.name_scope('GANShapingEvaluation'):
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
        "g_stats": self.g_stats.get_weights(),
        "discriminator": self.discriminator.get_weights(),
        "generator": self.generator.get_weights(),
    }
    return state

  def __setstate__(self, state):
    stored_vars = state.pop("tf")
    self.__init__(**state)
    self.o_stats.set_weights(stored_vars["o_stats"])
    self.g_stats.set_weights(stored_vars["g_stats"])
    self.discriminator.set_weights(stored_vars["discriminator"])
    self.generator.set_weights(stored_vars["generator"])