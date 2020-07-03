import abc

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rlfd import logger
from rlfd import memory, normalizer
from rlfd.td3 import td3_networks, gan_network, normalizing_flows

tfd = tfp.distributions


class EnsembleShaping(object):

  def __init__(self, shaping_cls, num_ensembles, num_epochs, batch_size, *args,
               **kwargs):
    self.shapings = [shaping_cls(*args, **kwargs) for _ in range(num_ensembles)]

    self.shaping_cls = shaping_cls
    self.num_epochs = num_epochs
    self.batch_size = batch_size

  def train(self, demo_data):
    dataset = self._construct_dataset(demo_data)
    for i, shaping in enumerate(self.shapings):
      logger.log("Training shaping function #{}...".format(i))

      shaping.training_before_hook(demo_data)

      self.training_step = self.shapings[i].training_step
      for epoch in range(self.num_epochs):
        dataset_iter = dataset.sample(return_iterator=True,
                                      shuffle=True,
                                      include_partial_batch=True)
        dataset_iter(self.batch_size)
        losses = np.empty(0)
        for batch in dataset_iter:
          loss = shaping.train(**batch)  # o, g, u, o_2, g_2, r, n, done
          losses = np.append(losses, loss)
          if epoch % (self.num_epochs / 100) == (self.num_epochs / 100 - 1):
            with tf.name_scope(self.shaping_cls.__name__ + 'Losses'):
              tf.summary.scalar(name='model_' + str(i) + ' loss vs epoch',
                                data=np.mean(losses),
                                step=epoch)
            logger.info("epoch: {} demo shaping loss: {}".format(
                epoch, np.mean(losses)))
            mean_pot = shaping.evaluate(**batch)
            logger.info("epoch: {} mean potential on demo data: {}".format(
                epoch, mean_pot))

      shaping.training_after_hook(demo_data)

  def evaluate(self, *args, **kwargs):
    return

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

  def training_before_hook(self, *args, **kwargs):
    pass

  def training_after_hook(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def potential(self, o, g, u):
    """return the shaping potential, has to be a tf.function"""

  def _train(self, *args, **kwargs):
    """train the shaping potential (implementation)"""

  def _evaluate(self, *args, **kwargs):
    """evaluate the shaping potential (implementation)"""


class OfflineRLShaping(Shaping):

  def __init__(self, dimo, dimg, dimu, max_u, pi_lr, q_lr, polyak, gamma,
               layer_sizes, norm_obs, norm_eps, norm_clip, **kwargs):
    self.init_args = locals()

    super(OfflineRLShaping, self).__init__()

    # Parameters
    self.dimo = dimo
    self.dimg = dimg
    self.dimu = dimu
    self.max_u = max_u
    self.layer_sizes = layer_sizes
    self.norm_obs = norm_obs
    self.norm_eps = norm_eps
    self.norm_clip = norm_clip
    self.pi_lr = pi_lr
    self.q_lr = q_lr
    self.polyak = polyak
    self.gamma = gamma
    # Normalizer for goal and observation.
    self.o_stats = normalizer.Normalizer(self.dimo, self.norm_eps,
                                         self.norm_clip)
    self.g_stats = normalizer.Normalizer(self.dimg, self.norm_eps,
                                         self.norm_clip)
    # Models
    # actor
    self.main_actor = td3_networks.Actor(self.dimo, self.dimg, self.dimu,
                                         self.max_u, self.layer_sizes)
    self.target_actor = td3_networks.Actor(self.dimo, self.dimg, self.dimu,
                                           self.max_u, self.layer_sizes)
    # critic
    self.main_critic = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                           self.max_u, self.layer_sizes)
    self.target_critic = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                             self.max_u, self.layer_sizes)
    # Optimizer
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)

    # Initialize target network
    self._update_target_net(0.0)

  def _update_stats(self, batch):
    # add transitions to normalizer
    if not self.norm_obs:
      return
    self.o_stats.update(batch["o"])
    self.g_stats.update(batch["g"])

  def _update_target_net(self, polyak=None):
    polyak = self.polyak if not polyak else polyak
    copy_func = lambda v: v[0].assign(polyak * v[0] + (1.0 - polyak) * v[1])
    list(map(copy_func, zip(self.target_actor.weights,
                            self.main_actor.weights)))
    list(
        map(copy_func, zip(self.target_critic.weights,
                           self.main_critic.weights)))

  @tf.function
  def potential(self, o, g, u):
    o = self.o_stats.normalize(o)
    g = self.g_stats.normalize(g)
    potential = self.main_critic([o, g, u])
    return potential

  def training_before_hook(self, batch):
    self._update_stats(batch)

  def _critic_loss_graph(self, o, g, o_2, g_2, u, r, n, done):
    # Normalize observations
    norm_o = self.o_stats.normalize(o)
    norm_g = self.g_stats.normalize(g)
    norm_o_2 = self.o_stats.normalize(o_2)
    norm_g_2 = self.g_stats.normalize(g_2)

    u_2 = self.target_actor([norm_o_2, norm_g_2])

    # immediate reward
    target = r
    # bellman term
    target += (1.0 - done) * tf.pow(self.gamma, n) * self.target_critic(
        [norm_o_2, norm_g_2, u_2])
    bellman_error = tf.square(
        tf.stop_gradient(target) - self.main_critic([norm_o, norm_g, u]))

    critic_loss = tf.reduce_mean(bellman_error)

    return critic_loss

  def _actor_loss_graph(self, o, g, u):
    # Normalize observations
    norm_o = self.o_stats.normalize(o)
    norm_g = self.g_stats.normalize(g)

    pi = self.main_actor([norm_o, norm_g])
    actor_loss = -tf.reduce_mean(self.main_critic([norm_o, norm_g, pi]))

    return actor_loss

  @tf.function
  def _train_graph(self, o, g, o_2, g_2, u, r, n, done):
    # Train critic
    critic_trainable_weights = self.main_critic.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(critic_trainable_weights)
      critic_loss = self._critic_loss_graph(o, g, o_2, g_2, u, r, n, done)
    critic_grads = tape.gradient(critic_loss, critic_trainable_weights)
    self.critic_optimizer.apply_gradients(
        zip(critic_grads, critic_trainable_weights))

    # Train actor
    actor_trainable_weights = self.main_actor.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(actor_trainable_weights)
      actor_loss = self._actor_loss_graph(o, g, u)
    actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, actor_trainable_weights))

    return critic_loss, actor_loss

  def _train(self, o, g, o_2, g_2, u, r, n, done, name="", **kwargs):
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    o_2_tf = tf.convert_to_tensor(o_2, dtype=tf.float32)
    g_2_tf = tf.convert_to_tensor(g_2, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    r_tf = tf.convert_to_tensor(r, dtype=tf.float32)
    n_tf = tf.convert_to_tensor(n, dtype=tf.float32)
    done_tf = tf.convert_to_tensor(done, dtype=tf.float32)

    loss_tf, _ = self._train_graph(o_tf, g_tf, o_2_tf, g_2_tf, u_tf, r_tf, n_tf,
                                   done_tf)
    self._update_target_net()

    with tf.name_scope('OfflineRLShapingLosses'):
      tf.summary.scalar(name=name + ' loss vs training_step',
                        data=loss_tf,
                        step=self.training_step)

    return loss_tf.numpy()

  def _evaluate(self, o, g, u, name="", **kwargs):
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    potential = self.potential(o_tf, g_tf, u_tf)
    potential = np.mean(potential.numpy())
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
        "main_actor": self.main_actor.get_weights(),
        "target_actor": self.target_actor.get_weights(),
        "main_critic": self.main_critic.get_weights(),
        "target_critic": self.target_critic.get_weights(),
    }
    return state

  def __setstate__(self, state):
    stored_vars = state.pop("tf")
    self.__init__(**state)
    self.o_stats.set_weights(stored_vars["o_stats"])
    self.g_stats.set_weights(stored_vars["g_stats"])
    self.main_actor.set_weights(stored_vars["main_actor"])
    self.target_actor.set_weights(stored_vars["target_actor"])
    self.main_critic.set_weights(stored_vars["main_critic"])
    self.target_critic.set_weights(stored_vars["target_critic"])


class NFShaping(Shaping):

  def __init__(self, dimo, dimg, dimu, max_u, num_bijectors, layer_sizes,
               num_masked, potential_weight, norm_obs, norm_eps, norm_clip,
               prm_loss_weight, reg_loss_weight, **kwargs):
    self.init_args = locals()

    super(NFShaping, self).__init__()

    # Prepare parameters
    self.dimo = dimo
    self.dimg = dimg
    self.dimu = dimu
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
    self.g_stats = normalizer.Normalizer(self.dimg, self.norm_eps,
                                         self.norm_clip)

    # normalizing flow
    state_dim = self.dimo[0] + self.dimg[0] + self.dimu[0]
    self.nf = normalizing_flows.create_maf(dim=state_dim,
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
    self.g_stats.update(batch["g"])

  @tf.function
  def potential(self, o, g, u):
    o = self.o_stats.normalize(o)
    g = self.g_stats.normalize(g)
    state_tf = tf.concat(axis=1, values=[o, g, u / self.max_u])

    state_tf = tf.cast(state_tf, tf.float64)

    potential = tf.reshape(self.nf.prob(state_tf), (-1, 1))
    potential = tf.math.log(potential + tf.exp(-self.scale))
    potential = potential + self.scale  # shift
    potential = self.potential_weight * potential / self.scale  # scale

    potential = tf.cast(potential, tf.float32)

    return potential

  def training_before_hook(self, batch):
    self._update_stats(batch)

  @tf.function
  def _train_graph(self, o, g, u):

    o = self.o_stats.normalize(o)
    g = self.g_stats.normalize(g)
    state = tf.concat(axis=1, values=[o, g, u / self.max_u])

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

  def _train(self, o, g, u, name="", **kwargs):

    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)

    loss_tf = self._train_graph(o_tf, g_tf, u_tf)

    with tf.name_scope('NFShapingLosses'):
      tf.summary.scalar(name=name + ' loss vs training_step',
                        data=loss_tf,
                        step=self.training_step)

    return loss_tf.numpy()

  def _evaluate(self, o, g, u, name="", **kwargs):
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    potential = self.potential(o_tf, g_tf, u_tf)
    potential = np.mean(potential.numpy())
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
        "nf": list(map(lambda v: v.numpy(), self.nf.variables)),
    }
    return state

  def __setstate__(self, state):
    stored_vars = state.pop("tf")
    self.__init__(**state)
    self.o_stats.set_weights(stored_vars["o_stats"])
    self.g_stats.set_weights(stored_vars["g_stats"])
    list(
        map(lambda v: v[0].assign(v[1]),
            zip(self.nf.variables, stored_vars["nf"])))


class GANShaping(Shaping):

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
    self.generator = gan_network.Generator(layer_sizes=layer_sizes +
                                           [state_dim])
    self.discriminator = gan_network.Discriminator(layer_sizes=layer_sizes +
                                                   [1])
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

  def training_before_hook(self, batch):
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