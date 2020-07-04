import numpy as np
import tensorflow as tf

from rlfd import normalizer
from rlfd.agents import td3_networks
from rlfd.shapings import shaping


class OfflineRLShaping(shaping.Shaping):

  def __init__(self, dims, max_u, pi_lr, q_lr, polyak, gamma, layer_sizes,
               norm_obs, norm_eps, norm_clip, **kwargs):
    self.init_args = locals()

    super(OfflineRLShaping, self).__init__()

    # Parameters
    self.dimo = dims["o"]
    self.dimg = dims["g"]
    self.dimu = dims["u"]
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

  def before_training_hook(self, batch):
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