import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
tfk = tf.keras

from rlfd import logger, memory, normalizer, policies
from rlfd.sac import sac, sac_networks
from rlfd.td3 import shaping


class SAC(sac.SAC):
  """This implementation of SAC is the original version that learns an extra
  value function
  """

  def __init__(
      self,
      batch_size,
      # environment configuration
      dims,
      max_u,
      gamma,
      eps_length,
      fix_T,
      norm_eps,
      norm_clip,
      # networks
      layer_sizes,
      q_lr,
      vf_lr,
      pi_lr,
      action_l2,
      # sac specific
      auto_alpha,
      alpha,
      # double q
      polyak,
      # multistep return
      use_n_step_return,
      # play with demonstrations
      buffer_size,
      batch_size_demo,
      sample_demo_buffer,
      use_demo_reward,
      initialize_with_bc,
      initialize_num_epochs,
      demo_strategy,
      bc_params,
      shaping_params,
      info):
    # Store initial args passed into the function
    self.init_args = locals()

    self.buffer_size = buffer_size
    self.batch_size = batch_size

    self.use_demo_reward = use_demo_reward
    self.sample_demo_buffer = sample_demo_buffer
    self.batch_size_demo = batch_size_demo
    self.initialize_with_bc = initialize_with_bc
    self.initialize_num_epochs = initialize_num_epochs

    self.eps_length = eps_length
    self.fix_T = fix_T

    # Parameters
    self.dims = dims
    self.dimo = self.dims["o"]
    self.dimg = self.dims["g"]
    self.dimu = self.dims["u"]
    self.max_u = max_u
    self.layer_sizes = layer_sizes
    self.q_lr = q_lr
    self.vf_lr = vf_lr
    self.pi_lr = pi_lr
    self.action_l2 = action_l2
    self.polyak = polyak

    self.norm_eps = norm_eps
    self.norm_clip = norm_clip

    # multistep return
    self.use_n_step_return = use_n_step_return
    self.n_step_return_steps = eps_length // 5

    # play with demonstrations
    self.demo_strategy = demo_strategy
    assert self.demo_strategy in ["none", "bc", "gan", "nf", "orl"]
    self.bc_params = bc_params
    self.shaping_params = shaping_params
    self.gamma = gamma
    self.info = info

    # SAC specific
    self.auto_alpha = auto_alpha
    self.alpha = tf.constant(alpha, dtype=tf.float32)
    self.alpha_lr = 3e-4

    self._create_memory()
    self._create_network()

    # Generate policies
    def process_observation(o, g):
      norm_o = self._o_stats.normalize(o)
      norm_g = self._g_stats.normalize(g)
      self._policy_inspect_graph(o, g)
      return norm_o, norm_g

    self.eval_policy = policies.Policy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g], sample=False)[0],
        process_observation=process_observation)
    self.expl_policy = policies.Policy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g], sample=True)[0],
        process_observation=process_observation)

    # Losses
    self._huber_loss = tfk.losses.Huber(delta=10.0,
                                        reduction=tfk.losses.Reduction.NONE)

    # Initialize training steps
    self.sac_training_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.exploration_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  def _criticq_loss_graph(self, o, g, o_2, g_2, u, r, n, done):
    norm_o = self._o_stats.normalize(o)
    norm_g = self._g_stats.normalize(g)
    norm_o_2 = self._o_stats.normalize(o_2)
    norm_g_2 = self._g_stats.normalize(g_2)

    # Immediate reward
    target_q = r
    # Shaping reward
    if self.shaping != None:
      pass  # TODO add shaping rewards.
    target_q += ((1.0 - done) * tf.pow(self.gamma, n) *
                 self._vf_target([norm_o_2, norm_g_2]))
    target_q = tf.stop_gradient(target_q)

    td_loss_q1 = self._huber_loss(target_q, self._criticq1([norm_o, norm_g, u]))
    td_loss_q2 = self._huber_loss(target_q, self._criticq2([norm_o, norm_g, u]))
    td_loss = td_loss_q1 + td_loss_q2

    if self.sample_demo_buffer and not self.use_demo_reward:
      # mask off entries from demonstration dataset
      mask = np.concatenate(
          (np.ones(self.batch_size), np.zeros(self.batch_size_demo)), axis=0)
      td_loss = tf.boolean_mask(td_loss, mask)

    criticq_loss = tf.reduce_mean(td_loss)

    with tf.name_scope('SACLosses/'):
      tf.summary.scalar(name='criticq_loss vs sac_training_step',
                        data=criticq_loss,
                        step=self.sac_training_step)

    return criticq_loss

  def _vf_loss_graph(self, o, g):
    norm_o = self._o_stats.normalize(o)
    norm_g = self._g_stats.normalize(g)

    mean_pi, logprob_pi = self._actor([norm_o, norm_g])
    current_q1 = self._criticq1([norm_o, norm_g, mean_pi])
    current_q2 = self._criticq2([norm_o, norm_g, mean_pi])
    current_min_q = tf.minimum(current_q1, current_q2)

    current_v = self._vf([norm_o, norm_g])
    target_v = tf.stop_gradient(current_min_q - self.alpha * logprob_pi)
    td_loss = self._huber_loss(target_v, current_v)

    vf_loss = tf.reduce_mean(td_loss)

    with tf.name_scope('SACLosses/'):
      tf.summary.scalar(name='vf_loss vs sac_training_step',
                        data=vf_loss,
                        step=self.sac_training_step)

    return vf_loss

  @tf.function
  def _train_graph(self, o, g, o_2, g_2, u, r, n, done):
    # Train critic q
    criticq_trainable_weights = (self._criticq1.trainable_weights +
                                 self._criticq2.trainable_weights)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(criticq_trainable_weights)
      criticq_loss = self._criticq_loss_graph(o, g, o_2, g_2, u, r, n, done)
    criticq_grads = tape.gradient(criticq_loss, criticq_trainable_weights)
    self._criticq_optimizer.apply_gradients(
        zip(criticq_grads, criticq_trainable_weights))

    # Train value function
    vf_trainable_weights = self._vf.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vf_trainable_weights)
      vf_loss = self._vf_loss_graph(o, g)
    vf_grads = tape.gradient(vf_loss, vf_trainable_weights)
    self._vf_optimizer.apply_gradients(zip(vf_grads, vf_trainable_weights))

    # Train actor
    actor_trainable_weights = self._actor.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(actor_trainable_weights)
      actor_loss = self._actor_loss_graph(o, g, u)
    actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
    self._actor_optimizer.apply_gradients(
        zip(actor_grads, actor_trainable_weights))

    # Train alpha (entropy weight)
    if self.auto_alpha:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self.log_alpha)
        alpha_loss = self._alpha_loss_graph(o, g)
      alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
      self._alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
      self.alpha.assign(tf.exp(self.log_alpha))

    self.sac_training_step.assign_add(1)

  def update_target_network(self, polyak=None):
    polyak = polyak if polyak else self.polyak
    copy_func = lambda v: v[0].assign(polyak * v[0] + (1.0 - polyak) * v[1])

    list(map(copy_func, zip(self._vf_target.weights, self._vf.weights)))

  def _create_network(self):
    # Normalizer for goal and observation.
    self._o_stats = normalizer.Normalizer(self.dimo, self.norm_eps,
                                          self.norm_clip)
    self._g_stats = normalizer.Normalizer(self.dimg, self.norm_eps,
                                          self.norm_clip)
    # Models
    self._actor = sac_networks.Actor(self.dimo, self.dimg, self.dimu,
                                     self.max_u, self.layer_sizes)
    self._criticq1 = sac_networks.CriticQ(self.dimo, self.dimg, self.dimu,
                                          self.max_u, self.layer_sizes)
    self._criticq2 = sac_networks.CriticQ(self.dimo, self.dimg, self.dimu,
                                          self.max_u, self.layer_sizes)
    self._vf = sac_networks.CriticV(self.dimo, self.dimg, self.layer_sizes)
    self._vf_target = sac_networks.CriticV(self.dimo, self.dimg,
                                           self.layer_sizes)
    self.update_target_network(polyak=0.0)
    # Optimizers
    self._actor_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)
    self._criticq_optimizer = tfk.optimizers.Adam(learning_rate=self.q_lr)
    self._vf_optimizer = tfk.optimizers.Adam(learning_rate=self.vf_lr)
    self._bc_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)
    # Entropy regularizer
    if self.auto_alpha:
      self.log_alpha = tf.Variable(0., dtype=tf.float32)
      self.alpha = tf.Variable(0., dtype=tf.float32)
      self.alpha.assign(tf.exp(self.log_alpha))
      self.target_alpha = -self.dimu[0]
      self._alpha_optimizer = tfk.optimizers.Adam(learning_rate=self.alpha_lr)

    # Add shaping reward
    shaping_class = {
        "nf": shaping.NFShaping,
        "gan": shaping.GANShaping,
        "orl": shaping.OfflineRLShaping
    }
    if self.demo_strategy in shaping_class.keys():
      # instantiate shaping version 1
      self.shaping = shaping.EnsembleShaping(
          shaping_cls=shaping_class[self.demo_strategy],
          num_ensembles=self.shaping_params["num_ensembles"],
          batch_size=self.shaping_params["batch_size"],
          num_epochs=self.shaping_params["num_epochs"],
          dimo=self.dimo,
          dimg=self.dimg,
          dimu=self.dimu,
          max_u=self.max_u,
          gamma=self.gamma,
          norm_obs=True,
          norm_eps=self.norm_eps,
          norm_clip=self.norm_clip,
          **self.shaping_params[self.demo_strategy].copy())
    else:
      self.shaping = None

    # Meta-learning for weight on potential
    self.potential_weight = tf.Variable(1.0, trainable=False)
    self.potential_decay_scale = self.shaping_params["potential_decay_scale"]
    self.potential_decay_epoch = 0  # eventually becomes self.shaping_params["potential_decay_epoch"]

  def __getstate__(self):
    """
    Our policies can be loaded from pkl, but after unpickling you cannot continue training.
    """
    state = {k: v for k, v in self.init_args.items() if not k == "self"}
    state["shaping"] = self.shaping
    state["tf"] = {
        "o_stats": self._o_stats.get_weights(),
        "g_stats": self._g_stats.get_weights(),
        "actor": self._actor.get_weights(),
        "criticq1": self._criticq1.get_weights(),
        "criticq2": self._criticq2.get_weights(),
        "vf": self._vf.get_weights(),
        "vf_target": self._vf_target.get_weights(),
    }
    return state

  def __setstate__(self, state):
    stored_vars = state.pop("tf")
    shaping = state.pop("shaping")
    self.__init__(**state)
    self._o_stats.set_weights(stored_vars["o_stats"])
    self._g_stats.set_weights(stored_vars["g_stats"])
    self._actor.set_weights(stored_vars["actor"])
    self._criticq1.set_weights(stored_vars["criticq1"])
    self._criticq2.set_weights(stored_vars["criticq2"])
    self._vf.set_weights(stored_vars["vf"])
    self._vf_target.set_weights(stored_vars["vf_target"])
    self.shaping = shaping
