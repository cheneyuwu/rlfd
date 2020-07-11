import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

from rlfd import logger, memory, normalizer, policies
from rlfd.agents import agent, sac, sac_networks


class CQL(sac.SAC):

  def __init__(
      self,
      # environment configuration
      dims,
      max_u,
      eps_length,
      gamma,
      # training
      online_batch_size,
      offline_batch_size,
      fix_T,
      # normalize
      norm_obs_online,
      norm_obs_offline,
      norm_eps,
      norm_clip,
      # networks
      layer_sizes,
      q_lr,
      pi_lr,
      action_l2,
      # sac specific
      auto_alpha,
      alpha,
      # cql specific
      cql_tau,
      cql_alpha_lr,
      # double q
      soft_target_tau,
      target_update_freq,
      # online training plus offline data
      online_data_strategy,
      # online bc regularizer
      bc_params,
      # replay buffer
      buffer_size,
      info):
    # Store initial args passed into the function
    self.init_args = locals()

    self.dims = dims
    self.dimo = self.dims["o"]
    self.dimg = self.dims["g"]
    self.dimu = self.dims["u"]
    self.max_u = max_u
    self.fix_T = fix_T
    self.eps_length = eps_length
    self.gamma = gamma

    self.online_batch_size = online_batch_size
    self.offline_batch_size = offline_batch_size

    self.buffer_size = buffer_size

    self.auto_alpha = auto_alpha
    self.alpha = tf.constant(alpha, dtype=tf.float32)
    self.alpha_lr = 3e-4

    self.cql_log_alpha = tf.Variable(0.0, dtype=tf.float32)
    # self.cql_alpha.assign(tf.exp(self.cql_log_alpha))
    self.cql_tau = cql_tau
    self.cql_alpha_lr = cql_alpha_lr
    self._cql_alpha_optimizer = tfk.optimizers.Adam(
        learning_rate=self.cql_alpha_lr)

    self.layer_sizes = layer_sizes
    self.q_lr = q_lr
    self.pi_lr = pi_lr
    self.action_l2 = action_l2
    self.soft_target_tau = soft_target_tau
    self.target_update_freq = target_update_freq

    self.norm_obs_online = norm_obs_online
    self.norm_obs_offline = norm_obs_offline
    self.norm_eps = norm_eps
    self.norm_clip = norm_clip

    self.online_data_strategy = online_data_strategy
    assert self.online_data_strategy in ["None", "BC", "Shaping"]
    self.bc_params = bc_params
    self.info = info

    # Create Replaybuffers
    buffer_shapes = dict(o=self.dimo,
                         o_2=self.dimo,
                         u=self.dimu,
                         r=(1,),
                         ag=self.dimg,
                         ag_2=self.dimg,
                         g=self.dimg,
                         g_2=self.dimg,
                         done=(1,))
    if self.fix_T:
      buffer_shapes = {
          k: (self.eps_length,) + v for k, v in buffer_shapes.items()
      }
      self.online_buffer = memory.EpisodeBaseReplayBuffer(
          buffer_shapes, self.buffer_size, self.eps_length)
      self.offline_buffer = memory.EpisodeBaseReplayBuffer(
          buffer_shapes, self.buffer_size, self.eps_length)
    else:
      self.online_buffer = memory.StepBaseReplayBuffer(buffer_shapes,
                                                       self.buffer_size)
      self.offline_buffer = memory.StepBaseReplayBuffer(buffer_shapes,
                                                        self.buffer_size)

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
    self._criticq1_target = sac_networks.CriticQ(self.dimo, self.dimg,
                                                 self.dimu, self.max_u,
                                                 self.layer_sizes)
    self._criticq2_target = sac_networks.CriticQ(self.dimo, self.dimg,
                                                 self.dimu, self.max_u,
                                                 self.layer_sizes)
    self._update_target_network(soft_target_tau=1.0)
    # Optimizers
    self._actor_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)
    self._criticq_optimizer = tfk.optimizers.Adam(learning_rate=self.q_lr)
    self._bc_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)
    # Entropy regularizer
    if self.auto_alpha:
      self.log_alpha = tf.Variable(0., dtype=tf.float32)
      self.alpha = tf.Variable(0., dtype=tf.float32)
      self.alpha.assign(tf.exp(self.log_alpha))
      self.target_alpha = -np.prod(self.dimu)
      self._alpha_optimizer = tfk.optimizers.Adam(learning_rate=self.alpha_lr)

    # Generate policies
    def process_observation_expl(o, g):
      norm_o = self._o_stats.normalize(o)
      norm_g = self._g_stats.normalize(g)
      return norm_o, norm_g

    self._expl_policy = policies.Policy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g], sample=True)[0],
        process_observation=process_observation_expl)

    def process_observation_eval(o, g):
      norm_o = self._o_stats.normalize(o)
      norm_g = self._g_stats.normalize(g)
      self._policy_inspect_graph(o, g)
      return norm_o, norm_g

    self._eval_policy = policies.Policy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g], sample=False)[0],
        process_observation=process_observation_eval)

    # Losses
    self._huber_loss = tfk.losses.Huber(delta=10.0,
                                        reduction=tfk.losses.Reduction.NONE)

    # Initialize training steps
    self.offline_training_step = tf.Variable(0,
                                             trainable=False,
                                             name="offline_training_step",
                                             dtype=tf.int64)
    self.online_training_step = tf.Variable(0,
                                            trainable=False,
                                            name="online_training_step",
                                            dtype=tf.int64)

  def _criticq_loss_graph(self, o, g, o_2, g_2, u, r, n, done, step):
    # Normalize observations
    norm_o = self._o_stats.normalize(o)
    norm_g = self._g_stats.normalize(g)
    norm_o_2 = self._o_stats.normalize(o_2)
    norm_g_2 = self._g_stats.normalize(g_2)

    pi_2, logprob_pi_2 = self._actor([norm_o_2, norm_g_2])

    # Immediate reward
    target_q = r
    # Shaping reward
    if self.online_data_strategy == "Shaping":
      potential_curr = self.shaping.potential(o=o, g=g, u=u)
      potential_next = self.shaping.potential(o=o_2, g=g_2, u=pi_2)
      target_q += (1.0 - done) * tf.pow(self.gamma,
                                        n) * potential_next - potential_curr
    # Q value from next state
    target_next_q1 = self._criticq1_target([norm_o_2, norm_g_2, pi_2])
    target_next_q2 = self._criticq2_target([norm_o_2, norm_g_2, pi_2])
    target_next_min_q = tf.minimum(target_next_q1, target_next_q2)
    target_q += ((1.0 - done) * tf.pow(self.gamma, n) *
                 (target_next_min_q - self.alpha * logprob_pi_2))
    target_q = tf.stop_gradient(target_q)

    td_loss_q1 = self._huber_loss(target_q, self._criticq1([norm_o, norm_g, u]))
    td_loss_q2 = self._huber_loss(target_q, self._criticq2([norm_o, norm_g, u]))
    td_loss = td_loss_q1 + td_loss_q2
    # Being Conservative (Eqn.4)
    # second term
    max_term_q1 = self._criticq1([norm_o, norm_g, u])
    max_term_q2 = self._criticq2([norm_o, norm_g, u])
    # first term (uniform)
    num_samples = 10
    tiled_norm_o = tf.tile(tf.expand_dims(norm_o, axis=1),
                           [1, num_samples] + [1] * len(self.dimo))
    tiled_norm_g = tf.tile(tf.expand_dims(norm_g, axis=1),
                           [1, num_samples] + [1] * len(self.dimg))
    uni_u_dist = tfd.Uniform(low=-self.max_u * tf.ones(self.dimu),
                             high=self.max_u * tf.ones(self.dimu))
    uni_u = uni_u_dist.sample((tf.shape(u)[0], num_samples))
    logprob_uni_u = tf.reduce_sum(uni_u_dist.log_prob(uni_u),
                                  axis=list(range(2, 2 + len(self.dimu))),
                                  keepdims=True)
    uni_q1 = self._criticq1([tiled_norm_o, tiled_norm_g, uni_u])
    uni_q2 = self._criticq2([tiled_norm_o, tiled_norm_g, uni_u])
    uni_q1_logprob_uni_u = uni_q1 - logprob_uni_u
    uni_q2_logprob_uni_u = uni_q2 - logprob_uni_u
    # first term (policy)
    pi, logprob_pi = self._actor([tiled_norm_o, tiled_norm_g])
    pi_q1 = self._criticq1([tiled_norm_o, tiled_norm_g, pi])
    pi_q2 = self._criticq2([tiled_norm_o, tiled_norm_g, pi])
    pi_q1_logprob_pi = pi_q1 - logprob_pi
    pi_q2_logprob_pi = pi_q2 - logprob_pi
    # Note: log(2N) not included in this case since it is constant.
    log_sum_exp_q1 = tf.math.reduce_logsumexp(tf.concat(
        (uni_q1_logprob_uni_u, pi_q1_logprob_pi), axis=1),
                                              axis=1)
    log_sum_exp_q2 = tf.math.reduce_logsumexp(tf.concat(
        (uni_q2_logprob_uni_u, pi_q2_logprob_pi), axis=1),
                                              axis=1)
    cql_loss_q1 = (tf.exp(self.cql_log_alpha) *
                   (log_sum_exp_q1 - max_term_q1 - self.cql_tau))
    cql_loss_q2 = (tf.exp(self.cql_log_alpha) *
                   (log_sum_exp_q2 - max_term_q2 - self.cql_tau))
    cql_loss = cql_loss_q1 + cql_loss_q2

    criticq_loss = tf.reduce_mean(td_loss) + tf.reduce_mean(cql_loss)
    tf.summary.scalar(name='criticq_loss vs {}'.format(step.name),
                      data=criticq_loss,
                      step=step)
    return criticq_loss

  @tf.function
  def _train_offline_graph(self, o, g, o_2, g_2, u, r, n, done):
    # Train critic q
    criticq_trainable_weights = (self._criticq1.trainable_weights +
                                 self._criticq2.trainable_weights)
    with tf.GradientTape(watch_accessed_variables=False,
                         persistent=True) as tape:
      tape.watch(criticq_trainable_weights)
      tape.watch([self.cql_log_alpha])
      with tf.name_scope('OfflineLosses/'):
        criticq_loss = self._criticq_loss_graph(o, g, o_2, g_2, u, r, n, done,
                                                self.offline_training_step)
        cql_alpha_loss = -criticq_loss
    criticq_grads = tape.gradient(criticq_loss, criticq_trainable_weights)
    self._criticq_optimizer.apply_gradients(
        zip(criticq_grads, criticq_trainable_weights))
    cql_alpha_grads = tape.gradient(cql_alpha_loss, [self.cql_log_alpha])
    self._cql_alpha_optimizer.apply_gradients(
        zip(cql_alpha_grads, [self.cql_log_alpha]))
    # self.cql_alpha.assign(tf.exp(self.cql_log_alpha))
    with tf.name_scope('OfflineLosses/'):
      tf.summary.scalar(name='cql alpha vs {}'.format(
          self.offline_training_step.name),
                        data=self.cql_log_alpha,
                        step=self.offline_training_step)

    # Train actor
    actor_trainable_weights = self._actor.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(actor_trainable_weights)
      with tf.name_scope('OfflineLosses/'):
        actor_loss = self._actor_loss_graph(o, g, u, self.offline_training_step)
    actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
    self._actor_optimizer.apply_gradients(
        zip(actor_grads, actor_trainable_weights))

    # Train alpha (entropy weight)
    if self.auto_alpha:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self.log_alpha)
        with tf.name_scope('OfflineLosses/'):
          alpha_loss = self._alpha_loss_graph(o, g, self.offline_training_step)
      alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
      self._alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
      self.alpha.assign(tf.exp(self.log_alpha))

    self.offline_training_step.assign_add(1)

  def train_offline(self):
    with tf.summary.record_if(lambda: self.offline_training_step % 200 == 0):
      batch = self.offline_buffer.sample(self.offline_batch_size)

      o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
      g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
      o_2_tf = tf.convert_to_tensor(batch["o_2"], dtype=tf.float32)
      g_2_tf = tf.convert_to_tensor(batch["g_2"], dtype=tf.float32)
      u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
      r_tf = tf.convert_to_tensor(batch["r"], dtype=tf.float32)
      n_tf = tf.convert_to_tensor(batch["n"], dtype=tf.float32)
      done_tf = tf.convert_to_tensor(batch["done"], dtype=tf.float32)

      self._train_offline_graph(o_tf, g_tf, o_2_tf, g_2_tf, u_tf, r_tf, n_tf,
                                done_tf)
      if self.offline_training_step % self.target_update_freq == 0:
        self._update_target_network()

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
        "criticq1_target": self._criticq1_target.get_weights(),
        "criticq2_target": self._criticq2_target.get_weights(),
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
    self._criticq1_target.set_weights(stored_vars["criticq1_target"])
    self._criticq2_target.set_weights(stored_vars["criticq2_target"])
    self.shaping = shaping
