import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

from rlfd import memory, normalizer, policies
from rlfd.agents import agent, sac, sac_networks


class CQL(sac.SAC):
  """Default online training algorithm of CQL is SAC.
  """

  def __init__(
      self,
      # environment configuration
      dims,
      max_u,
      eps_length,
      gamma,
      # training
      offline_batch_size,
      online_batch_size,
      online_sample_ratio,
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
      auto_cql_alpha,
      cql_log_alpha,
      cql_alpha_lr,
      cql_weight_decay_factor,
      # double q
      soft_target_tau,
      target_update_freq,
      # online training plus offline data
      use_pretrained_actor,
      use_pretrained_critic,
      use_pretrained_alpha,
      online_data_strategy,
      # replay buffer
      buffer_size,
      info):
    agent.Agent.__init__(self, locals())

    self.dims = dims
    self.dimo = self.dims["o"]
    self.dimu = self.dims["u"]
    self.max_u = max_u
    self.fix_T = fix_T
    self.eps_length = eps_length
    self.gamma = gamma

    self.offline_batch_size = offline_batch_size
    self.online_batch_size = online_batch_size
    self.online_sample_ratio = online_sample_ratio

    self.buffer_size = buffer_size

    self.auto_alpha = auto_alpha
    self.alpha = tf.constant(alpha, dtype=tf.float32)
    self.alpha_lr = 3e-4

    self.auto_cql_alpha = auto_cql_alpha
    self.cql_log_alpha = tf.constant(cql_log_alpha, dtype=tf.float32)
    self.cql_alpha_lr = cql_alpha_lr
    self.cql_tau = cql_tau
    self.cql_weight = tf.Variable(1.0, dtype=tf.float32, trainable=False)
    self.cql_weight_decay_factor = cql_weight_decay_factor

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

    self.use_pretrained_actor = use_pretrained_actor
    self.use_pretrained_critic = use_pretrained_critic
    self.use_pretrained_alpha = use_pretrained_alpha
    self.online_data_strategy = online_data_strategy
    assert self.online_data_strategy in ["None", "BC", "Shaping"]
    self.info = info

    self._create_memory()
    self._create_model()
    self._initialize_training_steps()

  def _create_model(self):
    self._initialize_actor()
    self._initialize_critic()

    # For BC initialization
    self._bc_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)
    # Losses
    self._huber_loss = tfk.losses.Huber(delta=10.0,
                                        reduction=tfk.losses.Reduction.NONE)
    # Entropy regularizer
    if self.auto_alpha:
      self.log_alpha = tf.Variable(0., dtype=tf.float32)
      self.alpha = tf.Variable(0., dtype=tf.float32)
      self.alpha.assign(tf.exp(self.log_alpha))
      self.target_alpha = -np.prod(self.dimu)
      self._alpha_optimizer = tfk.optimizers.Adam(learning_rate=self.alpha_lr)
      self.save_var({"alpha": self.alpha, "log_alpha": self.log_alpha})

    if self.auto_cql_alpha:
      self.cql_log_alpha = tf.Variable(0.0, dtype=tf.float32)
      self._cql_alpha_optimizer = tfk.optimizers.Adam(
          learning_rate=self.cql_alpha_lr)

    # Generate policies
    def process_observation_expl(o):
      return self._actor_o_norm(o)

    self._expl_policy = policies.Policy(
        self.dimo,
        self.dimu,
        get_action=lambda o: self._actor([o], sample=True)[0],
        process_observation=process_observation_expl)

    def process_observation_eval(o):
      self._policy_inspect_graph(o)
      return self._actor_o_norm(o)

    self._eval_policy = policies.Policy(
        self.dimo,
        self.dimu,
        get_action=lambda o: self._actor([o], sample=False)[0],
        process_observation=process_observation_eval)

  def _cql_criticq_loss_graph(self, o, o_2, u, r, done, step):
    pi_2, logprob_pi_2 = self._actor([self._actor_o_norm(o_2)])

    # Immediate reward
    target_q = r
    # Shaping reward
    if self.online_data_strategy == "Shaping":
      potential_curr = self.shaping.potential(o=o, u=u)
      potential_next = self.shaping.potential(o=o_2, u=pi_2)
      target_q += (1.0 - done) * self.gamma * potential_next - potential_curr
    # Q value from next state
    target_next_q1 = self._criticq1_target([self._critic_o_norm(o_2), pi_2])
    target_next_q2 = self._criticq2_target([self._critic_o_norm(o_2), pi_2])
    target_next_min_q = tf.minimum(target_next_q1, target_next_q2)
    target_q += ((1.0 - done) * self.gamma *
                 (target_next_min_q - self.alpha * logprob_pi_2))
    target_q = tf.stop_gradient(target_q)

    td_loss_q1 = self._huber_loss(target_q,
                                  self._criticq1([self._critic_o_norm(o), u]))
    td_loss_q2 = self._huber_loss(target_q,
                                  self._criticq2([self._critic_o_norm(o), u]))
    td_loss = td_loss_q1 + td_loss_q2
    # Being Conservative (Eqn.4)
    critic_o = self._critic_o_norm(o)
    # second term
    max_term_q1 = self._criticq1([critic_o, u])
    max_term_q2 = self._criticq2([critic_o, u])
    # first term (uniform)
    num_samples = 10
    tiled_critic_o = tf.tile(tf.expand_dims(critic_o, axis=1),
                             [1, num_samples] + [1] * len(self.dimo))
    uni_u_dist = tfd.Uniform(low=-self.max_u * tf.ones(self.dimu),
                             high=self.max_u * tf.ones(self.dimu))
    uni_u = uni_u_dist.sample((tf.shape(u)[0], num_samples))
    logprob_uni_u = tf.reduce_sum(uni_u_dist.log_prob(uni_u),
                                  axis=list(range(2, 2 + len(self.dimu))),
                                  keepdims=True)
    uni_q1 = self._criticq1([tiled_critic_o, uni_u])
    uni_q2 = self._criticq2([tiled_critic_o, uni_u])
    uni_q1_logprob_uni_u = uni_q1 - logprob_uni_u
    uni_q2_logprob_uni_u = uni_q2 - logprob_uni_u
    # first term (policy)
    actor_o = self._actor_o_norm(o)
    tiled_actor_o = tf.tile(tf.expand_dims(actor_o, axis=1),
                            [1, num_samples] + [1] * len(self.dimo))
    pi, logprob_pi = self._actor([tiled_actor_o])
    pi_q1 = self._criticq1([tiled_critic_o, pi])
    pi_q2 = self._criticq2([tiled_critic_o, pi])
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

    criticq_loss = (tf.reduce_mean(td_loss) +
                    self.cql_weight * tf.reduce_mean(cql_loss))
    tf.summary.scalar(name='criticq_loss vs {}'.format(step.name),
                      data=criticq_loss,
                      step=step)
    return criticq_loss

  @tf.function
  def _train_offline_graph(self, o, o_2, u, r, done):
    # Train critic q
    criticq_trainable_weights = (self._criticq1.trainable_weights +
                                 self._criticq2.trainable_weights)
    with tf.GradientTape(watch_accessed_variables=False,
                         persistent=True) as tape:
      tape.watch(criticq_trainable_weights)
      if self.auto_cql_alpha:
        tape.watch([self.cql_log_alpha])
      with tf.name_scope('OfflineLosses/'):
        criticq_loss = self._cql_criticq_loss_graph(o, o_2, u, r, done,
                                                    self.offline_training_step)
        cql_alpha_loss = -criticq_loss
    criticq_grads = tape.gradient(criticq_loss, criticq_trainable_weights)
    self._criticq_optimizer.apply_gradients(
        zip(criticq_grads, criticq_trainable_weights))
    if self.auto_cql_alpha:
      cql_alpha_grads = tape.gradient(cql_alpha_loss, [self.cql_log_alpha])
      self._cql_alpha_optimizer.apply_gradients(
          zip(cql_alpha_grads, [self.cql_log_alpha]))
      # clip for numerical stability
      self.cql_log_alpha.assign(tf.clip_by_value(self.cql_log_alpha, -20., 10.))
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
        actor_loss = self._sac_actor_loss_graph(o, u,
                                                self.offline_training_step)
    actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
    self._actor_optimizer.apply_gradients(
        zip(actor_grads, actor_trainable_weights))

    # Train alpha (entropy weight)
    if self.auto_alpha:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self.log_alpha)
        with tf.name_scope('OfflineLosses/'):
          alpha_loss = self._alpha_loss_graph(o, self.offline_training_step)
      alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
      self._alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
      self.alpha.assign(tf.exp(self.log_alpha))

    self.offline_training_step.assign_add(1)

  def train_offline(self):
    with tf.summary.record_if(lambda: self.offline_training_step % 1000 == 0):
      batch = self.offline_buffer.sample(self.offline_batch_size)

      o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
      o_2_tf = tf.convert_to_tensor(batch["o_2"], dtype=tf.float32)
      u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
      r_tf = tf.convert_to_tensor(batch["r"], dtype=tf.float32)
      done_tf = tf.convert_to_tensor(batch["done"], dtype=tf.float32)

      self._train_offline_graph(o_tf, o_2_tf, u_tf, r_tf, done_tf)
      if self.offline_training_step % self.target_update_freq == 0:
        self._copy_weights(self._criticq1, self._criticq1_target)
        self._copy_weights(self._criticq2, self._criticq2_target)