import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

from rlfd import logger, memory, normalizer, policies
from rlfd.agents import agent, cql, sac_networks


class CQLDP(cql.CQL):
  """This agent adds on top of normal CQL a double penalty to the Q value
  regularizer.
  """

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
      auto_cql_alpha,
      cql_log_alpha,
      cql_alpha_lr,
      # double penalty specific
      target_lower_bound,
      # double q
      soft_target_tau,
      target_update_freq,
      # online training plus offline data
      use_pretrained_actor,
      use_pretrained_critic,
      use_pretrained_alpha,
      online_data_strategy,
      # online bc regularizer
      bc_params,
      # replay buffer
      buffer_size,
      info):
    agent.Agent.__init__(self, locals())

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

    self.auto_cql_alpha = auto_cql_alpha
    self.cql_log_alpha = tf.constant(cql_log_alpha, dtype=tf.float32)
    self.cql_alpha_lr = cql_alpha_lr
    self.cql_tau = cql_tau

    self.target_lower_bound = target_lower_bound

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
    self.bc_params = bc_params
    self.info = info

    self._create_memory()
    self._create_model()
    self._initialize_training_steps()

  def _cql_criticq_loss_graph(self, o, g, o_2, g_2, u, r, n, done, step):
    pi_2, logprob_pi_2 = self._actor(
        [self._actor_o_norm(o_2),
         self._actor_g_norm(g_2)])

    # Immediate reward
    target_q = r
    # Shaping reward
    if self.online_data_strategy == "Shaping":
      potential_curr = self.shaping.potential(o=o, g=g, u=u)
      potential_next = self.shaping.potential(o=o_2, g=g_2, u=pi_2)
      target_q += (1.0 - done) * tf.pow(self.gamma,
                                        n) * potential_next - potential_curr
    # Q value from next state
    target_next_q1 = self._criticq1_target(
        [self._critic_o_norm(o_2),
         self._critic_g_norm(g_2), pi_2])
    target_next_q2 = self._criticq2_target(
        [self._critic_o_norm(o_2),
         self._critic_g_norm(g_2), pi_2])
    target_next_min_q = tf.minimum(target_next_q1, target_next_q2)
    target_q += ((1.0 - done) * tf.pow(self.gamma, n) *
                 (target_next_min_q - self.alpha * logprob_pi_2))
    target_q = tf.stop_gradient(target_q)

    td_loss_q1 = self._huber_loss(
        target_q,
        self._criticq1([self._critic_o_norm(o),
                        self._critic_g_norm(g), u]))
    td_loss_q2 = self._huber_loss(
        target_q,
        self._criticq2([self._critic_o_norm(o),
                        self._critic_g_norm(g), u]))
    td_loss = td_loss_q1 + td_loss_q2
    # Being Conservative (Eqn.4)
    critic_o = self._critic_o_norm(o)
    critic_g = self._critic_g_norm(g)
    # second term
    max_term_q1 = self._criticq1([critic_o, critic_g, u])
    max_term_q2 = self._criticq2([critic_o, critic_g, u])
    # first term (uniform)
    num_samples = 10
    tiled_critic_o = tf.tile(tf.expand_dims(critic_o, axis=1),
                             [1, num_samples] + [1] * len(self.dimo))
    tiled_critic_g = tf.tile(tf.expand_dims(critic_g, axis=1),
                             [1, num_samples] + [1] * len(self.dimg))
    uni_u_dist = tfd.Uniform(low=-self.max_u * tf.ones(self.dimu),
                             high=self.max_u * tf.ones(self.dimu))
    uni_u = uni_u_dist.sample((tf.shape(u)[0], num_samples))
    logprob_uni_u = tf.reduce_sum(uni_u_dist.log_prob(uni_u),
                                  axis=list(range(2, 2 + len(self.dimu))),
                                  keepdims=True)
    uni_q1 = self._criticq1([tiled_critic_o, tiled_critic_g, uni_u])
    uni_q2 = self._criticq2([tiled_critic_o, tiled_critic_g, uni_u])
    # apply double side penalty
    uni_q1 = tf.abs(uni_q1 - self.target_lower_bound)
    uni_q2 = tf.abs(uni_q2 - self.target_lower_bound)
    # apply double side penalty
    uni_q1_logprob_uni_u = uni_q1 - logprob_uni_u
    uni_q2_logprob_uni_u = uni_q2 - logprob_uni_u
    # first term (policy)
    actor_o = self._actor_o_norm(o)
    actor_g = self._actor_g_norm(g)
    tiled_actor_o = tf.tile(tf.expand_dims(actor_o, axis=1),
                            [1, num_samples] + [1] * len(self.dimo))
    tiled_actor_g = tf.tile(tf.expand_dims(actor_g, axis=1),
                            [1, num_samples] + [1] * len(self.dimg))
    pi, logprob_pi = self._actor([tiled_actor_o, tiled_actor_g])
    pi_q1 = self._criticq1([tiled_critic_o, tiled_critic_g, pi])
    pi_q2 = self._criticq2([tiled_critic_o, tiled_critic_g, pi])
    # apply double side penalty
    pi_q1 = tf.abs(pi_q1 - self.target_lower_bound)
    pi_q2 = tf.abs(pi_q2 - self.target_lower_bound)
    # apply double side penalty
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