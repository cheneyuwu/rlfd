import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

from rlfd import memory
from rlfd.agents import agent, cql


class CQLOnline(cql.CQL):
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
      # online bc regularizer
      bc_params,
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
    self.bc_params = bc_params
    self.info = info

    self._create_memory()
    self._create_model()
    self._initialize_training_steps()

  @tf.function
  def _train_online_graph(self, o, o_2, u, r, done):
    # Train critic q
    criticq_trainable_weights = (self._criticq1.trainable_weights +
                                 self._criticq2.trainable_weights)
    with tf.GradientTape(watch_accessed_variables=False,
                         persistent=True) as tape:
      tape.watch(criticq_trainable_weights)
      if self.auto_cql_alpha:
        tape.watch([self.cql_log_alpha])
      with tf.name_scope('OnlineLosses/'):
        criticq_loss = self._cql_criticq_loss_graph(o, o_2, u, r, done,
                                                    self.online_training_step)
        cql_alpha_loss = -criticq_loss
    criticq_grads = tape.gradient(criticq_loss, criticq_trainable_weights)
    self._criticq_optimizer.apply_gradients(
        zip(criticq_grads, criticq_trainable_weights))
    if self.auto_cql_alpha:
      cql_alpha_grads = tape.gradient(cql_alpha_loss, [self.cql_log_alpha])
      self._cql_alpha_optimizer.apply_gradients(
          zip(cql_alpha_grads, [self.cql_log_alpha]))
    with tf.name_scope('OnlineLosses/'):
      tf.summary.scalar(name='cql alpha vs {}'.format(
          self.online_training_step.name),
                        data=self.cql_log_alpha,
                        step=self.online_training_step)

    # Train actor
    actor_trainable_weights = self._actor.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(actor_trainable_weights)
      with tf.name_scope('OnlineLosses/'):
        actor_loss = self._sac_actor_loss_graph(o, u, self.online_training_step)
    actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
    self._actor_optimizer.apply_gradients(
        zip(actor_grads, actor_trainable_weights))

    # Train alpha (entropy weight)
    if self.auto_alpha:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self.log_alpha)
        with tf.name_scope('OnlineLosses/'):
          alpha_loss = self._alpha_loss_graph(o, self.online_training_step)
      alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
      self._alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
      self.alpha.assign(tf.exp(self.log_alpha))

    # Reduce weight on cql regularization term.
    self.cql_weight.assign(self.cql_weight * self.cql_weight_decay_factor)
    with tf.name_scope('OnlineLosses/'):
      tf.summary.scalar(name='cql_weight vs {}'.format(
          self.online_training_step.name),
                        data=self.cql_weight,
                        step=self.online_training_step)

    self.online_training_step.assign_add(1)