import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
tfk = tf.keras

from rlfd import memory
from rlfd.agents import agent, sac


class SACOffline(sac.SAC):
  """Use SAC both offline and online
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

  @tf.function
  def _train_offline_graph(self, o, o_2, u, r, done):
    # Train alpha (entropy weight)
    if self.auto_alpha:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self.log_alpha)
        with tf.name_scope('OfflineLosses/'):
          alpha_loss = self._alpha_loss_graph(o, self.offline_training_step)
      alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
      self._alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
      self.alpha.assign(tf.exp(self.log_alpha))
      with tf.name_scope('OfflineLosses/'):
        tf.summary.scalar(name='alpha vs {}'.format(
            self.offline_training_step.name),
                          data=self.log_alpha,
                          step=self.offline_training_step)
    # Critic q loss
    criticq_trainable_weights = (self._criticq1.trainable_weights +
                                 self._criticq2.trainable_weights)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(criticq_trainable_weights)
      with tf.name_scope('OfflineLosses/'):
        criticq_loss = self._sac_criticq_loss_graph(o, o_2, u, r, done,
                                                    self.offline_training_step)
    criticq_grads = tape.gradient(criticq_loss, criticq_trainable_weights)
    # Actor loss
    actor_trainable_weights = self._actor.trainable_weights
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(actor_trainable_weights)
      with tf.name_scope('OfflineLosses/'):
        actor_loss = self._sac_actor_loss_graph(o, u,
                                                self.offline_training_step)
    actor_grads = tape.gradient(actor_loss, actor_trainable_weights)

    # Update networks
    self._criticq_optimizer.apply_gradients(
        zip(criticq_grads, criticq_trainable_weights))
    self._actor_optimizer.apply_gradients(
        zip(actor_grads, actor_trainable_weights))

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