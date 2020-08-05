import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tf.keras
tfl = tfk.layers

from rlfd import memory, normalizer, policies
from rlfd.agents import agent, sac_networks


class ClippedAutoregressiveNetwork(tfk.Model):

  def __init__(self, layer_sizes, name="can"):
    super().__init__(name=name)
    self.autoregressive_network = tfb.AutoregressiveNetwork(
        params=2,
        hidden_units=layer_sizes,
        kernel_initializer="glorot_normal",
        bias_initializer="zeros",
        activation="relu",
        dtype=tf.float32,
    )

  def call(self, x):
    x = self.autoregressive_network(x)
    shift, log_scale = tf.unstack(x, num=2, axis=-1)

    clip_log_scale = tf.clip_by_value(log_scale, -5.0, 3.0)
    log_scale = log_scale + tf.stop_gradient(clip_log_scale - log_scale)

    return shift, log_scale


class MAF(tfk.Model):

  def __init__(self, dim, num_bijectors=4, layer_sizes=[256, 256], name="maf"):
    super().__init__(name=name)
    # Build layers
    bijectors = []
    for _ in range(num_bijectors):
      bijectors.append(
          tfb.MaskedAutoregressiveFlow(
              shift_and_log_scale_fn=ClippedAutoregressiveNetwork(layer_sizes)))
      bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))
    # Discard the last Permute layer.
    chained_bijectors = tfb.Chain(list(reversed(bijectors[:-1])))
    self._trans_dist = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(
            loc=tf.zeros([dim], dtype=tf.float32)),
        bijector=chained_bijectors)
    # Create weights
    self._trans_dist.sample()

  def call(self, inputs):
    return self._trans_dist.log_prob(inputs)


class NF(agent.Agent):
  """This agent does not provide a policy. It is used to pre-train a critic.
  """

  def __init__(
      self,
      # environment configuration
      dims,
      max_u,
      eps_length,
      # training
      offline_batch_size,
      fix_T,
      # normalize
      norm_obs_offline,
      norm_eps,
      norm_clip,
      q_lr,
      layer_sizes,
      # networks
      maf_lr,
      maf_layer_sizes,
      num_bijectors,
      prm_loss_weight,
      reg_loss_weight,
      logprob_scale,
      min_logprob,
      # replay buffer
      buffer_size,
      info):
    super().__init__(locals())

    self.dims = dims
    self.dimo = self.dims["o"]
    self.dimg = self.dims["g"]
    self.dimu = self.dims["u"]
    self.max_u = max_u
    self.fix_T = fix_T
    self.eps_length = eps_length

    self.offline_batch_size = offline_batch_size

    self.buffer_size = buffer_size

    self.q_lr = q_lr
    self.layer_sizes = layer_sizes

    self.maf_lr = maf_lr
    self.maf_layer_sizes = maf_layer_sizes
    self.num_bijectors = num_bijectors
    self.prm_loss_weight = prm_loss_weight
    self.reg_loss_weight = reg_loss_weight
    self.logprob_scale = logprob_scale
    self.min_logprob = min_logprob

    self.norm_obs_offline = norm_obs_offline
    self.norm_eps = norm_eps
    self.norm_clip = norm_clip

    self.info = info

    self._create_memory()
    self._create_model()
    self._initialize_training_steps()

  def _create_memory(self):
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

  def _initialize_maf(self):
    self._maf_o_norm = normalizer.Normalizer(self.dimo, self.norm_eps,
                                             self.norm_clip)
    self._maf_g_norm = normalizer.Normalizer(self.dimg, self.norm_eps,
                                             self.norm_clip)
    self._maf = MAF(dim=self.dimo[0] + self.dimg[0] + self.dimu[0],
                    num_bijectors=self.num_bijectors,
                    layer_sizes=self.layer_sizes)
    self._maf_optimizer = tfk.optimizers.Adam(learning_rate=self.maf_lr)

    self._maf_models = {
        "maf": self._maf,
    }
    self.save_model(self._maf_models)

  def _initialize_critic(self):
    self._critic_o_norm = normalizer.Normalizer(self.dimo, self.norm_eps,
                                                self.norm_clip)
    self._critic_g_norm = normalizer.Normalizer(self.dimg, self.norm_eps,
                                                self.norm_clip)

    self._critic = sac_networks.CriticQ(self.dimo, self.dimg, self.dimu,
                                        self.max_u, self.layer_sizes)

    self._critic_optimizer = tfk.optimizers.Adam(learning_rate=self.q_lr)

    self._critic_models = {
        "critic_o_norm": self._critic_o_norm,
        "critic_g_norm": self._critic_g_norm,
        "criticq1": self._critic,
        "criticq2": self._critic,
        "criticq1_target": self._critic,
        "criticq2_target": self._critic,
    }
    self.save_model(self._critic_models)

  def _create_model(self):
    self._initialize_maf()
    self._initialize_critic()
    # Losses
    self._huber_loss = tfk.losses.Huber(delta=10.0,
                                        reduction=tfk.losses.Reduction.NONE)

    self._expl_policy = self._eval_policy = policies.RandomPolicy(
        self.dimo, self.dimg, self.dimu, self.max_u)

  def _initialize_training_steps(self):
    self.offline_training_step = tf.Variable(0,
                                             trainable=False,
                                             name="offline_training_step",
                                             dtype=tf.int64)
    self.online_training_step = tf.Variable(0,
                                            trainable=False,
                                            name="online_training_step",
                                            dtype=tf.int64)

  @property
  def expl_policy(self):
    return self._expl_policy

  @property
  def eval_policy(self):
    return self._eval_policy

  @tf.function
  def estimate_q_graph(self, o, g, u):
    """A convenient function for shaping"""
    return self._critic([self._critic_o_norm(o), self._critic_g_norm(g), u])

  def before_training_hook(self, data_dir=None, env=None, **kwargs):
    """Adds data to the offline replay buffer and add shaping"""
    # Offline data
    # D4RL
    experiences = env.get_dataset()  # T not fixed by assumption
    if experiences:
      self.offline_buffer.store(experiences)
    # Ours
    demo_file = osp.join(data_dir, "demo_data.npz")
    if (not experiences) and osp.isfile(demo_file):
      experiences = self.offline_buffer.load_from_file(data_file=demo_file)
    if self.norm_obs_offline:
      assert experiences, "Offline dataset does not exist."
      self._update_stats(experiences)

  def _update_stats(self, experiences):
    # add transitions to normalizer
    if self.fix_T:
      transitions = {
          k: v.reshape((v.shape[0] * v.shape[1], v.shape[2])).copy()
          for k, v in experiences.items()
      }
    else:
      transitions = experiences.copy()
    o_tf = tf.convert_to_tensor(transitions["o"], dtype=tf.float32)
    g_tf = tf.convert_to_tensor(transitions["g"], dtype=tf.float32)
    self._maf_o_norm.update(o_tf)
    self._maf_g_norm.update(g_tf)
    self._critic_o_norm.update(o_tf)
    self._critic_g_norm.update(g_tf)

  @tf.function
  def _train_offline_graph(self, o, g, u):
    with tf.name_scope('OfflineLosses/'):
      # Train maf
      maf_input = tf.concat(
          [self._maf_o_norm(o),
           self._maf_g_norm(g), u / self.max_u], axis=1)

      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self._maf.trainable_weights)
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
          tape2.watch(maf_input)
          logprob = tf.clip_by_value(self._maf(maf_input), -1e5, 1e5)
          logprob = tf.reshape(logprob, (-1, 1))
          neg_logprob = -tf.reduce_mean(logprob)
        jacobian = tape2.gradient(neg_logprob, maf_input)
        regularizer = tf.norm(jacobian, ord=2)
        maf_loss = (self.prm_loss_weight * neg_logprob +
                    self.reg_loss_weight * regularizer)
        tf.summary.scalar(name='maf_loss vs {}'.format(
            self.offline_training_step.name),
                          data=maf_loss,
                          step=self.offline_training_step)

      maf_grads = tape.gradient(maf_loss, self._maf.trainable_weights)
      self._maf_optimizer.apply_gradients(
          zip(maf_grads, self._maf.trainable_weights))

      # Distill maf to critic
      # add random data
      rand_o = self._critic_o_norm.denormalize(
          tf.random.uniform(tf.shape(o), -1, 1))
      rand_g = self._critic_g_norm.denormalize(
          tf.random.uniform(tf.shape(g), -1, 1))
      rand_u = tf.random.uniform(tf.shape(u), -1, 1) * self.max_u
      # axis 0 => batch dim
      o = tf.concat((o, rand_o), axis=0)
      g = tf.concat((g, rand_g), axis=0)
      u = tf.concat((u, rand_u), axis=0)

      maf_input = tf.concat(
          [self._maf_o_norm(o),
           self._maf_g_norm(g), u / self.max_u], axis=1)
      logprob = self._maf(maf_input)
      logprob = tf.clip_by_value(logprob, -1e5, 1e5)
      logprob = tf.reshape(logprob, (-1, 1))
      # logprob = tf.math.log(tf.exp(logprob) + tf.exp(self.min_logprob))
      logprob = self.logprob_scale * logprob

      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(self._critic.trainable_weights)
        critic_loss = self._huber_loss(
            tf.stop_gradient(logprob),
            self._critic([self._critic_o_norm(o),
                          self._critic_g_norm(g), u]))
        critic_loss = tf.reduce_mean(critic_loss)
        tf.summary.scalar(name='critic_loss vs {}'.format(
            self.offline_training_step.name),
                          data=critic_loss,
                          step=self.offline_training_step)
      critic_grads = tape.gradient(critic_loss, self._critic.trainable_weights)
      self._critic_optimizer.apply_gradients(
          zip(critic_grads, self._critic.trainable_weights))

    self.offline_training_step.assign_add(1)

  def train_offline(self):
    with tf.summary.record_if(lambda: self.offline_training_step % 200 == 0):
      batch = self.offline_buffer.sample(self.offline_batch_size)
      o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
      g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
      u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
      self._train_offline_graph(o_tf, g_tf, u_tf)

  def store_experiences(self, experiences):
    pass  # no online training

  def train_online(self):
    self.online_training_step.assign_add(1)
