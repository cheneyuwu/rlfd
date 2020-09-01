import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
tfk = tf.keras
tfl = tfk.layers

from rlfd import memory, normalizer, policies
from rlfd.agents import agent, sac_networks


class Generator(tfk.Model):

  def __init__(self, dimo, dimu, max_u, latent_dim, layer_sizes, name="g"):
    super().__init__(name=name)
    self._dimo = dimo
    self._dimu = dimu
    self._max_u = max_u

    self._mlp_layers = []
    for size in layer_sizes[:-1]:
      layer = tfl.Dense(units=size,
                        activation="relu",
                        kernel_initializer="glorot_normal")
      self._mlp_layers.append(layer)
    self._obs_output_layer = tfl.Dense(units=self._dimo[0],
                                       kernel_initializer="glorot_normal")
    self._act_output_layer = tfl.Dense(units=self._dimu[0],
                                       kernel_initializer="glorot_normal")
    # Create weights
    self(tf.zeros([0, latent_dim]))

  @tf.function
  def call(self, inputs):
    res = inputs
    for l in self._mlp_layers:
      res = l(res)
    obs = self._obs_output_layer(res)
    act = self._act_output_layer(res)
    return obs, act * self._max_u


class GAN(agent.Agent):
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
      # networks
      layer_sizes,
      latent_dim,
      gp_lambda,
      critic_freq,
      # replay buffer
      buffer_size,
      info):
    super().__init__(locals())

    self.dims = dims
    self.dimo = self.dims["o"]
    self.dimu = self.dims["u"]
    self.max_u = max_u
    self.fix_T = fix_T
    self.eps_length = eps_length

    self.offline_batch_size = offline_batch_size

    self.buffer_size = buffer_size

    self.layer_sizes = layer_sizes
    self.latent_dim = latent_dim
    self.gp_lambda = gp_lambda
    self.critic_freq = critic_freq

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

  def _initialize_generator(self):
    self._generator = Generator(self.dimo, self.dimu, self.max_u,
                                self.latent_dim, self.layer_sizes)
    self._generator_optimizer = tfk.optimizers.Adam(learning_rate=1e-4,
                                                    beta_1=0.5,
                                                    beta_2=0.9)

    self._generator_models = {
        "generator": self._generator,
    }
    self.save_model(self._generator_models)

  def _initialize_discriminator(self):
    self._discriminator_o_norm = normalizer.Normalizer(self.dimo, self.norm_eps,
                                                       self.norm_clip)
    self._discriminator = sac_networks.CriticQ(self.dimo, self.dimu, self.max_u,
                                               self.layer_sizes)
    self._discriminator_optimizer = tfk.optimizers.Adam(learning_rate=1e-4,
                                                        beta_1=0.5,
                                                        beta_2=0.9)

    self._critic_models = {  # discriminator => critic because we use it as pretrained critic
        "critic_o_norm": self._discriminator_o_norm,
        "criticq1": self._discriminator,
        "criticq2": self._discriminator,
        "criticq1_target": self._discriminator,
        "criticq2_target": self._discriminator,
    }
    self.save_model(self._critic_models)

  def _create_model(self):
    self._initialize_generator()
    self._initialize_discriminator()

    self._expl_policy = self._eval_policy = policies.RandomPolicy(
        self.dimo, self.dimu, self.max_u)

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
  def estimate_q_graph(self, o, u):
    """A convenient function for shaping"""
    return self._discriminator([self._discriminator_o_norm(o), u])

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
    self._discriminator_o_norm.update(o_tf)

  @tf.function
  def _train_offline_graph(self, o, u):
    with tf.name_scope('OfflineLosses/'):

      o = self._discriminator_o_norm(o)

      with tf.GradientTape(watch_accessed_variables=False,
                           persistent=True) as tape:
        tape.watch([
            self._discriminator.trainable_weights,
            self._generator.trainable_weights
        ])
        fo, fu = self._generator(
            tf.random.uniform(
                tf.concat((tf.shape(o)[:-1], [self.latent_dim]), axis=0)))

        disc_fake = self._discriminator([fo, fu])
        disc_real = self._discriminator([o, u])
        # Discriminator loss on generator (including gradient penalty)
        alpha = tf.random.uniform(shape=tf.concat((tf.shape(o)[:-1], [1]),
                                                  axis=0),
                                  minval=0.0,
                                  maxval=1.0)  # assumes 1 dim state+action
        interpolates = list(
            map(lambda v: alpha * v[0] + (1.0 - alpha) * v[1],
                zip([o, u], [fo, fu])))
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
          tape2.watch(interpolates)
          disc_interpolates = self._discriminator(interpolates)

        gradients = tape2.gradient(disc_interpolates, interpolates)
        slopes = tf.sqrt(
            tf.reduce_sum(tf.square(tf.concat(gradients, axis=-1)), axis=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1)**2)
        disc_loss = (tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) +
                     self.gp_lambda * gradient_penalty)
        tf.summary.scalar(name='discriminator_loss vs {}'.format(
            self.offline_training_step.name),
                          data=disc_loss,
                          step=self.offline_training_step)
        # Generator loss
        gen_loss = -tf.reduce_mean(disc_fake)
        tf.summary.scalar(name='generator_loss vs {}'.format(
            self.offline_training_step.name),
                          data=gen_loss,
                          step=self.offline_training_step)
    disc_grads = tape.gradient(disc_loss, self._discriminator.trainable_weights)
    gen_grads = tape.gradient(gen_loss, self._generator.trainable_weights)

    self._discriminator_optimizer.apply_gradients(
        zip(disc_grads, self._discriminator.trainable_weights))
    if self.offline_training_step % self.critic_freq == 0:
      self._generator_optimizer.apply_gradients(
          zip(gen_grads, self._generator.trainable_weights))

    self.offline_training_step.assign_add(1)

  def train_offline(self):
    with tf.summary.record_if(lambda: self.offline_training_step % 200 == 0):
      batch = self.offline_buffer.sample(self.offline_batch_size)
      o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
      u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
      self._train_offline_graph(o_tf, u_tf)

  def store_experiences(self, experiences):
    pass  # no online training

  def train_online(self):
    self.online_training_step.assign_add(1)
