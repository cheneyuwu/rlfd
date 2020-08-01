import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
tfk = tf.keras

from rlfd import logger, memory, normalizer, policies
from rlfd.agents import agent, sac_networks


class BC(agent.Agent):
  """Not tested. Behavior cloning.
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
      pi_lr,
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

    self.layer_sizes = layer_sizes
    self.pi_lr = pi_lr

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

  def _initialize_actor(self):
    self._actor_o_norm = normalizer.Normalizer(self.dimo, self.norm_eps,
                                               self.norm_clip)
    self._actor_g_norm = normalizer.Normalizer(self.dimg, self.norm_eps,
                                               self.norm_clip)

    self._actor = sac_networks.Actor(self.dimo, self.dimg, self.dimu,
                                     self.max_u, self.layer_sizes)
    self._actor_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)

    self._actor_models = {
        "actor_o_norm": self._actor_o_norm,
        "actor_g_norm": self._actor_g_norm,
        "actor": self._actor,
    }
    self.save_model(self._actor_models)

  def _create_model(self):
    self._initialize_actor()

    # Generate policies
    def process_observation_expl(o, g):
      norm_o = self._actor_o_norm(o)
      norm_g = self._actor_g_norm(g)
      return norm_o, norm_g

    self._expl_policy = policies.Policy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g], sample=True)[0],
        process_observation=process_observation_expl)

    def process_observation_eval(o, g):
      norm_o = self._actor_o_norm(o)
      norm_g = self._actor_g_norm(g)
      self._policy_inspect_graph(o, g)
      return norm_o, norm_g

    self._eval_policy = policies.Policy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g], sample=False)[0],
        process_observation=process_observation_eval)

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

  def before_training_hook(self,
                           data_dir=None,
                           env=None,
                           shaping=None,
                           pretrained_agent=None):
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
    self._actor_o_norm.update(o_tf)
    self._actor_g_norm.update(g_tf)

  @tf.function
  def _train_offline_graph(self, o, g, u):
    with tf.GradientTape() as tape:
      # pi if using td3_network actor.
      pi, logprob_pi = self._actor(
          [self._actor_o_norm(o), self._actor_g_norm(g)])
      bc_loss = tf.reduce_mean(tf.square(pi - u))
    actor_grads = tape.gradient(bc_loss, self._actor.trainable_weights)
    self._actor_optimizer.apply_gradients(
        zip(actor_grads, self._actor.trainable_weights))

    with tf.name_scope('OfflineLosses/'):
      tf.summary.scalar(name='bc_loss vs offline_training_step',
                        data=bc_loss,
                        step=self.offline_training_step)

    self.offline_training_step.assign_add(1)

  def train_offline(self):
    with tf.summary.record_if(lambda: self.offline_training_step % 200 == 0):
      batch = self.offline_buffer.sample(self.offline_batch_size)
      o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
      g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
      u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
      self._train_offline_graph(o_tf, g_tf, u_tf)

  def store_experiences(self, experiences):
    pass

  def train_online(self):
    # No online training
    self.offline_training_step.assign_add(1)