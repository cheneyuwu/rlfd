import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
tfk = tf.keras

from rlfd import memory, normalizer, policies
from rlfd.agents import agent, td3_networks


class TD3(agent.Agent):

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
      # exploration
      expl_gaussian_noise,
      expl_random_prob,
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
      # td3 specific
      policy_freq,
      policy_noise,
      policy_noise_clip,
      # double q
      soft_target_tau,
      target_update_freq,
      # online training plus offline data
      use_pretrained_actor,
      use_pretrained_critic,
      online_data_strategy,
      # online bc regularizer
      bc_params,
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
    self.gamma = gamma

    self.online_batch_size = online_batch_size
    self.offline_batch_size = offline_batch_size

    self.buffer_size = buffer_size

    self.expl_gaussian_noise = expl_gaussian_noise
    self.expl_random_prob = expl_random_prob

    self.policy_freq = policy_freq
    self.policy_noise = policy_noise
    self.policy_noise_clip = policy_noise_clip

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
    self.online_data_strategy = online_data_strategy
    assert self.online_data_strategy in ["None", "BC", "Shaping"]
    self.bc_params = bc_params
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

    self._actor = td3_networks.Actor(self.dimo, self.dimg, self.dimu,
                                     self.max_u, self.layer_sizes)
    self._actor_target = td3_networks.Actor(self.dimo, self.dimg, self.dimu,
                                            self.max_u, self.layer_sizes)

    self._actor_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)

    self._copy_weights(self._actor, self._actor_target, 1.0)

    self._actor_models = {
        "actor_o_norm": self._actor_o_norm,
        "actor_g_norm": self._actor_g_norm,
        "actor": self._actor,
    }
    self.save_model(self._actor_models)

  def _initialize_critic(self):
    self._critic_o_norm = normalizer.Normalizer(self.dimo, self.norm_eps,
                                                self.norm_clip)
    self._critic_g_norm = normalizer.Normalizer(self.dimg, self.norm_eps,
                                                self.norm_clip)

    self._criticq1 = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                         self.max_u, self.layer_sizes)
    self._criticq1_target = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                                self.max_u, self.layer_sizes)
    self._copy_weights(self._criticq1, self._criticq1_target, 1.0)

    self._criticq2 = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                         self.max_u, self.layer_sizes)
    self._criticq2_target = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                                self.max_u, self.layer_sizes)
    self._copy_weights(self._criticq2, self._criticq2_target, 1.0)

    self._criticq_optimizer = tfk.optimizers.Adam(learning_rate=self.q_lr)

    self._critic_models = {
        "critic_o_norm": self._critic_o_norm,
        "critic_g_norm": self._critic_g_norm,
        "criticq1": self._criticq1,
        "criticq2": self._criticq2,
        "criticq1_target": self._criticq1_target,
        "criticq2_target": self._criticq2_target,
    }
    self.save_model(self._critic_models)

  def _create_model(self):
    self._initialize_actor()
    self._initialize_critic()
    # Losses
    self._huber_loss = tfk.losses.Huber(delta=10.0,
                                        reduction=tfk.losses.Reduction.NONE)

    # Generate policies
    def process_observation_expl(o, g):
      norm_o = self._actor_o_norm(o)
      norm_g = self._actor_g_norm(g)
      return norm_o, norm_g

    self._expl_policy = policies.GaussianEpsilonGreedyPolicy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g]),
        max_u=self.max_u,
        noise_eps=self.expl_gaussian_noise,
        random_prob=self.expl_random_prob,
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
        get_action=lambda o, g: self._actor([o, g]),
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

  @tf.function
  def estimate_q_graph(self, o, g, u):
    """A convenient function for shaping"""
    return self._criticq1([self._critic_o_norm(o), self._critic_g_norm(g), u])

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

    self.shaping = shaping
    self.pretrained_agent = pretrained_agent

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
    self._critic_o_norm.update(o_tf)
    self._critic_g_norm.update(g_tf)

  def train_offline(self):
    # No offline training.
    self.offline_training_step.assign_add(1)

  def before_online_hook(self):
    if self.use_pretrained_actor:
      for k, v in self._actor_models.items():
        self._copy_weights(self.pretrained_agent.get_saved_model(k), v, 1.0)

    if self.use_pretrained_critic:
      for k, v in self._critic_models.items():
        self._copy_weights(self.pretrained_agent.get_saved_model(k), v, 1.0)

  def store_experiences(self, experiences):
    self.online_buffer.store(experiences)
    if self.norm_obs_online:
      self._update_stats(experiences)

  def _merge_batch_experiences(self, batch1, batch2):
    """Helper function to merge experiences from offline and online data."""
    assert batch1.keys() == batch2.keys()
    merged_batch = {}
    for k in batch1.keys():
      merged_batch[k] = np.concatenate((batch1[k], batch2[k]))

    return merged_batch

  def sample_batch(self):
    if self.online_data_strategy == "None":
      batch = self.online_buffer.sample(self.online_batch_size)
    else:
      online_batch = self.online_buffer.sample(self.online_batch_size -
                                               self.offline_batch_size)
      offline_batch = self.offline_buffer.sample(self.offline_batch_size)
      batch = self._merge_batch_experiences(online_batch, offline_batch)
    return batch

  def _td3_criticq_loss_graph(self, o, g, o_2, g_2, u, r, n, done, step):
    # Add noise to target policy output
    noise = tf.random.normal(tf.shape(u), 0.0, self.policy_noise)
    noise = tf.clip_by_value(noise, -self.policy_noise_clip,
                             self.policy_noise_clip) * self.max_u
    u_2 = tf.clip_by_value(
        self._actor_target([self._actor_o_norm(o_2),
                            self._actor_g_norm(g_2)]) + noise, -self.max_u,
        self.max_u)

    # Immediate reward
    target_q = r
    # Shaping reward
    if self.online_data_strategy == "Shaping":
      potential_curr = self.shaping.potential(o=o, g=g, u=u)
      potential_next = self.shaping.potential(o=o_2, g=g_2, u=u_2)
      target_q += (1.0 - done) * tf.pow(self.gamma,
                                        n) * potential_next - potential_curr
    # Q value from next state
    target_next_q1 = self._criticq1_target(
        [self._critic_o_norm(o_2),
         self._critic_g_norm(g_2), u_2])
    target_next_q2 = self._criticq2_target(
        [self._critic_o_norm(o_2),
         self._critic_g_norm(g_2), u_2])
    target_next_min_q = tf.minimum(target_next_q1, target_next_q2)
    target_q += (1.0 - done) * tf.pow(self.gamma, n) * target_next_min_q
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

    criticq_loss = tf.reduce_mean(td_loss)
    tf.summary.scalar(name='criticq_loss vs {}'.format(step.name),
                      data=criticq_loss,
                      step=step)
    return criticq_loss

  def _td3_actor_loss_graph(self, o, g, u, step):
    pi = self._actor([self._actor_o_norm(o), self._actor_g_norm(g)])
    actor_loss = -tf.reduce_mean(
        self._criticq1([self._critic_o_norm(o),
                        self._critic_g_norm(g), pi]))
    actor_loss += self.action_l2 * tf.reduce_mean(tf.square(pi / self.max_u))
    if self.online_data_strategy == "Shaping":
      actor_loss += -tf.reduce_mean(self.shaping.potential(o=o, g=g, u=pi))
    if self.online_data_strategy == "BC":
      mask = np.concatenate(
          (np.zeros(self.online_batch_size - self.offline_batch_size),
           np.ones(self.offline_batch_size)),
          axis=0)
      demo_pi = tf.boolean_mask((pi), mask)
      demo_u = tf.boolean_mask((u), mask)
      if self.bc_params["q_filter"]:
        q_u = self._criticq1(
            [self._critic_o_norm(o),
             self._critic_g_norm(g), u])
        q_pi = self._criticq1(
            [self._critic_o_norm(o),
             self._critic_g_norm(g), pi])
        q_filter_mask = tf.reshape(tf.boolean_mask(q_u > q_pi, mask), [-1])
        bc_loss = tf.reduce_mean(
            tf.square(
                tf.boolean_mask(demo_pi, q_filter_mask, axis=0) -
                tf.boolean_mask(demo_u, q_filter_mask, axis=0)))
      else:
        bc_loss = tf.reduce_mean(tf.square(demo_pi - demo_u))
      actor_loss = (self.bc_params["prm_loss_weight"] * actor_loss +
                    self.bc_params["aux_loss_weight"] * bc_loss)
    tf.summary.scalar(name='actor_loss vs {}'.format(step.name),
                      data=actor_loss,
                      step=step)
    return actor_loss

  @tf.function
  def _train_online_graph(self, o, g, o_2, g_2, u, r, n, done):
    # Train critic q
    criticq_trainable_weights = (self._criticq1.trainable_weights +
                                 self._criticq2.trainable_weights)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(criticq_trainable_weights)
      with tf.name_scope('OnlineLosses/'):
        criticq_loss = self._td3_criticq_loss_graph(o, g, o_2, g_2, u, r, n,
                                                    done,
                                                    self.online_training_step)
    criticq_grads = tape.gradient(criticq_loss, criticq_trainable_weights)
    self._criticq_optimizer.apply_gradients(
        zip(criticq_grads, criticq_trainable_weights))

    # Train actor
    if self.online_training_step % self.policy_freq == 0:
      actor_trainable_weights = self._actor.trainable_weights
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(actor_trainable_weights)
        with tf.name_scope('OnlineLosses/'):
          actor_loss = self._td3_actor_loss_graph(o, g, u,
                                                  self.online_training_step)
      actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
      self._actor_optimizer.apply_gradients(
          zip(actor_grads, actor_trainable_weights))

    self.online_training_step.assign_add(1)

  def train_online(self):
    with tf.summary.record_if(lambda: self.online_training_step % 200 == 0):

      batch = self.sample_batch()

      o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
      g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
      o_2_tf = tf.convert_to_tensor(batch["o_2"], dtype=tf.float32)
      g_2_tf = tf.convert_to_tensor(batch["g_2"], dtype=tf.float32)
      u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
      r_tf = tf.convert_to_tensor(batch["r"], dtype=tf.float32)
      n_tf = tf.convert_to_tensor(batch["n"], dtype=tf.float32)
      done_tf = tf.convert_to_tensor(batch["done"], dtype=tf.float32)

      self._train_online_graph(o_tf, g_tf, o_2_tf, g_2_tf, u_tf, r_tf, n_tf,
                               done_tf)
      if self.online_training_step % self.target_update_freq == 0:
        self._copy_weights(self._actor, self._actor_target)
        self._copy_weights(self._criticq1, self._criticq1_target)
        self._copy_weights(self._criticq2, self._criticq2_target)

  def _copy_weights(self, source, target, soft_target_tau=None):
    soft_target_tau = (soft_target_tau
                       if soft_target_tau else self.soft_target_tau)
    copy_func = lambda v: v[1].assign(
        (1.0 - soft_target_tau) * v[1] + soft_target_tau * v[0])
    list(map(copy_func, zip(source.weights, target.weights)))

  @tf.function
  def _policy_inspect_graph(self, o, g):
    # should only happen for in the first call of this function
    if not hasattr(self, "_total_policy_inspect_count"):
      self._total_policy_inspect_count = tf.Variable(0,
                                                     trainable=False,
                                                     dtype=tf.int64)
      self._policy_inspect_count = tf.Variable(0.0, trainable=False)
      self._policy_inspect_estimate_q = tf.Variable(0.0, trainable=False)
      if self.shaping != None:
        self._policy_inspect_potential = tf.Variable(0.0, trainable=False)

    u = self._actor([self._actor_o_norm(o), self._actor_g_norm(g)])

    self._total_policy_inspect_count.assign_add(1)
    self._policy_inspect_count.assign_add(1)
    q = self._criticq1([self._critic_o_norm(o), self._critic_g_norm(g), u])
    self._policy_inspect_estimate_q.assign_add(tf.reduce_sum(q))
    if self.shaping != None:
      p = self.shaping.potential(o=o, g=g, u=u)
      self._policy_inspect_potential.assign_add(tf.reduce_sum(p))

    if self._total_policy_inspect_count % 5000 == 0:
      with tf.name_scope('PolicyInspect'):
        if self.shaping == None:
          mean_estimate_q = (self._policy_inspect_estimate_q /
                             self._policy_inspect_count)
          success = tf.summary.scalar(name='mean_estimate_q vs evaluation_step',
                                      data=mean_estimate_q,
                                      step=self._total_policy_inspect_count)
        else:
          mean_potential = (self._policy_inspect_potential /
                            self._policy_inspect_count)
          tf.summary.scalar(name='mean_potential vs evaluation_step',
                            data=mean_potential,
                            step=self._total_policy_inspect_count)
          mean_estimate_q = (
              self._policy_inspect_estimate_q +
              self._policy_inspect_potential) / self._policy_inspect_count
          success = tf.summary.scalar(name='mean_estimate_q vs evaluation_step',
                                      data=mean_estimate_q,
                                      step=self._total_policy_inspect_count)
      if success:
        self._policy_inspect_count.assign(0.0)
        self._policy_inspect_estimate_q.assign(0.0)
        if self.shaping != None:
          self._policy_inspect_potential.assign(0.0)

  def __getstate__(self):
    """For pickle. Store weights"""
    state = super().__getstate__()
    state["shaping"] = self.shaping
    return state

  def __setstate__(self, state):
    """For pickle. Re-instantiate the class, load weights"""
    shaping = state.pop("shaping")
    super().__setstate__(state)
    self.shaping = shaping
