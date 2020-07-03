import os
import pickle
osp = os.path

import numpy as np
import tensorflow as tf
tfk = tf.keras

from rlfd import logger, memory, normalizer, policies
from rlfd.td3 import td3_networks, shaping


class TD3(object):

  def __init__(
      self,
      batch_size,
      # exploration
      expl_gaussian_noise,
      expl_random_prob,
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
      pi_lr,
      action_l2,
      # td3 specific
      policy_freq,
      policy_noise,
      policy_noise_clip,
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
    self.expl_gaussian_noise = expl_gaussian_noise
    self.expl_random_prob = expl_random_prob

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

    # TD3 specific
    self.policy_freq = policy_freq
    self.policy_noise = policy_noise
    self.policy_noise_clip = policy_noise_clip

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
        get_action=lambda o, g: self._actor([o, g]),
        process_observation=process_observation)
    self.expl_policy = policies.GaussianEpsilonGreedyPolicy(
        self.dimo,
        self.dimg,
        self.dimu,
        get_action=lambda o, g: self._actor([o, g]),
        max_u=self.max_u,
        noise_eps=expl_gaussian_noise,
        random_prob=expl_random_prob,
        process_observation=process_observation)

    # Losses
    self._huber_loss = tfk.losses.Huber(delta=10.0,
                                        reduction=tfk.losses.Reduction.NONE)

    # Initialize training steps
    self.td3_training_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.exploration_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  def _increment_exploration_step(self, num_steps):
    self.exploration_step.assign_add(num_steps)
    self._policy_inspect_graph(summarize=True)

  def before_training_hook(self, data_dir=None, env=None):
    if self.demo_strategy != "none" or self.sample_demo_buffer:
      demo_file = osp.join(data_dir, "demo_data.npz")
      assert osp.isfile(demo_file), "Demo file does not exist."
      experiences = self.demo_buffer.load_from_file(data_file=demo_file)
      if self.sample_demo_buffer:
        self.update_stats(experiences)

    if self.shaping != None:
      self.train_shaping()

    if self.initialize_with_bc:
      self.train_bc()
      self.update_target_network(polyak=0.0)

  def store_experiences(self, experiences):

    with tf.summary.record_if(lambda: self.exploration_step % 200 == 0):
      self.replay_buffer.store(experiences)
      if self.use_n_step_return:
        self.n_step_replay_buffer.store(experiences)

      num_steps = np.prod(experiences["o"].shape[:-1])
      self._increment_exploration_step(num_steps)

  def clear_n_step_replay_buffer(self):
    if not self.use_n_step_return:
      return
    self.n_step_replay_buffer.clear_buffer()

  def save(self, path):
    """Pickles the current policy.
    """
    with open(path, "wb") as f:
      pickle.dump(self, f)

  def _merge_batch_experiences(self, batch1, batch2):
    assert batch1.keys() == batch2.keys()
    merged_batch = {}
    for k in batch1.keys():
      merged_batch[k] = np.concatenate((batch1[k], batch2[k]))

    return merged_batch

  def sample_batch(self):
    if self.use_n_step_return:
      one_step_batch = self.replay_buffer.sample(self.batch_size // 2)
      n_step_batch = self.n_step_replay_buffer.sample(self.batch_size -
                                                      self.batch_size // 2)
      batch = self._merge_batch_experiences(one_step_batch, n_step_batch)
    else:
      batch = self.replay_buffer.sample(self.batch_size)

    if self.sample_demo_buffer:
      rollout_batch = batch
      demo_batch = self.demo_buffer.sample(self.batch_size_demo)
      batch = self._merge_batch_experiences(rollout_batch, demo_batch)
    return batch

  @tf.function
  def _train_bc_graph(self, o, g, u):
    o = self._o_stats.normalize(o)
    g = self._g_stats.normalize(g)
    with tf.GradientTape() as tape:
      pi = self._actor([o, g])
      bc_loss = tf.reduce_mean(tf.square(pi - u))
    actor_grads = tape.gradient(bc_loss, self._actor.trainable_weights)
    self._bc_optimizer.apply_gradients(
        zip(actor_grads, self._actor.trainable_weights))
    return bc_loss

  def train_bc(self):
    if not self.initialize_with_bc:
      return

    for epoch in range(self.initialize_num_epochs):
      demo_data_iter = self.demo_buffer.sample(batch_size=self.batch_size_demo,
                                               shuffle=True,
                                               return_iterator=True,
                                               include_partial_batch=True)
      for batch in demo_data_iter:
        o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
        g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
        u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
        bc_loss_tf = self._train_bc_graph(o_tf, g_tf, u_tf)
        bc_loss = bc_loss_tf.numpy()

      if epoch % (self.initialize_num_epochs / 100) == (
          self.initialize_num_epochs / 100 - 1):
        logger.info("epoch: {} policy initialization loss: {}".format(
            epoch, bc_loss))

  def train_shaping(self):

    demo_data_iter = self.demo_buffer.sample(return_iterator=True)
    demo_data = next(demo_data_iter)

    self.shaping.train(demo_data)
    self.shaping.evaluate(demo_data)

  def _criticq_loss_graph(self, o, g, o_2, g_2, u, r, n, done):
    # Normalize observations
    norm_o = self._o_stats.normalize(o)
    norm_g = self._g_stats.normalize(g)
    norm_o_2 = self._o_stats.normalize(o_2)
    norm_g_2 = self._g_stats.normalize(g_2)

    # add noise to target policy output
    noise = tf.random.normal(tf.shape(u), 0.0, self.policy_noise)
    noise = tf.clip_by_value(noise, -self.policy_noise_clip,
                             self.policy_noise_clip) * self.max_u
    u_2 = tf.clip_by_value(
        self._actor_target([norm_o_2, norm_g_2]) + noise, -self.max_u,
        self.max_u)

    # immediate reward
    target_q = r
    # shaping reward
    if self.shaping != None:
      potential_curr = self.potential_weight * self.shaping.potential(
          o=o, g=g, u=u)
      potential_next = self.potential_weight * self.shaping.potential(
          o=o_2, g=g_2, u=u_2)
      target_q += (1.0 - done) * tf.pow(self.gamma,
                                        n) * potential_next - potential_curr
    # td3 target_q with clipping
    target_q += (1.0 - done) * tf.pow(self.gamma, n) * tf.minimum(
        self._criticq1_target([norm_o_2, norm_g_2, u_2]),
        self._criticq2_target([norm_o_2, norm_g_2, u_2]),
    )
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

    with tf.name_scope('TD3Losses/'):
      tf.summary.scalar(name='criticq_loss vs td3_training_step',
                        data=criticq_loss,
                        step=self.td3_training_step)

    return criticq_loss

  def _actor_loss_graph(self, o, g, u):
    # Normalize observations
    norm_o = self._o_stats.normalize(o)
    norm_g = self._g_stats.normalize(g)

    pi = self._actor([norm_o, norm_g])
    actor_loss = -tf.reduce_mean(self._criticq1([norm_o, norm_g, pi]))
    actor_loss += self.action_l2 * tf.reduce_mean(tf.square(pi / self.max_u))
    if self.shaping != None:
      actor_loss += -tf.reduce_mean(
          self.potential_weight * self.shaping.potential(o=o, g=g, u=pi))
    if self.demo_strategy == "bc":
      assert self.sample_demo_buffer, "must sample from the demonstration buffer to use behavior cloning"
      mask = np.concatenate(
          (np.zeros(self.batch_size), np.ones(self.batch_size_demo)), axis=0)
      demo_pi = tf.boolean_mask((pi), mask)
      demo_u = tf.boolean_mask((u), mask)
      if self.bc_params["q_filter"]:
        q_u = self._criticq1([norm_o, norm_g, u])
        q_pi = self._criticq1([norm_o, norm_g, pi])
        q_filter_mask = tf.reshape(tf.boolean_mask(q_u > q_pi, mask), [-1])
        bc_loss = tf.reduce_mean(
            tf.square(
                tf.boolean_mask(demo_pi, q_filter_mask, axis=0) -
                tf.boolean_mask(demo_u, q_filter_mask, axis=0)))
      else:
        bc_loss = tf.reduce_mean(tf.square(demo_pi - demo_u))
      actor_loss = (self.bc_params["prm_loss_weight"] * actor_loss +
                    self.bc_params["aux_loss_weight"] * bc_loss)

    with tf.name_scope('TD3Losses/'):
      tf.summary.scalar(name='actor_loss vs td3_training_step',
                        data=actor_loss,
                        step=self.td3_training_step)

    return actor_loss

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

    # Train actor
    if self.td3_training_step % self.policy_freq == 0:
      actor_trainable_weights = self._actor.trainable_weights
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(actor_trainable_weights)
        actor_loss = self._actor_loss_graph(o, g, u)
      actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
      self._actor_optimizer.apply_gradients(
          zip(actor_grads, actor_trainable_weights))

    self.td3_training_step.assign_add(1)

  def train(self):
    with tf.summary.record_if(lambda: self.td3_training_step % 200 == 0):

      batch = self.sample_batch()

      o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
      g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
      o_2_tf = tf.convert_to_tensor(batch["o_2"], dtype=tf.float32)
      g_2_tf = tf.convert_to_tensor(batch["g_2"], dtype=tf.float32)
      u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
      r_tf = tf.convert_to_tensor(batch["r"], dtype=tf.float32)
      n_tf = tf.convert_to_tensor(batch["n"], dtype=tf.float32)
      done_tf = tf.convert_to_tensor(batch["done"], dtype=tf.float32)

      self._train_graph(o_tf, g_tf, o_2_tf, g_2_tf, u_tf, r_tf, n_tf, done_tf)

  def update_potential_weight(self):
    if self.potential_decay_epoch < self.shaping_params["potential_decay_epoch"]:
      self.potential_decay_epoch += 1
      return self.potential_weight.numpy()
    potential_weight_tf = self.potential_weight.assign(
        self.potential_weight * self.potential_decay_scale)
    return potential_weight_tf.numpy()

  def update_target_network(self, polyak=None):
    polyak = polyak if polyak else self.polyak
    copy_func = lambda v: v[0].assign(polyak * v[0] + (1.0 - polyak) * v[1])
    list(map(copy_func, zip(self._actor_target.weights, self._actor.weights)))
    list(
        map(copy_func, zip(self._criticq1_target.weights,
                           self._criticq1.weights)))
    list(
        map(copy_func, zip(self._criticq2_target.weights,
                           self._criticq2.weights)))

  def logs(self, prefix=""):
    logs = []
    logs.append(
        (prefix + "stats_o/mean", np.mean(self._o_stats.mean_tf.numpy())))
    logs.append((prefix + "stats_o/std", np.mean(self._o_stats.std_tf.numpy())))
    logs.append(
        (prefix + "stats_g/mean", np.mean(self._g_stats.mean_tf.numpy())))
    logs.append((prefix + "stats_g/std", np.mean(self._g_stats.std_tf.numpy())))
    return logs

  def update_stats(self, experiences):
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
    self._o_stats.update(o_tf)
    self._g_stats.update(g_tf)

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
      self.replay_buffer = memory.UniformReplayBuffer(buffer_shapes,
                                                      self.buffer_size,
                                                      self.eps_length)
      if self.use_n_step_return:
        self.n_step_replay_buffer = memory.MultiStepReplayBuffer(
            buffer_shapes, self.buffer_size, self.eps_length,
            self.n_step_return_steps, self.gamma)
      if self.demo_strategy != "none" or self.sample_demo_buffer:
        self.demo_buffer = memory.UniformReplayBuffer(buffer_shapes,
                                                      self.buffer_size,
                                                      self.eps_length)
    else:
      self.replay_buffer = memory.RingReplayBuffer(buffer_shapes,
                                                   self.buffer_size)
      assert not self.use_n_step_return, "not implemented yet"
      if self.demo_strategy != "none" or self.sample_demo_buffer:
        self.demo_buffer = memory.RingReplayBuffer(buffer_shapes,
                                                   self.buffer_size)

  def _create_network(self):
    # Normalizer for goal and observation.
    self._o_stats = normalizer.Normalizer(self.dimo, self.norm_eps,
                                          self.norm_clip)
    self._g_stats = normalizer.Normalizer(self.dimg, self.norm_eps,
                                          self.norm_clip)
    # Models
    self._actor = td3_networks.Actor(self.dimo, self.dimg, self.dimu,
                                     self.max_u, self.layer_sizes)
    self._actor_target = td3_networks.Actor(self.dimo, self.dimg, self.dimu,
                                            self.max_u, self.layer_sizes)
    self._criticq1 = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                         self.max_u, self.layer_sizes)
    self._criticq1_target = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                                self.max_u, self.layer_sizes)
    self._criticq2 = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                         self.max_u, self.layer_sizes)
    self._criticq2_target = td3_networks.Critic(self.dimo, self.dimg, self.dimu,
                                                self.max_u, self.layer_sizes)
    self.update_target_network(polyak=0.0)
    # Optimizers
    self._actor_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)
    self._criticq_optimizer = tfk.optimizers.Adam(learning_rate=self.q_lr)
    self._bc_optimizer = tfk.optimizers.Adam(learning_rate=self.pi_lr)

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

  @tf.function
  def _policy_inspect_graph(self, o=None, g=None, summarize=False):
    # should only happen for in the first call of this function
    if not hasattr(self, "_policy_inspect_count"):
      self._policy_inspect_count = tf.Variable(0.0, trainable=False)
      self._policy_inspect_estimate_q = tf.Variable(0.0, trainable=False)
      if self.shaping != None:
        self._policy_inspect_potential = tf.Variable(0.0, trainable=False)

    if summarize:
      with tf.name_scope('PolicyInspect'):
        if self.shaping == None:
          mean_estimate_q = (self._policy_inspect_estimate_q /
                             self._policy_inspect_count)
          success = tf.summary.scalar(
              name='mean_estimate_q vs exploration_step',
              data=mean_estimate_q,
              step=self.exploration_step)
        else:
          mean_potential = (self._policy_inspect_potential /
                            self._policy_inspect_count)
          tf.summary.scalar(name='mean_potential vs exploration_step',
                            data=mean_potential,
                            step=self.exploration_step)
          mean_estimate_q = (
              self._policy_inspect_estimate_q +
              self._policy_inspect_potential) / self._policy_inspect_count
          success = tf.summary.scalar(
              name='mean_estimate_q vs exploration_step',
              data=mean_estimate_q,
              step=self.exploration_step)
      if success:
        self._policy_inspect_count.assign(0.0)
        self._policy_inspect_estimate_q.assign(0.0)
        if self.shaping != None:
          self._policy_inspect_potential.assign(0.0)
      return

    assert o != None and g != None, "Provide the same arguments passed to get action."

    norm_o = self._o_stats.normalize(o)
    norm_g = self._g_stats.normalize(g)
    u = self._actor([norm_o, norm_g])

    self._policy_inspect_count.assign_add(1)
    q = self._criticq1([norm_o, norm_g, u])
    self._policy_inspect_estimate_q.assign_add(tf.reduce_sum(q))
    if self.shaping != None:
      p = self.potential_weight * self.shaping.potential(o=o, g=g, u=u)
      self._policy_inspect_potential.assign_add(tf.reduce_sum(p))

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
        "actor_target": self._actor_target.get_weights(),
        "criticq1": self._criticq1.get_weights(),
        "criticq1_target": self._criticq1_target.get_weights(),
        "criticq2": self._criticq2.get_weights(),
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
    self._actor_target.set_weights(stored_vars["actor_target"])
    self._criticq1.set_weights(stored_vars["criticq1"])
    self._criticq1_target.set_weights(stored_vars["criticq1_target"])
    self._criticq2.set_weights(stored_vars["criticq2"])
    self._criticq2_target.set_weights(stored_vars["criticq2_target"])
    self.shaping = shaping
