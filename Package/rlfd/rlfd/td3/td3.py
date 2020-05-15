import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rlfd import logger
from rlfd.td3.actorcritic_network import Actor, Critic
from rlfd.td3.shaping_v2 import EnsembleShaping as EnsembleShapingV2
from rlfd.td3.shaping import EnsembleShaping, NFShaping, GANShaping
from rlfd.td3.normalizer import Normalizer
from rlfd.memory import RingReplayBuffer, UniformReplayBuffer, MultiStepReplayBuffer, iterbatches

tfd = tfp.distributions


class TD3(object):

  def __init__(
      self,
      # for learning
      random_exploration_cycles,
      num_epochs,
      num_cycles,
      num_batches,
      batch_size,
      # environment configuration
      dims,
      max_u,
      gamma,
      eps_length,
      fix_T,
      norm_eps,
      norm_clip,
      # networks
      scope,
      layer_sizes,
      q_lr,
      pi_lr,
      action_l2,
      # td3
      twin_delayed,
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
      info,
  ):
    # Store initial args passed into the function
    self.init_args = locals()

    self.random_exploration_cycles = random_exploration_cycles
    self.num_epochs = num_epochs
    self.num_cycles = num_cycles
    self.num_batches = num_batches
    self.buffer_size = buffer_size
    self.batch_size = batch_size

    self.use_demo_reward = use_demo_reward
    self.sample_demo_buffer = sample_demo_buffer
    self.batch_size_demo = batch_size_demo
    self.initialize_with_bc = initialize_with_bc
    self.initialize_num_epochs = initialize_num_epochs

    self.eps_length = eps_length
    self.fix_T = fix_T

    # Parameters
    self.dims = dims
    self.max_u = max_u
    self.q_lr = q_lr
    self.pi_lr = pi_lr
    self.action_l2 = action_l2
    self.polyak = polyak

    self.norm_eps = norm_eps
    self.norm_clip = norm_clip

    self.layer_sizes = layer_sizes
    self.twin_delayed = twin_delayed
    self.policy_freq = policy_freq
    self.policy_noise = policy_noise
    self.policy_noise_clip = policy_noise_clip

    # multistep return
    self.use_n_step_return = use_n_step_return
    self.n_step_return_steps = eps_length // 5

    # play with demonstrations
    self.demo_strategy = demo_strategy
    assert self.demo_strategy in ["none", "bc", "gan", "nf"]
    self.bc_params = bc_params
    self.shaping_params = shaping_params
    self.gamma = gamma
    self.info = info

    # Prepare parameters
    self.dimo = self.dims["o"]
    self.dimg = self.dims["g"]
    self.dimu = self.dims["u"]

    self._create_memory()
    self._create_network()

    # Initialize training steps
    self.td3_training_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.model_training_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.exploration_step = tf.Variable(0, trainable=False, dtype=tf.int64)

  def _increment_exploration_step(self, num_steps):
    self.exploration_step.assign_add(num_steps)
    self._policy_inspect(summarize=True)

  @tf.function
  def get_actions_graph(self, o, g):

    norm_o = self.o_stats.normalize(o)
    norm_g = self.g_stats.normalize(g)

    u = self.main_actor([norm_o, norm_g])

    self._policy_inspect(o, g)

    return u

  def get_actions(self, o, g):
    o = o.reshape((-1, *self.dimo))
    g = g.reshape((o.shape[0], *self.dimg))
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)

    u_tf = self.get_actions_graph(o_tf, g_tf)
    u = u_tf.numpy()
    if o.shape[0] == 1:
      u = u[0]

    return u

  def before_training_hook(self, demo_file=None):
    if self.demo_strategy != "none" or self.sample_demo_buffer:
      assert os.path.isfile(demo_file), "Demo file not exist."
      experiences = self.demo_buffer.load_from_file(data_file=demo_file)
      if self.sample_demo_buffer:
        self.update_stats(experiences)

    if self.demo_strategy in ["nf", "gan"]:
      self.train_shaping()

    if self.initialize_with_bc:
      self.train_bc()
      self.initialize_target_net()

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
    """Pickles the current policy for later inspection.
        """
    with open(path, "wb") as f:
      pickle.dump(self, f)

  def sample_batch(self):
    if self.use_n_step_return:
      one_step_batch = self.replay_buffer.sample(self.batch_size // 2)
      n_step_batch = self.n_step_replay_buffer.sample(self.batch_size -
                                                      self.batch_size // 2)
      assert one_step_batch.keys() == n_step_batch.keys()
      batch = dict()
      for k in one_step_batch.keys():
        batch[k] = np.concatenate((one_step_batch[k], n_step_batch[k]))
    else:
      batch = self.replay_buffer.sample(self.batch_size)

    if self.sample_demo_buffer:
      rollout_batch = batch
      demo_batch = self.demo_buffer.sample(self.batch_size_demo)
      assert rollout_batch.keys() == demo_batch.keys()
      for k in rollout_batch.keys():
        batch[k] = np.concatenate((rollout_batch[k], demo_batch[k]))

    return batch

  @tf.function
  def train_bc_graph(self, o, g, u):
    o = self.o_stats.normalize(o)
    g = self.g_stats.normalize(g)
    with tf.GradientTape() as tape:
      pi = self.main_actor([o, g])
      bc_loss = tf.reduce_mean(tf.square(pi - u))
    actor_grads = tape.gradient(bc_loss, self.main_actor.trainable_weights)
    self.bc_optimizer.apply_gradients(
        zip(actor_grads, self.main_actor.trainable_weights))
    return bc_loss

  def train_bc(self):

    if not self.initialize_with_bc:
      return

    # NOTE: keep the following two versions equivalent
    for epoch in range(self.initialize_num_epochs):
      demo_data_iter = self.demo_buffer.sample(batch_size=self.batch_size_demo,
                                               shuffle=True,
                                               return_iterator=True,
                                               include_partial_batch=True)
      for batch in demo_data_iter:
        o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
        g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
        u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
        bc_loss_tf = self.train_bc_graph(o_tf, g_tf, u_tf)
        bc_loss = bc_loss_tf.numpy()

      if epoch % (self.initialize_num_epochs / 100) == (
          self.initialize_num_epochs / 100 - 1):
        logger.info("epoch: {} policy initialization loss: {}".format(
            epoch, bc_loss))

    # demo_data_iter = self.demo_buffer.sample(return_iterator=True)
    # demo_data = next(demo_data_iter)
    # for epoch in range(self.initialize_num_epochs):
    #   for (o, g, u) in iterbatches(
    #       (demo_data["o"], demo_data["g"], demo_data["u"]),
    #       batch_size=self.batch_size_demo,
    #       include_final_partial_batch=True):

    #     o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    #     g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    #     u_tf = tf.convert_to_tensor(u, dtype=tf.float32)
    #     bc_loss_tf = self.train_bc_graph(o_tf, g_tf, u_tf)
    #     bc_loss = bc_loss_tf.numpy()

    #   if epoch % (self.initialize_num_epochs / 100) == (
    #       self.initialize_num_epochs / 100 - 1):
    #     logger.info("epoch: {} policy initialization loss: {}".format(
    #         epoch, bc_loss))

  def train_shaping(self):

    with tf.summary.record_if(lambda: self.shaping.training_step % 1000 == 0):

      demo_data_iter = self.demo_buffer.sample(return_iterator=True)
      demo_data = next(demo_data_iter)

      # for version 1
      self.shaping.train(demo_data)
      self.shaping.evaluate(demo_data)

      # for version 2
      # self.shaping.training_before_hook(batch=demo_data)

      # for i in range(self.shaping_params["num_epochs"]):
      #   demo_data_iter = self.demo_buffer.sample(return_iterator=True,
      #                                            include_partial_batch=True)
      #   demo_data_iter(128)  # shaping batch size
      #   for batch in demo_data_iter:
      #     self.shaping.train(batch["o"], batch["g"], batch["u"])
      #     # self.shaping.evaluate(batch["o"], batch["g"], batch["u"])

      #   if i % 100 == 0:
      #     logger.info("traing shaping finished epoch", i)

      # self.shaping.training_after_hook(batch=demo_data)

  def critic_loss_graph(self, o, g, o_2, g_2, u, r, n, done):

    # normalize observations
    norm_o = self.o_stats.normalize(o)
    norm_g = self.g_stats.normalize(g)
    norm_o_2 = self.o_stats.normalize(o_2)
    norm_g_2 = self.g_stats.normalize(g_2)

    # add noise to target policy output
    # noise = tfd.Normal(loc=0.0, scale=self.policy_noise).sample(tf.shape(u))
    noise = tf.random.normal(tf.shape(u), 0.0, self.policy_noise)

    noise = tf.clip_by_value(noise, -self.policy_noise_clip,
                             self.policy_noise_clip) * self.max_u
    u_2 = tf.clip_by_value(
        self.target_actor([norm_o_2, norm_g_2]) + noise, -self.max_u,
        self.max_u)

    # immediate reward
    target = r
    # demo shaping reward
    if self.demo_strategy in ["gan", "nf"]:
      potential_curr = self.potential_weight * self.shaping.potential(
          o=o, g=g, u=u)
      potential_next = self.potential_weight * self.shaping.potential(
          o=o_2, g=g_2, u=u_2)
      target += (1.0 - done) * tf.pow(self.gamma,
                                      n) * potential_next - potential_curr
    # ddpg or td3 target with or without clipping
    if self.twin_delayed:
      target += (1.0 - done) * tf.pow(self.gamma, n) * tf.minimum(
          self.target_critic([norm_o_2, norm_g_2, u_2]),
          self.target_critic_twin([norm_o_2, norm_g_2, u_2]),
      )
    else:
      target += (1.0 - done) * tf.pow(self.gamma, n) * self.target_critic(
          [norm_o_2, norm_g_2, u_2])

    if self.twin_delayed:
      bellman_error = tf.square(
          tf.stop_gradient(target) -
          self.main_critic([norm_o, norm_g, u])) + tf.square(
              tf.stop_gradient(target) -
              self.main_critic_twin([norm_o, norm_g, u]))
    else:
      bellman_error = tf.square(
          tf.stop_gradient(target) - self.main_critic([norm_o, norm_g, u]))

    if self.sample_demo_buffer and not self.use_demo_reward:
      # mask off entries from demonstration dataset
      mask = np.concatenate(
          (np.ones(self.batch_size), np.zeros(self.batch_size_demo)), axis=0)
      bellman_error = tf.boolean_mask(bellman_error, mask)

    critic_loss = tf.reduce_mean(bellman_error)

    with tf.name_scope('TD3Losses/'):
      tf.summary.scalar(name='critic_loss vs td3_training_step',
                        data=critic_loss,
                        step=self.td3_training_step)

    return critic_loss

  def actor_loss_graph(self, o, g, u):

    # normalize observations
    norm_o = self.o_stats.normalize(o)
    norm_g = self.g_stats.normalize(g)

    pi = self.main_actor([norm_o, norm_g])
    actor_loss = -tf.reduce_mean(self.main_critic([norm_o, norm_g, pi]))
    actor_loss += self.action_l2 * tf.reduce_mean(tf.square(pi / self.max_u))
    if self.demo_strategy in ["gan", "nf"]:
      actor_loss += -tf.reduce_mean(
          self.potential_weight * self.shaping.potential(o=o, g=g, u=pi))
    if self.demo_strategy == "bc":
      assert self.sample_demo_buffer, "must sample from the demonstration buffer to use behavior cloning"
      mask = np.concatenate(
          (np.zeros(self.batch_size), np.ones(self.batch_size_demo)), axis=0)
      demo_pi = tf.boolean_mask((pi), mask)
      demo_u = tf.boolean_mask((u), mask)
      if self.bc_params["q_filter"]:
        q_u = self.main_critic([norm_o, norm_g, u])
        q_pi = self.main_critic([norm_o, norm_g, pi])
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
  def train_graph(self, o, g, o_2, g_2, u, r, n, done):

    # Train critic
    critic_trainable_weights = self.main_critic.trainable_weights
    if self.twin_delayed:
      critic_trainable_weights += self.main_critic_twin.trainable_weights

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(critic_trainable_weights)
      critic_loss = self.critic_loss_graph(o, g, o_2, g_2, u, r, n, done)

    critic_grads = tape.gradient(critic_loss, critic_trainable_weights)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, critic_trainable_weights))

    # Train actor
    if self.td3_training_step % self.policy_freq == 0:
      actor_trainable_weights = self.main_actor.trainable_weights
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(actor_trainable_weights)
        actor_loss = self.actor_loss_graph(o, g, u)

      actor_grads = tape.gradient(actor_loss, actor_trainable_weights)
      self.actor_optimizer.apply_gradients(
          zip(actor_grads, actor_trainable_weights))
    else:
      actor_loss = tf.constant(0.0)

    self.td3_training_step.assign_add(1)

    return critic_loss, actor_loss

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
      critic_loss_tf, actor_loss_tf = self.train_graph(o_tf, g_tf, o_2_tf,
                                                       g_2_tf, u_tf, r_tf, n_tf,
                                                       done_tf)

  def update_potential_weight(self):
    if self.potential_decay_epoch < self.shaping_params["potential_decay_epoch"]:
      self.potential_decay_epoch += 1
      return self.potential_weight.numpy()
    potential_weight_tf = self.potential_weight.assign(
        self.potential_weight * self.potential_decay_scale)
    return potential_weight_tf.numpy()

  def initialize_target_net(self):
    hard_copy_func = lambda v: v[0].assign(v[1])
    list(
        map(hard_copy_func,
            zip(self.target_actor.weights, self.main_actor.weights)))
    list(
        map(hard_copy_func,
            zip(self.target_critic.weights, self.main_critic.weights)))
    if self.twin_delayed:
      list(
          map(
              hard_copy_func,
              zip(self.target_critic_twin.weights,
                  self.main_critic_twin.weights)))

  def update_target_net(self):
    soft_copy_func = lambda v: v[0].assign(self.polyak * v[0] +
                                           (1.0 - self.polyak) * v[1])
    list(
        map(soft_copy_func,
            zip(self.target_actor.weights, self.main_actor.weights)))
    list(
        map(soft_copy_func,
            zip(self.target_critic.weights, self.main_critic.weights)))
    if self.twin_delayed:
      list(
          map(
              soft_copy_func,
              zip(self.target_critic_twin.weights,
                  self.main_critic_twin.weights)))

  def logs(self, prefix=""):
    logs = []
    logs.append(
        (prefix + "stats_o/mean", np.mean(self.o_stats.mean_tf.numpy())))
    logs.append((prefix + "stats_o/std", np.mean(self.o_stats.std_tf.numpy())))
    logs.append(
        (prefix + "stats_g/mean", np.mean(self.g_stats.mean_tf.numpy())))
    logs.append((prefix + "stats_g/std", np.mean(self.g_stats.std_tf.numpy())))
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
    self.o_stats.update(o_tf)
    self.g_stats.update(g_tf)

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
      self.replay_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size,
                                               self.eps_length)
      if self.use_n_step_return:
        self.n_step_replay_buffer = MultiStepReplayBuffer(
            buffer_shapes, self.buffer_size, self.eps_length,
            self.n_step_return_steps, self.gamma)
      if self.demo_strategy != "none" or self.sample_demo_buffer:
        self.demo_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size,
                                               self.eps_length)
    else:
      self.replay_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)
      assert not self.use_n_step_return, "not implemented yet"
      if self.demo_strategy != "none" or self.sample_demo_buffer:
        self.demo_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)

  def _create_network(self):

    # Normalizer for goal and observation.
    self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
    self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)

    # Models
    # actor
    self.main_actor = Actor(self.dimo, self.dimg, self.dimu, self.max_u,
                            self.layer_sizes)
    self.target_actor = Actor(self.dimo, self.dimg, self.dimu, self.max_u,
                              self.layer_sizes)
    # critic
    self.main_critic = Critic(self.dimo, self.dimg, self.dimu, self.max_u,
                              self.layer_sizes)
    self.target_critic = Critic(self.dimo, self.dimg, self.dimu, self.max_u,
                                self.layer_sizes)
    if self.twin_delayed:
      self.main_critic_twin = Critic(self.dimo, self.dimg, self.dimu,
                                     self.max_u, self.layer_sizes)
      self.target_critic_twin = Critic(self.dimo, self.dimg, self.dimu,
                                       self.max_u, self.layer_sizes)

    # create weights
    o_tf = tf.zeros([0, *self.dimo])
    g_tf = tf.zeros([0, *self.dimg])
    u_tf = tf.zeros([0, *self.dimu])
    self.main_actor([o_tf, g_tf])
    self.target_actor([o_tf, g_tf])
    self.main_critic([o_tf, g_tf, u_tf])
    self.target_critic([o_tf, g_tf, u_tf])
    if self.twin_delayed:
      self.main_critic_twin([o_tf, g_tf, u_tf])
      self.target_critic_twin([o_tf, g_tf, u_tf])

    # Optimizers
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
    if self.twin_delayed:
      self.critic_twin_optimizer = tf.keras.optimizers.Adam(
          learning_rate=self.q_lr)
    self.bc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)

    # Add shaping reward
    shaping_class = {"nf": NFShaping, "gan": GANShaping}
    if self.demo_strategy in shaping_class.keys():
      # instantiate shaping version 1
      self.shaping = EnsembleShaping(
          shaping_cls=shaping_class[self.demo_strategy],
          num_ensembles=self.shaping_params["num_ensembles"],
          batch_size=self.shaping_params["batch_size"],
          num_epochs=self.shaping_params["num_epochs"],
          dimo=self.dimo,
          dimg=self.dimg,
          dimu=self.dimu,
          max_u=self.max_u,
          norm_obs=True,
          norm_eps=self.norm_eps,
          norm_clip=self.norm_clip,
          **self.shaping_params[self.demo_strategy].copy())
      # instantiate shaping version 2
      # self.shaping = EnsembleShapingV2(
      #     shaping_cls=shaping_class[self.demo_strategy],
      #     num_ensembles=self.shaping_params["num_ensembles"],
      #     dimo=self.dimo,
      #     dimg=self.dimg,
      #     dimu=self.dimu,
      #     max_u=self.max_u,
      #     norm_obs=True,
      #     norm_eps=self.norm_eps,
      #     norm_clip=self.norm_clip,
      #     **self.shaping_params[self.demo_strategy].copy())
    else:
      self.shaping = None

    # Meta-learning for weight on potential
    self.potential_weight = tf.Variable(1.0, trainable=False)
    self.potential_decay_scale = self.shaping_params["potential_decay_scale"]
    self.potential_decay_epoch = 0  # eventually becomes self.shaping_params["potential_decay_epoch"]

    # Initialize all variables
    self.initialize_target_net()

  @tf.function
  def _policy_inspect(self, o=None, g=None, summarize=False):
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

    norm_o = self.o_stats.normalize(o)
    norm_g = self.g_stats.normalize(g)
    u = self.main_actor([norm_o, norm_g])

    self._policy_inspect_count.assign_add(1)
    q = self.main_critic([norm_o, norm_g, u])
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
        "o_stats": self.o_stats.get_weights(),
        "g_stats": self.g_stats.get_weights(),
        "main_actor": self.main_actor.get_weights(),
        "target_actor": self.target_actor.get_weights(),
        "main_critic": self.main_critic.get_weights(),
        "target_critic": self.target_critic.get_weights(),
        "main_critic_twin": self.main_critic_twin.get_weights(),
        "target_critic_twin": self.target_critic_twin.get_weights(),
    }
    return state

  def __setstate__(self, state):
    stored_vars = state.pop("tf")
    shaping = state.pop("shaping")
    self.__init__(**state)
    self.o_stats.set_weights(stored_vars["o_stats"])
    self.g_stats.set_weights(stored_vars["g_stats"])
    self.main_actor.set_weights(stored_vars["main_actor"])
    self.target_actor.set_weights(stored_vars["target_actor"])
    self.main_critic.set_weights(stored_vars["main_critic"])
    self.target_critic.set_weights(stored_vars["target_critic"])
    self.main_critic_twin.set_weights(stored_vars["main_critic_twin"])
    self.target_critic_twin.set_weights(stored_vars["target_critic_twin"])
    self.shaping = shaping
