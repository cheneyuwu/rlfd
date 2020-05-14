import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rlfd import logger
from rlfd.td3.actorcritic_network import Actor, Critic
from rlfd.td3.shaping import EnsembleRewardShapingWrapper
from rlfd.td3.normalizer import Normalizer
from rlfd.memory import RingReplayBuffer, UniformReplayBuffer, MultiStepReplayBuffer, iterbatches
from rlfd.mage.model_network import EnsembleModelNetwork

tfd = tfp.distributions


class MAGE(object):

  def __init__(
      self,
      # for learning
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
      num_demo,
      demo_strategy,
      bc_params,
      shaping_params,
      info,
  ):
    # Store initial args passed into the function
    self.init_args = locals()

    self.num_epochs = num_epochs
    self.num_cycles = num_cycles
    self.num_batches = num_batches
    self.buffer_size = buffer_size
    self.batch_size = batch_size

    self.num_demo = num_demo
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

  @tf.function
  def get_actions_graph(self, o, g):

    norm_o = self.o_stats.normalize(o)
    norm_g = self.g_stats.normalize(g)

    u = self.main_actor([norm_o, norm_g])

    q = self.main_critic([norm_o, norm_g, u])
    if self.shaping != None:
      p = self.potential_weight * self.shaping.potential(o=o, g=g, u=u)
    else:
      p = tf.zeros_like(q)

    if tf.shape(o)[0] == 1:
      u = u[0]

    return u, q, p

  def get_actions(self, o, g, compute_q=False):
    o = o.reshape((-1, *self.dimo))
    g = g.reshape((o.shape[0], *self.dimg))
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)

    u_tf, q_tf, p_tf = self.get_actions_graph(o_tf, g_tf)

    if compute_q:
      return [u_tf.numpy(), q_tf.numpy(), p_tf.numpy() + q_tf.numpy()]
    else:
      return u_tf.numpy()

  def init_demo_buffer(self, demo_file):
    """Initialize the demonstration buffer.
        """
    episode_batch = self.demo_buffer.load_from_file(data_file=demo_file)
    return episode_batch

  def add_to_demo_buffer(self, episode_batch):
    self.demo_buffer.store(episode_batch)

  def store_episode(self, episode_batch):
    """
    episode_batch: array of batch_size x (T or T+1) x dim_key ('o' and 'ag' is of size T+1, others are of size T)
    """
    self.replay_buffer.store(episode_batch)
    if self.use_n_step_return:
      self.n_step_replay_buffer.store(episode_batch)

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
  def evaluate_model_graph(self, o_tf, o_2_tf, u_tf, r_tf):
    (mean, var) = self.model_network((o_tf, u_tf))
    delta_o_mean, r_mean = mean
    o_var, r_var = var

    # Use delta observation as target
    delta_o_2_tf = o_2_tf - o_tf

    o_mean_loss = tf.reduce_mean(
        tf.square(delta_o_mean - delta_o_2_tf),
        axis=[-2, -1],
    )
    r_mean_loss = tf.reduce_mean(
        tf.square(r_mean - r_tf),
        axis=[-2, -1],
    )

    model_loss = o_mean_loss + r_mean_loss

    return model_loss

  @tf.function
  def train_model_graph(self, o_tf, o_2_tf, u_tf, r_tf):

    # TODO: normalize observations
    # o_tf = self.o_stats.normalize(o_tf)
    # norm_o_2_tf = self.o_stats.normalize(o_2_tf)

    random_idx = tf.random.uniform(
        shape=[self.model_network.num_networks, o_tf.shape[0]],
        minval=0,
        maxval=o_tf.shape[0],
        dtype=tf.dtypes.int32,
    )
    o_tf = tf.gather(o_tf, random_idx)
    o_2_tf = tf.gather(o_2_tf, random_idx)
    u_tf = tf.gather(u_tf, random_idx)
    r_tf = tf.gather(r_tf, random_idx)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.model_network.trainable_variables)

      (mean, var) = self.model_network((o_tf, u_tf))
      delta_o_mean, r_mean = mean
      o_var, r_var = var

      # Use delta observation as target
      delta_o_2_tf = o_2_tf - o_tf

      o_mean_loss = tf.reduce_mean(
          tf.square(delta_o_mean - delta_o_2_tf) / o_var,
          axis=[-2, -1],
      )
      o_var_loss = tf.reduce_mean(
          tf.math.log(o_var),
          axis=[-2, -1],
      )
      r_mean_loss = tf.reduce_mean(
          tf.square(r_mean - r_tf) / r_var,
          axis=[-2, -1],
      )
      r_var_loss = tf.reduce_mean(tf.math.log(r_var), axis=[-2, -1])

      model_loss = tf.reduce_sum(o_mean_loss + r_mean_loss + o_var_loss +
                                 r_var_loss)

      # TODO: add layer decays

      # network variance boundary loss
      logvar_bound_loss = 0.01 * tf.reduce_sum(
          self.model_network.max_logvar) - 0.01 * tf.reduce_sum(
              self.model_network.min_logvar)
      model_loss = model_loss + logvar_bound_loss

    model_grads = tape.gradient(model_loss,
                                self.model_network.trainable_variables)
    self.model_optimizer.apply_gradients(
        zip(model_grads, self.model_network.trainable_variables))

    return model_loss

  def train_model(self):

    # TODO: move hyperparameters to config files
    holdout_ratio = 0.2  # used in mbpo to determine validation dataset size (not used for now)
    max_training_steps = 5000  # in mbpo this is based on time taken to train the model
    batch_size = 256  # used in mbpo
    max_logging = 5000  # maximum validation and evaluation number of experiences

    training_steps = 0
    training_epochs = 0
    while True:  # 1 epoch of training (or use max training epochs maybe)
      validation_size = min(
          int(self.replay_buffer.stored_steps * holdout_ratio), max_logging)
      iterator = self.replay_buffer.sample(batch_size=validation_size,
                                           shuffle=True,
                                           return_iterator=True,
                                           include_partial_batch=True)
      validation_experiences = next(iterator)
      iterator(batch_size)  # start training

      for experiences in iterator:
        training_exps_tf = {
            k + "_tf": tf.convert_to_tensor(experiences[k], dtype=tf.float32)
            for k in ["o", "o_2", "u", "r"]
        }
        self.train_model_graph(**training_exps_tf).numpy()

        training_steps += 1

      # validation
      validation_exps_tf = {
          k + "_tf": tf.convert_to_tensor(validation_experiences[k],
                                          dtype=tf.float32)
          for k in ["o", "o_2", "u", "r"]
      }
      holdout_loss = self.evaluate_model_graph(**validation_exps_tf).numpy()
      logger.info("epoch: {} holdout loss: {}".format(training_epochs,
                                                      holdout_loss))

      sorted_inds = np.argsort(holdout_loss)

      training_epochs += 1
      if training_steps > max_training_steps:
        break

    elites_inds = sorted_inds[:self.model_network.num_elites].tolist()
    self.model_network.set_elite_inds(elites_inds)

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
    assert self.demo_strategy in ["nf", "gan"]
    demo_data_iter = self.demo_buffer.sample(return_iterator=True)
    demo_data = next(demo_data_iter)
    self.shaping.train(demo_data)
    self.shaping.evaluate(demo_data)

  def critic_gradient_loss_graph(self, o, g, o_2, g_2, u, r, n, done):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(u)  # with respect to action

      # compute model output
      (mean, var) = self.model_network((o, u))
      delta_o_mean, r_mean = mean
      o_var, r_var = var

      o_mean = o + delta_o_mean
      o_std, r_std = tf.sqrt(o_var), tf.sqrt(r_var)

      # choose between deterministic and non-deterministic
      o_sample = o_mean + tf.random.normal(o_mean.shape) * o_std
      r_sample = r_mean + tf.random.normal(r_mean.shape) * r_std

      # choose a random netowrk
      batch_inds = tf.range(o_sample.shape[1])
      model_inds = tf.random.uniform(shape=[o_sample.shape[1]],
                                     minval=0,
                                     maxval=self.model_network.num_elites,
                                     dtype=tf.dtypes.int32)
      model_inds = tf.gather(self.model_network.get_elite_inds(), model_inds)
      indices = tf.stack([model_inds, batch_inds], axis=-1)

      # Replace reward and next observation
      o_2 = tf.gather_nd(o_sample, indices)
      r = tf.gather_nd(r_sample, indices)

      # normalize observations
      norm_o = self.o_stats.normalize(o)
      norm_g = self.g_stats.normalize(g)
      norm_o_2 = self.o_stats.normalize(o_2)
      norm_g_2 = self.g_stats.normalize(g_2)

      # add noise to target policy output
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
            target - self.main_critic([norm_o, norm_g, u])) + tf.square(
                target - self.main_critic_twin([norm_o, norm_g, u]))
      else:
        bellman_error = tf.square(target -
                                  self.main_critic([norm_o, norm_g, u]))

      if self.sample_demo_buffer and not self.use_demo_reward:
        # mask off entries from demonstration dataset
        mask = np.concatenate(
            (np.ones(self.batch_size), np.zeros(self.batch_size_demo)), axis=0)
        bellman_error = tf.boolean_mask(bellman_error, mask)

      critic_loss = tf.reduce_mean(bellman_error)

    critic_gradient_loss = tf.norm(tape.gradient(critic_loss, u))

    with tf.name_scope('MAGETD3Losses'):
      tf.summary.scalar(name='critic_gradient_loss vs td3_training_step',
                        data=critic_loss,
                        step=self.td3_training_step)

    return critic_gradient_loss

  def critic_loss_graph(self, o, g, o_2, g_2, u, r, n, done):

    # # compute model output
    # (mean, var) = self.model_network((o, u))
    # delta_o_mean, r_mean = mean
    # o_var, r_var = var

    # o_mean = o + delta_o_mean
    # o_std, r_std = tf.sqrt(o_var), tf.sqrt(r_var)

    # # choose between deterministic and non-deterministic
    # o_sample = o_mean + tf.random.normal(o_mean.shape) * o_std
    # r_sample = r_mean + tf.random.normal(r_mean.shape) * r_std

    # # choose a random netowrk
    # batch_inds = tf.range(o_sample.shape[1])
    # model_inds = tf.random.uniform(shape=[o_sample.shape[1]],
    #                                minval=0,
    #                                maxval=self.model_network.num_elites,
    #                                dtype=tf.dtypes.int32)
    # model_inds = tf.gather(self.model_network.get_elite_inds(), model_inds)
    # indices = tf.stack([model_inds, batch_inds], axis=-1)

    # o_sample = tf.gather_nd(o_sample, indices)
    # r_sample = tf.gather_nd(r_sample, indices)

    # # Replace reward and next observation
    # o_2 = o_sample
    # norm_o_2 = self.o_stats.normalize(o_2)
    # r = r_sample

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

    with tf.name_scope('MAGETD3Losses'):
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

    with tf.name_scope('MAGETD3Losses'):
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
      critic_gradient_loss = self.critic_gradient_loss_graph(
          o, g, o_2, g_2, u, r, n, done)
      # TODO: hyper parameter, move to config files
      mage_critic_loss = critic_gradient_loss + 0.01 * critic_loss

    critic_grads = tape.gradient(mage_critic_loss, critic_trainable_weights)

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

    batch = self.sample_batch()

    o_tf = tf.convert_to_tensor(batch["o"], dtype=tf.float32)
    g_tf = tf.convert_to_tensor(batch["g"], dtype=tf.float32)
    o_2_tf = tf.convert_to_tensor(batch["o_2"], dtype=tf.float32)
    g_2_tf = tf.convert_to_tensor(batch["g_2"], dtype=tf.float32)
    u_tf = tf.convert_to_tensor(batch["u"], dtype=tf.float32)
    r_tf = tf.convert_to_tensor(batch["r"], dtype=tf.float32)
    n_tf = tf.convert_to_tensor(batch["n"], dtype=tf.float32)
    done_tf = tf.convert_to_tensor(batch["done"], dtype=tf.float32)
    critic_loss_tf, actor_loss_tf = self.train_graph(o_tf, g_tf, o_2_tf, g_2_tf,
                                                     u_tf, r_tf, n_tf, done_tf)

    return critic_loss_tf.numpy(), actor_loss_tf.numpy()

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

  def update_stats(self, episode_batch):
    # add transitions to normalizer
    if self.fix_T:
      transitions = {
          k: v.reshape((-1, v.shape[-1])).copy()
          for k, v in episode_batch.items()
      }
    else:
      transitions = episode_batch.copy()
    o_tf = tf.convert_to_tensor(transitions["o"], dtype=tf.float32)
    g_tf = tf.convert_to_tensor(transitions["g"], dtype=tf.float32)
    self.o_stats.update(o_tf)
    self.g_stats.update(g_tf)

  def _create_memory(self):
    buffer_shapes = {}
    if self.fix_T:
      buffer_shapes["o"] = (self.eps_length, *self.dimo)
      buffer_shapes["o_2"] = (self.eps_length, *self.dimo)
      buffer_shapes["u"] = (self.eps_length, *self.dimu)
      buffer_shapes["r"] = (self.eps_length, 1)
      buffer_shapes["ag"] = (self.eps_length, *self.dimg)
      buffer_shapes["ag_2"] = (self.eps_length, *self.dimg)
      buffer_shapes["g"] = (self.eps_length, *self.dimg)
      buffer_shapes["g_2"] = (self.eps_length, *self.dimg)
      buffer_shapes["done"] = (self.eps_length, 1)

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
      buffer_shapes["o"] = self.dimo
      buffer_shapes["o_2"] = self.dimo
      buffer_shapes["u"] = self.dimu
      buffer_shapes["r"] = (1,)
      buffer_shapes["ag"] = self.dimg
      buffer_shapes["g"] = self.dimg
      buffer_shapes["ag_2"] = self.dimg
      buffer_shapes["g_2"] = self.dimg
      buffer_shapes["done"] = (1,)

      self.replay_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)
      assert not self.use_n_step_return, "not implemented yet"
      if self.demo_strategy != "none" or self.sample_demo_buffer:
        self.demo_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)

  def _create_network(self):

    # Normalizer for goal and observation.
    self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
    self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)

    # Models
    # model
    self.model_network = EnsembleModelNetwork(
        self.dimo,
        self.dimu,
        self.max_u,
        layer_sizes=[200, 200, 200, 200],
        num_networks=5,  # TODO: hyper parameters
        num_elites=3,
    )
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
    self.model_network([o_tf, u_tf])
    self.main_actor([o_tf, g_tf])
    self.target_actor([o_tf, g_tf])
    self.main_critic([o_tf, g_tf, u_tf])
    self.target_critic([o_tf, g_tf, u_tf])
    if self.twin_delayed:
      self.main_critic_twin([o_tf, g_tf, u_tf])
      self.target_critic_twin([o_tf, g_tf, u_tf])

    # Optimizers
    self.model_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3)  # TODO: hyperparameter, same as mbpo
    self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)
    self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
    if self.twin_delayed:
      self.critic_twin_optimizer = tf.keras.optimizers.Adam(
          learning_rate=self.q_lr)
    self.bc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)

    # Add shaping reward
    if self.demo_strategy in ["nf", "gan"]:
      self.shaping = EnsembleRewardShapingWrapper(
          dims=self.dims,
          max_u=self.max_u,
          gamma=self.gamma,
          demo_strategy=self.demo_strategy,
          norm_obs=True,
          norm_eps=self.norm_eps,
          norm_clip=self.norm_clip,
          **self.shaping_params.copy())
    else:
      self.shaping = None

    # Meta-learning for weight on potential
    self.potential_weight = tf.Variable(1.0, trainable=False)
    self.potential_decay_scale = self.shaping_params["potential_decay_scale"]
    self.potential_decay_epoch = 0  # eventually becomes self.shaping_params["potential_decay_epoch"]

    # Initialize all variables
    self.initialize_target_net()
    self.training_step = tf.Variable(0, trainable=False)

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
