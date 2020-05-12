import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rlfd import logger
from rlfd.td3.actorcritic_network import Actor, Critic
from rlfd.td3.shaping import EnsembleRewardShapingWrapper
from rlfd.td3.normalizer import Normalizer
from rlfd.memory import RingReplayBuffer, UniformReplayBuffer, MultiStepReplayBuffer, iterbatches

tfd = tfp.distributions


class TD3(object):

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
    """
    Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER). Added functionality
    to use demonstrations for training to Overcome exploration problem.

    Args:
        # Environment I/O and Config
        max_u              (float)        - maximum action magnitude, i.e. actions are in [-max_u, max_u]
        T                  (int)          - the time horizon for rollouts
        fix_T              (bool)         - every episode has fixed length
        # Normalizer
        norm_eps           (float)        - a small value used in the normalizer to avoid numerical instabilities
        norm_clip          (float)        - normalized inputs are clipped to be in [-norm_clip, norm_clip]
        # NN Configuration
        scope              (str)          - the scope used for the TensorFlow graph
        dims               (dict of tps)  - dimensions for the observation (o), the goal (g), and the actions (u)
        layer_sizes        (list of ints) - number of units in each hidden layers
        initializer_type   (str)          - initializer of the weight for both policy and critic
        reuse              (boolean)      - whether or not the networks should be reused
        # Replay Buffer
        buffer_size        (int)          - number of transitions that are stored in the replay buffer
        # Dual Network Set
        polyak             (float)        - coefficient for Polyak-averaging of the target network
        # Training
        batch_size         (int)          - batch size for training
        Q_lr               (float)        - learning rate for the Q (critic) network
        pi_lr              (float)        - learning rate for the pi (actor) network
        action_l2          (float)        - coefficient for L2 penalty on the actions
        gamma              (float)        - gamma used for Q learning updates
        # Use demonstration to shape critic or actor
        sample_demo_buffer (int)          - whether or not to sample from demonstration buffer
        batch_size_demo    (int)          - number of samples to be used from the demonstrations buffer, per mpi thread
        use_demo_reward    (int)          - whether or not to assue that demonstration dataset has rewards
        num_demo           (int)          - number of episodes in to be used in the demonstration buffer
        demo_strategy      (str)          - whether or not to use demonstration with different strategies
        bc_params          (dict)
        shaping_params     (dict)
    """
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

  @tf.function
  def get_actions_tf(self, o_tf, g_tf):

    norm_o_tf = self.o_stats.normalize(o_tf)
    norm_g_tf = self.g_stats.normalize(g_tf)

    u_tf = self.main_actor([norm_o_tf, norm_g_tf])

    q_tf = self.main_critic([norm_o_tf, norm_g_tf, u_tf])
    if self.shaping != None:
      p_tf = self.potential_weight * self.shaping.potential(
          o=o_tf, g=g_tf, u=u_tf)
    else:
      p_tf = tf.zeros_like(q_tf)

    if tf.shape(o_tf)[0] == 1:
      u_tf = u_tf[0]

    return u_tf, q_tf, p_tf

  def get_actions(self, o, g, compute_q=False):
    o = o.reshape((-1, *self.dimo))
    g = g.reshape((o.shape[0], *self.dimg))
    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)

    u_tf, q_tf, p_tf = self.get_actions_tf(o_tf, g_tf)

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
  def train_bc_tf(self, o_tf, g_tf, u_tf):
    o_tf = self.o_stats.normalize(o_tf)
    g_tf = self.g_stats.normalize(g_tf)
    with tf.GradientTape() as tape:
      pi_tf = self.main_actor([o_tf, g_tf])
      bc_loss_tf = tf.reduce_mean(tf.square(pi_tf - u_tf))
    actor_grads = tape.gradient(bc_loss_tf, self.main_actor.trainable_weights)
    self.bc_optimizer.apply_gradients(
        zip(actor_grads, self.main_actor.trainable_weights))
    return bc_loss_tf

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
        bc_loss_tf = self.train_bc_tf(o_tf, g_tf, u_tf)
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
    #     bc_loss_tf = self.train_bc_tf(o_tf, g_tf, u_tf)
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

  @tf.function
  def train_tf(self, o_tf, g_tf, o_2_tf, g_2_tf, u_tf, r_tf, n_tf, done_tf):

    self.training_step.assign((self.training_step + 1) % self.policy_freq)

    # normalize observations
    norm_o_tf = self.o_stats.normalize(o_tf)
    norm_g_tf = self.g_stats.normalize(g_tf)
    norm_o_2_tf = self.o_stats.normalize(o_2_tf)
    norm_g_2_tf = self.g_stats.normalize(g_2_tf)

    # add noise to target policy output
    noise = tfd.Normal(loc=0.0, scale=self.policy_noise).sample(tf.shape(u_tf))
    noise = tf.clip_by_value(noise, -self.policy_noise_clip,
                             self.policy_noise_clip) * self.max_u
    u_2_tf = tf.clip_by_value(
        self.target_actor([norm_o_2_tf, norm_g_2_tf]) + noise, -self.max_u,
        self.max_u)

    # Critic loss
    # immediate reward
    target_tf = r_tf
    # demo shaping reward
    if self.demo_strategy in ["gan", "nf"]:
      potential_curr = self.potential_weight * self.shaping.potential(
          o=o_tf, g=g_tf, u=u_tf)
      potential_next = self.potential_weight * self.shaping.potential(
          o=o_2_tf, g=g_2_tf, u=u_2_tf)
      target_tf += (1.0 - done_tf) * tf.pow(
          self.gamma, n_tf) * potential_next - potential_curr
    # ddpg or td3 target with or without clipping
    if self.twin_delayed:
      target_tf += (1.0 - done_tf) * tf.pow(self.gamma, n_tf) * tf.minimum(
          self.target_critic([norm_o_2_tf, norm_g_2_tf, u_2_tf]),
          self.target_critic_twin([norm_o_2_tf, norm_g_2_tf, u_2_tf]),
      )
    else:
      target_tf += (1.0 - done_tf) * tf.pow(
          self.gamma, n_tf) * self.target_critic(
              [norm_o_2_tf, norm_g_2_tf, u_2_tf])
    with tf.GradientTape(persistent=True) as tape:
      if self.twin_delayed:
        rl_bellman_tf = tf.square(target_tf - self.main_critic(
            [norm_o_tf, norm_g_tf, u_tf])) + tf.square(
                target_tf - self.main_critic_twin([norm_o_tf, norm_g_tf, u_tf]))
      else:
        rl_bellman_tf = tf.square(
            target_tf - self.main_critic([norm_o_tf, norm_g_tf, u_tf]))

      if self.sample_demo_buffer and not self.use_demo_reward:
        # mask off entries from demonstration dataset
        mask = np.concatenate(
            (np.ones(self.batch_size), np.zeros(self.batch_size_demo)), axis=0)
        rl_bellman_tf = tf.boolean_mask(rl_bellman_tf, mask)

      critic_loss_tf = tf.reduce_mean(rl_bellman_tf)

    critic_grads = tape.gradient(critic_loss_tf,
                                 self.main_critic.trainable_weights)
    if self.twin_delayed:
      critic_twin_grads = tape.gradient(critic_loss_tf,
                                        self.main_critic_twin.trainable_weights)

    self.critic_optimizer.apply_gradients(
        zip(critic_grads, self.main_critic.trainable_weights))
    if self.twin_delayed:
      self.critic_twin_optimizer.apply_gradients(
          zip(critic_twin_grads, self.main_critic_twin.trainable_weights))

    if self.training_step % self.policy_freq != 0:
      return critic_loss_tf, tf.constant(0.0)

    # Actor Loss
    with tf.GradientTape(persistent=True) as tape:
      pi_tf = self.main_actor([norm_o_tf, norm_g_tf])
      actor_loss_tf = -tf.reduce_mean(
          self.main_critic([norm_o_tf, norm_g_tf, pi_tf]))
      actor_loss_tf += self.action_l2 * tf.reduce_mean(
          tf.square(pi_tf / self.max_u))
      if self.demo_strategy in ["gan", "nf"]:
        actor_loss_tf += -tf.reduce_mean(
            self.potential_weight *
            self.shaping.potential(o=o_tf, g=g_tf, u=pi_tf))
      if self.demo_strategy == "bc":
        assert self.sample_demo_buffer, "must sample from the demonstration buffer to use behavior cloning"
        # define the cloning loss on the actor's actions only on the samples which adhere to the above masks
        mask = np.concatenate(
            (np.zeros(self.batch_size), np.ones(self.batch_size_demo)), axis=0)
        demo_pi_tf = tf.boolean_mask((pi_tf), mask)
        demo_u_tf = tf.boolean_mask((u_tf), mask)
        if self.bc_params["q_filter"]:
          q_u_tf = self.main_critic([norm_o_tf, norm_g_tf, u_tf])
          q_pi_tf = self.main_critic([norm_o_tf, norm_g_tf, pi_tf])
          q_filter_mask = tf.reshape(tf.boolean_mask(q_u_tf > q_pi_tf, mask),
                                     [-1])
          # use to be tf.reduce_sum, however, use tf.reduce_mean makes the loss function independent from number
          # of demonstrations
          bc_loss_tf = tf.reduce_mean(
              tf.square(
                  tf.boolean_mask(demo_pi_tf, q_filter_mask, axis=0) -
                  tf.boolean_mask(demo_u_tf, q_filter_mask, axis=0)))
        else:
          # use to be tf.reduce_sum, however, use tf.reduce_mean makes the loss function independent from number
          # of demonstrations
          bc_loss_tf = tf.reduce_mean(tf.square(demo_pi_tf - demo_u_tf))
        actor_loss_tf = (self.bc_params["prm_loss_weight"] * actor_loss_tf +
                         self.bc_params["aux_loss_weight"] * bc_loss_tf)

    actor_grads = tape.gradient(actor_loss_tf,
                                self.main_actor.trainable_weights)
    self.actor_optimizer.apply_gradients(
        zip(actor_grads, self.main_actor.trainable_weights))

    return critic_loss_tf, actor_loss_tf

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
    critic_loss_tf, actor_loss_tf = self.train_tf(o_tf, g_tf, o_2_tf, g_2_tf,
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
    # actor
    self.main_actor = Actor(self.dimo, self.dimg, self.dimu, self.max_u,
                            self.layer_sizes)
    self.target_actor = Actor(
        self.dimo,
        self.dimg,
        self.dimu,
        self.max_u,
        self.layer_sizes,
    )
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
