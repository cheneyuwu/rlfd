import pickle

import numpy as np
import tensorflow as tf

from td3fd.ddpg.demo_shaping import EnsGANDemoShaping, EnsNFDemoShaping, GANDemoShaping
from td3fd.ddpg.actorcritic_network import Actor, Critic
from td3fd.memory import RingReplayBuffer, UniformReplayBuffer
from td3fd.ddpg2.normalizer import Normalizer


class DDPG(object):
    def __init__(
        self,
        input_dims,
        use_td3,
        layer_sizes,
        polyak,
        buffer_size,
        batch_size,
        q_lr,
        pi_lr,
        norm_eps,
        norm_clip,
        max_u,
        action_l2,
        clip_obs,
        scope,
        eps_length,
        fix_T,
        clip_pos_returns,
        clip_return,
        sample_demo_buffer,
        batch_size_demo,
        use_demo_reward,
        num_demo,
        demo_strategy,
        bc_params,
        shaping_params,
        gamma,
        info,
        num_epochs,
        num_cycles,
        num_batches,
    ):
        """
        Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER). Added functionality
        to use demonstrations for training to Overcome exploration problem.

        Args:
            # Environment I/O and Config
            max_u              (float)        - maximum action magnitude, i.e. actions are in [-max_u, max_u]
            T                  (int)          - the time horizon for rollouts
            fix_T              (bool)         - every episode has fixed length
            clip_obs           (float)        - clip observations before normalization to be in [-clip_obs, clip_obs]
            clip_pos_returns   (boolean)      - whether or not positive returns should be clipped (i.e. clip to 0)
            clip_return        (float)        - clip returns to be in [-clip_return, clip_return]
            # Normalizer
            norm_eps           (float)        - a small value used in the normalizer to avoid numerical instabilities
            norm_clip          (float)        - normalized inputs are clipped to be in [-norm_clip, norm_clip]
            # NN Configuration
            scope              (str)          - the scope used for the TensorFlow graph
            input_dims         (dict of tps)  - dimensions for the observation (o), the goal (g), and the actions (u)
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

        # Parameters
        self.num_epochs = num_epochs
        self.num_cycles = num_cycles
        self.num_batches = num_batches
        self.input_dims = input_dims
        self.use_td3 = use_td3
        self.layer_sizes = layer_sizes
        self.polyak = polyak
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.max_u = max_u
        self.action_l2 = action_l2
        self.eps_length = eps_length
        self.fix_T = fix_T
        self.sample_demo_buffer = sample_demo_buffer
        self.batch_size_demo = batch_size_demo
        self.use_demo_reward = use_demo_reward
        self.num_demo = num_demo
        self.demo_strategy = demo_strategy
        assert self.demo_strategy in ["none", "pure_bc", "bc", "gan", "nf"]
        self.bc_params = bc_params
        self.shaping_params = shaping_params
        self.gamma = gamma
        self.info = info

        # Prepare parameters
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        # Get a tf session
        with tf.compat.v1.variable_scope(scope):
            self.scope = tf.compat.v1.get_variable_scope()
            self._create_memory()
            self._create_network()
        self.saver = tf.compat.v1.train.Saver(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        )

    @tf.function
    def _get_actions(self, o, g):
        norm_o = self.o_stats.normalize(o)
        norm_g = self.g_stats.normalize(g)

        main_pi = self.main_actor(o=norm_o, g=norm_g)
        main_q_pi = self.main_critic(o=norm_o, g=norm_g, u=main_pi)
        res = [main_pi, main_q_pi]
        if self.demo_shaping != None:
            true_q = self.demo_shaping.potential(o=o, g=g, u=main_pi) + main_q_pi
            res.append(true_q)
        else:
            res.append(main_q_pi)

        return res

    def get_actions(self, o, g, compute_q=False):
        o = tf.convert_to_tensor(o.reshape(-1, *self.dimo))
        g = tf.convert_to_tensor(g.reshape(-1, *self.dimg))
        res = self._get_actions(o, g)

        if compute_q:
            return [r.numpy() for r in res]
        else:
            return res[0].numpy()

    def init_demo_buffer(self, demo_file, update_stats=True):
        """Initialize the demonstration buffer.
        """
        # load the demonstration data from data file
        episode_batch = self.demo_buffer.load_from_file(data_file=demo_file, num_demo=self.num_demo)
        self._update_demo_stats(episode_batch)
        if update_stats:
            self._update_stats(episode_batch)

    def add_to_demo_buffer(self, episode_batch):
        self.demo_buffer.store_episode(episode_batch)

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key ('o' and 'ag' is of size T+1, others are of size T)
        """
        self.replay_buffer.store_episode(episode_batch)
        if update_stats:
            self._update_stats(episode_batch)

    def sample_batch(self):
        # use demonstration buffer to sample as well if demo flag is set TRUE
        if self.sample_demo_buffer:
            transitions = {}
            transition_rollout = self.replay_buffer.sample(self.batch_size)
            transition_demo = self.demo_buffer.sample(self.batch_size_demo)
            assert transition_rollout.keys() == transition_demo.keys()
            for k in transition_rollout.keys():
                transitions[k] = np.concatenate((transition_rollout[k], transition_demo[k]))
        else:
            transitions = self.replay_buffer.sample(self.batch_size)  # otherwise only sample from primary buffer
        return transitions

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def save_replay_buffer(self, path):
        self.replay_buffer.dump_to_file(path)

    def load_replay_buffer(self, path):
        self.replay_buffer.load_from_file(path)

    def save_weights(self, path):
        self.saver.save(self.sess, path)

    def load_weights(self, path):
        self.saver.restore(self.sess, path)

    def train_shaping(self):
        assert self.demo_strategy in ["nf", "gan"]
        # train normalizing flow or gan for 1 epoch
        loss = 0
        self.demo_shaping.initialize_dataset(sess=self.sess)
        losses = np.empty(0)
        while True:
            try:
                loss = self.demo_shaping.train(sess=self.sess)
                losses = np.append(losses, loss)
            except tf.errors.OutOfRangeError:
                loss = np.mean(losses)
                break
        return loss

    def evaluate_shaping(self):
        pass

    def train(self):
        batch = self.sample_batch()

        self.training_step.assign_add(1)

        with tf.GradientTape() as critic_tape, tf.GradientTape() as actor_tape:
            actor_loss, critic_loss = self._compute_actorcritic_loss(
                tf.convert_to_tensor(batch["o"]),
                tf.convert_to_tensor(batch["o_2"]),
                tf.convert_to_tensor(batch["g"]),
                tf.convert_to_tensor(batch["g_2"]),
                tf.convert_to_tensor(batch["u"]),
                tf.convert_to_tensor(batch["r"]),
            )

        critic_grad = critic_tape.gradient(critic_loss, self.main_critic.trainable_variables)
        self.critic_adam.apply_gradients(zip(critic_grad, self.main_critic.trainable_variables))

        if (not self.use_td3) or self.training_step % 2 == 0:
            actor_grad = actor_tape.gradient(actor_loss, self.main_actor.trainable_variables)
            self.actor_adam.apply_gradients(zip(actor_grad, self.main_actor.trainable_variables))

        return critic_loss, actor_loss

    def initialize_target_net(self):
        for t, m in zip(self.target_actor.variables, self.main_actor.variables):
            t.assign(m)
        for t, m in zip(self.target_critic.variables, self.main_critic.variables):
            t.assign(m)
        if self.use_td3:
            for t, m in zip(self.target_critic_twin.variables, self.main_critic_twin.variables):
                t.assign(m)

    def update_target_net(self):
        for t, m in zip(self.target_actor.variables, self.main_actor.variables):
            t.assign(self.polyak * t + (1.0 - self.polyak) * m)
        for t, m in zip(self.target_critic.variables, self.main_critic.variables):
            t.assign(self.polyak * t + (1.0 - self.polyak) * m)
        if self.use_td3:
            for t, m in zip(self.target_critic_twin.variables, self.main_critic_twin.variables):
                t.assign(self.polyak * t + (1.0 - self.polyak) * m)

    def logs(self, prefix=""):
        logs = []
        logs.append((prefix + "stats_o/mean", np.mean(self.o_stats.mean_tf.numpy())))
        logs.append((prefix + "stats_o/std", np.mean(self.o_stats.std_tf.numpy())))
        if self.dimg != (0,):
            logs.append((prefix + "stats_g/mean", np.mean(self.g_stats.mean_tf.numpy())))
            logs.append((prefix + "stats_g/std", np.mean(self.g_stats.std_tf.numpy())))
        return logs

    def _create_memory(self):
        # buffer shape
        buffer_shapes = {}
        if self.fix_T:
            buffer_shapes["o"] = (self.eps_length + 1, *self.dimo)
            buffer_shapes["u"] = (self.eps_length, *self.dimu)
            buffer_shapes["r"] = (self.eps_length, 1)
            if self.dimg != (0,):  # for multigoal environment - or states that do not change over episodes.
                buffer_shapes["ag"] = (self.eps_length + 1, *self.dimg)
                buffer_shapes["g"] = (self.eps_length, *self.dimg)
            for key, val in self.input_dims.items():
                if key.startswith("info"):
                    buffer_shapes[key] = (self.eps_length, *val)
        else:
            buffer_shapes["o"] = self.dimo
            buffer_shapes["o_2"] = self.dimo
            buffer_shapes["u"] = self.dimu
            buffer_shapes["r"] = (1,)
            if self.dimg != (0,):  # for multigoal environment - or states that do not change over episodes.
                buffer_shapes["ag"] = self.dimg
                buffer_shapes["g"] = self.dimg
                buffer_shapes["ag_2"] = self.dimg
                buffer_shapes["g_2"] = self.dimg
            for key, val in self.input_dims.items():
                if key.startswith("info"):
                    buffer_shapes[key] = val
            # need the "done" signal for restarting from training
            buffer_shapes["done"] = (1,)
        # initialize replay buffer(s)
        if self.fix_T:
            self.replay_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
            if self.demo_strategy != "none" or self.sample_demo_buffer:
                self.demo_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
        else:
            self.replay_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)
            if self.demo_strategy != "none" or self.sample_demo_buffer:
                self.demo_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)

    def _create_network(self):
        # Normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)

        # Actor Critic Models
        # main networks
        self.main_actor = Actor(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes, noise=False
        )
        self.main_critic = Critic(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes
        )
        if self.use_td3:
            self.main_critic_twin = Critic(
                dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes
            )
        # target networks
        self.target_actor = Actor(
            dimo=self.dimo,
            dimg=self.dimg,
            dimu=self.dimu,
            max_u=self.max_u,
            layer_sizes=self.layer_sizes,
            noise=self.use_td3,
        )
        self.target_critic = Critic(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes
        )
        if self.use_td3:
            self.target_critic_twin = Critic(
                dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes
            )

        # Input Dataset that loads from demonstration buffer.
        demo_shapes = {}
        demo_shapes["o"] = self.dimo
        if self.dimg != (0,):
            demo_shapes["g"] = self.dimg
        demo_shapes["u"] = self.dimu
        max_num_transitions = self.num_demo * self.eps_length * 100  # TODO: this sets an upper bound of the dataset

        def generate_demo_data():
            demo_data = self.demo_buffer.sample()
            num_transitions = demo_data["u"].shape[0]
            assert all([demo_data[k].shape[0] == num_transitions for k in demo_data.keys()])
            for i in range(num_transitions):
                yield {k: demo_data[k][i] for k in demo_shapes.keys()}

        demo_dataset = (
            tf.data.Dataset.from_generator(
                generate_demo_data, output_types={k: tf.float32 for k in demo_shapes.keys()}, output_shapes=demo_shapes
            )
            .take(max_num_transitions)
            .shuffle(max_num_transitions)
            .repeat(1)
        )

        # Add shaping reward
        # normalizer for goal and observation.
        self.demo_o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
        self.demo_g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)
        # potential function approximator, nf or gan
        if self.demo_strategy == "nf":
            self.demo_shaping = EnsNFDemoShaping(
                gamma=self.gamma,
                max_num_transitions=max_num_transitions,
                batch_size=self.shaping_params["batch_size"],
                demo_dataset=demo_dataset,
                o_stats=self.demo_o_stats,
                g_stats=self.demo_g_stats,
                **self.shaping_params["nf"]
            )
        elif self.demo_strategy == "gan":
            self.demo_shaping = EnsGANDemoShaping(
                gamma=self.gamma,
                max_num_transitions=max_num_transitions,
                batch_size=self.shaping_params["batch_size"],
                demo_dataset=demo_dataset,
                o_stats=self.demo_o_stats,
                g_stats=self.demo_g_stats,
                **self.shaping_params["gan"]
            )
        else:
            self.demo_shaping = None

        self.critic_adam = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
        self.actor_adam = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)

        # extra variable for guarding the training
        self.training_step = tf.Variable(1, trainable=False, dtype=tf.int32)

        # Initialize all variables
        self.initialize_target_net()

    @tf.function
    def _compute_actorcritic_loss(self, o, o_2, g, g_2, u, r):
        # normalized o g
        norm_o = self.o_stats.normalize(o)
        norm_o_2 = self.o_stats.normalize(o_2)
        norm_g = self.g_stats.normalize(g)
        norm_g_2 = self.g_stats.normalize(g_2)

        # actor output
        main_pi = self.main_actor(o=norm_o, g=norm_g)
        # critic output
        main_q = self.main_critic(o=norm_o, g=norm_g, u=u)
        main_q_pi = self.main_critic(o=norm_o, g=norm_g, u=main_pi)
        if self.use_td3:
            main_q2 = self.main_critic_twin(o=norm_o, g=norm_g, u=u)
            main_q2_pi = self.main_critic_twin(o=norm_o, g=norm_g, u=main_pi)

        # actor output
        target_pi = self.target_actor(o=norm_o_2, g=norm_g_2)
        # critic output
        target_q = self.target_critic(o=norm_o_2, g=norm_g_2, u=u)
        target_q_pi = self.target_critic(o=norm_o_2, g=norm_g_2, u=target_pi)
        if self.use_td3:
            target_q2 = self.target_critic_twin(o=norm_o_2, g=norm_g_2, u=u)
            target_q2_pi = self.target_critic_twin(o=norm_o_2, g=norm_g_2, u=target_pi)

        # Q Loss
        # immediate reward
        bellman_target = r
        # demo shaping reward
        if self.demo_shaping != None:
            bellman_target += self.demo_shaping.reward(o=o, g=g, u=u, o_2=o_2, g_2=g_2, u_2=target_pi)
        # ddpg or td3 target with or without clipping
        if self.use_td3:
            bellman_target += self.gamma * tf.minimum(target_q_pi, target_q2_pi)
        else:
            bellman_target += self.gamma * target_q_pi
        assert bellman_target.shape[1] == 1
        # final ddpg or td3 loss
        if self.use_td3:
            rl_bellman_1 = tf.square(tf.stop_gradient(bellman_target) - main_q)
            rl_bellman_2 = tf.square(tf.stop_gradient(bellman_target) - main_q2)
            rl_bellman = rl_bellman_1 + rl_bellman_2
        else:
            rl_bellman = tf.square(tf.stop_gradient(bellman_target) - main_q)
        # whether or not to train the critic on demo reward (if sample from demonstration buffer)
        if self.sample_demo_buffer and not self.use_demo_reward:
            # mask off entries from demonstration dataset
            mask = np.concatenate((np.ones(self.batch_size), np.zeros(self.batch_size_demo)), axis=0)
            rl_bellman = tf.boolean_mask(rl_bellman, mask)
        q_loss = tf.reduce_mean(rl_bellman)

        # Pi Loss
        if self.demo_strategy == "bc":
            assert self.sample_demo_buffer, "must sample from the demonstration buffer to use behavior cloning"
            # primary loss scaled by it's respective weight prm_loss_weight
            pi_loss = -self.bc_params["prm_loss_weight"] * tf.reduce_mean(main_pi)
            # L2 loss on action values scaled by the same weight prm_loss_weight
            pi_loss += (
                self.bc_params["prm_loss_weight"] * self.action_l2 * tf.reduce_mean(tf.square(main_pi / self.max_u))
            )
            # define the cloning loss on the actor's actions only on the samples which adhere to the above masks
            mask = np.concatenate((np.zeros(self.batch_size), np.ones(self.batch_size_demo)), axis=0)
            actor_pi = tf.boolean_mask((main_pi), mask)
            demo_pi = tf.boolean_mask((u), mask)
            if self.bc_params["q_filter"]:
                q_filter_mask = tf.reshape(tf.boolean_mask(main_q > main_q_pi, mask), [-1])
                # use to be tf.reduce_sum, however, use tf.reduce_mean makes the loss function independent from number
                # of demonstrations
                cloning_loss = tf.reduce_mean(
                    tf.square(
                        tf.boolean_mask(actor_pi, q_filter_mask, axis=0)
                        - tf.boolean_mask(demo_pi, q_filter_mask, axis=0)
                    )
                )
            else:
                # use to be tf.reduce_sum, however, use tf.reduce_mean makes the loss function independent from number
                # of demonstrations
                cloning_loss = tf.reduce_mean(tf.square(actor_pi - demo_pi))
            # adding the cloning loss to the actor loss as an auxilliary loss scaled by its weight aux_loss_weight
            pi_loss += self.bc_params["aux_loss_weight"] * cloning_loss
        elif self.demo_shaping != None:  # any type of shaping method
            pi_loss = -tf.reduce_mean(main_q_pi)
            pi_loss += -tf.reduce_mean(self.demo_shaping.potential(o=o, g=g, u=main_pi))
            pi_loss += self.action_l2 * tf.reduce_mean(tf.square(main_pi / self.max_u))
        else:  # not training with demonstrations
            pi_loss = -tf.reduce_mean(main_q_pi)
            pi_loss += self.action_l2 * tf.reduce_mean(tf.square(main_pi / self.max_u))

        return pi_loss, q_loss

    def _update_stats(self, episode_batch):
        # add transitions to normalizer
        if self.fix_T:
            episode_batch["o_2"] = episode_batch["o"][:, 1:, :]
            if self.dimg != (0,):
                episode_batch["ag_2"] = episode_batch["ag"][:, :, :]
                episode_batch["g_2"] = episode_batch["g"][:, :, :]
            num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
            transitions = self.replay_buffer.sample_transitions(episode_batch, num_normalizing_transitions)
        else:
            transitions = episode_batch.copy()

        self.o_stats.update(transitions["o"])
        if self.dimg != (0,):
            self.g_stats.update(transitions["g"])

    def _update_demo_stats(self, episode_batch):
        # add transitions to normalizer
        if self.fix_T:
            episode_batch["o_2"] = episode_batch["o"][:, 1:, :]
            if self.dimg != (0,):
                episode_batch["ag_2"] = episode_batch["ag"][:, :, :]
                episode_batch["g_2"] = episode_batch["g"][:, :, :]
            num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
            transitions = self.demo_buffer.sample_transitions(episode_batch, num_normalizing_transitions)
        else:
            transitions = episode_batch.copy()

        self.demo_o_stats.update(transitions["o"])
        if self.dimg != (0,):
            self.demo_g_stats.update(transitions["g"])

    def __getstate__(self):
        """
        Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        state = {k: v for k, v in self.init_args.items() if not k == "self"}
        # state["tf"] = self.sess.run(
        #     [x for x in tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)]
        # )
        return state

    def __setstate__(self, state):
        # stored_vars = state.pop("tf")
        self.__init__(**state)
        # vars = [x for x in tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)]
        # assert len(vars) == len(stored_vars)
        # node = [tf.assign(var, val) for var, val in zip(vars, stored_vars)]
        # self.sess.run(node)
