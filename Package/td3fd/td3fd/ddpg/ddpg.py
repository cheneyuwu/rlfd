import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from td3fd.ddpg.actorcritic_network import Actor, Critic
from td3fd.ddpg.demo_shaping import EnsGANDemoShaping, EnsNFDemoShaping
from td3fd.memory import RingReplayBuffer, UniformReplayBuffer, iterbatches
from td3fd.ddpg.normalizer import Normalizer

tfd = tfp.distributions



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
        assert self.demo_strategy in ["none", "bc", "gan", "nf"]
        self.bc_params = bc_params
        self.shaping_params = shaping_params
        self.gamma = gamma
        self.info = info

        # Prepare parameters
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        # Get a tf session
        self.sess = tf.compat.v1.get_default_session()
        assert self.sess != None, "must have a default session before creating DDPG"
        with tf.compat.v1.variable_scope(scope):
            self.scope = tf.compat.v1.get_variable_scope()
            self._create_memory()
            self._create_network()
        self.saver = tf.compat.v1.train.Saver(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        )

    def get_actions(self, o, g, compute_q=False):
        # values to compute
        vals = [self.main_pi_tf, self.main_q_pi_tf]
        if self.demo_shaping != None:
            vals.append(self.demo_actor_shaping)
        # feed
        feed = {}
        feed[self.inputs_tf["o"]] = o.reshape(-1, *self.dimo)
        if self.dimg != (0,):
            feed[self.inputs_tf["g"]] = g.reshape(-1, *self.dimg)
        # compute
        ret = self.sess.run(vals, feed_dict=feed)
        # post processing
        if self.demo_shaping != None:
            ret[2] = ret[2] + ret[1]
        else:
            ret.append(ret[1])
        # return u only if compute_q is set to false
        if compute_q:
            return ret
        else:
            return ret[0]

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
        losses = np.empty(0)
        demo_data = self.demo_buffer.sample()
        if self.dimg != (0,):
            for (o, g, u) in iterbatches(
                (demo_data["o"], demo_data["g"], demo_data["u"]), batch_size=self.shaping_params["batch_size"]
            ):
                feed = {self.demo_inputs_tf["o"]: o, self.demo_inputs_tf["g"]: g, self.demo_inputs_tf["u"]: u}
                loss = self.demo_shaping.train(sess=self.sess, feed_dict=feed)
                losses = np.append(losses, loss)
        else:
            for (o, u) in iterbatches((demo_data["o"], demo_data["u"]), batch_size=self.shaping_params["batch_size"]):
                feed = {self.demo_inputs_tf["o"]: o, self.demo_inputs_tf["u"]: u}
                loss = self.demo_shaping.train(sess=self.sess, feed_dict=feed)
                losses = np.append(losses, loss)
        return np.mean(losses)

    def train_shaping_policy(self, episode):
        assert self.demo_strategy in ["nf", "gan"]
        # train normalizing flow or gan for 1 epoch
        losses = np.empty(0)
        demo_data = self.demo_buffer.sample()
        episode["o"] = episode["o"][:, :-1, ...].reshape(-1, *self.dimo)
        if self.dimg != (0,):
            episode["g"] = episode["g"].reshape(-1, *self.dimg)
        episode["u"] = episode["u"].reshape(-1, *self.dimu)
        assert episode["u"].shape == demo_data["u"].shape
        if self.dimg != (0,):
            for (o, g, u, po, pg, pu) in iterbatches(
                (demo_data["o"], demo_data["g"], demo_data["u"], episode["o"], episode["g"], episode["u"]),
                batch_size=self.shaping_params["batch_size"],
            ):
                feed = {
                    self.demo_inputs_tf["o"]: o,
                    self.demo_inputs_tf["g"]: g,
                    self.demo_inputs_tf["u"]: u,
                    self.policy_inputs_tf["o"]: po,
                    self.policy_inputs_tf["g"]: pg,
                    self.policy_inputs_tf["u"]: pu,
                }
                loss = self.demo_shaping.train_on_policy(sess=self.sess, feed_dict=feed)
                losses = np.append(losses, loss)
        else:
            for (o, u, po, pu) in iterbatches(
                (demo_data["o"], demo_data["u"], episode["o"], episode["u"]),
                batch_size=self.shaping_params["batch_size"],
            ):
                feed = {
                    self.demo_inputs_tf["o"]: o,
                    self.demo_inputs_tf["u"]: u,
                    self.policy_inputs_tf["o"]: po,
                    self.policy_inputs_tf["u"]: pu,
                }
                loss = self.demo_shaping.train_on_policy(sess=self.sess, feed_dict=feed)
                losses = np.append(losses, loss)
        return np.mean(losses)

    def evaluate_shaping(self):
        pass
        # TODO: future work, try vision based RL
        # images = self.demo_shaping.evaluate(self.sess)
        # images = ((images + 1.0) * 127.5).astype(np.int32)
        # import matplotlib.pyplot as plt

        # plt.imshow(images[0])
        # plt.show()

    def train(self):
        batch = self.sample_batch()

        self.training_step = self.sess.run(self.update_training_step_op)

        feed = {self.inputs_tf[k]: batch[k] for k in self.inputs_tf.keys()}
        if self.use_td3 and self.training_step % 2 == 1:
            critic_loss, actor_loss, _ = self.sess.run(
                [self.q_loss_tf, self.pi_loss_tf, self.q_update_op], feed_dict=feed
            )
        else:
            critic_loss, actor_loss, _, _ = self.sess.run(
                [self.q_loss_tf, self.pi_loss_tf, self.q_update_op, self.pi_update_op], feed_dict=feed
            )

        return critic_loss, actor_loss

    def initialize_target_net(self):
        self.sess.run(self.initialize_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def logs(self, prefix=""):
        logs = []
        logs.append((prefix + "stats_o/mean", np.mean(self.sess.run([self.o_stats.mean_tf]))))
        logs.append((prefix + "stats_o/std", np.mean(self.sess.run([self.o_stats.std_tf]))))
        if self.dimg != (0,):
            logs.append((prefix + "stats_g/mean", np.mean(self.sess.run([self.g_stats.mean_tf]))))
            logs.append((prefix + "stats_g/std", np.mean(self.sess.run([self.g_stats.std_tf]))))
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
            
            self.replay_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
            if self.demo_strategy != "none" or self.sample_demo_buffer:
                self.demo_buffer = UniformReplayBuffer(buffer_shapes, self.buffer_size, self.eps_length)
        
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

            self.replay_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)
            if self.demo_strategy != "none" or self.sample_demo_buffer:
                self.demo_buffer = RingReplayBuffer(buffer_shapes, self.buffer_size)

    def _create_network(self):
        # Inputs to DDPG
        self.inputs_tf = {}
        self.inputs_tf["o"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimo))
        self.inputs_tf["o_2"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimo))
        self.inputs_tf["u"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimu))
        self.inputs_tf["r"] = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        if self.dimg != (0,):
            self.inputs_tf["g"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimg))
            self.inputs_tf["g_2"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimg))

        # create a variable for each o, g, u key in the inputs_tf dict
        input_o_tf = self.inputs_tf["o"]
        input_o_2_tf = self.inputs_tf["o_2"]
        input_g_tf = self.inputs_tf["g"] if self.dimg != (0,) else None
        input_g_2_tf = self.inputs_tf["g_2"] if self.dimg != (0,) else None
        input_u_tf = self.inputs_tf["u"]

        # Normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        # normalized o g
        norm_input_o_tf = self.o_stats.normalize(input_o_tf)
        norm_input_o_2_tf = self.o_stats.normalize(input_o_2_tf)
        norm_input_g_tf = self.g_stats.normalize(input_g_tf) if self.dimg != (0,) else None
        norm_input_g_2_tf = self.g_stats.normalize(input_g_2_tf) if self.dimg != (0,) else None

        # Actor Critic Models
        # main networks
        self.main_actor = Actor(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes
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
        )
        self.target_critic = Critic(
            dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes
        )
        if self.use_td3:
            self.target_critic_twin = Critic(
                dimo=self.dimo, dimg=self.dimg, dimu=self.dimu, max_u=self.max_u, layer_sizes=self.layer_sizes
            )
        # actor output
        self.main_pi_tf = self.main_actor(o=norm_input_o_tf, g=norm_input_g_tf)
        self.target_pi_tf = self.target_actor(o=norm_input_o_2_tf, g=norm_input_g_2_tf)
        if self.use_td3:
            # TODO pass noise scale as parameter
            noise = tfd.Normal(loc=0.0, scale=0.2).sample(tf.shape(self.target_pi_tf))
            noise = tf.clip_by_value(noise, -0.5, 0.5) * self.max_u
            self.target_pi_tf = tf.clip_by_value(self.target_pi_tf + noise, -self.max_u, self.max_u)
        # critic output
        self.main_q_tf = self.main_critic(o=norm_input_o_tf, g=norm_input_g_tf, u=input_u_tf)
        self.main_q_pi_tf = self.main_critic(o=norm_input_o_tf, g=norm_input_g_tf, u=self.main_pi_tf)
        self.target_q_tf = self.target_critic(o=norm_input_o_2_tf, g=norm_input_g_2_tf, u=input_u_tf)
        self.target_q_pi_tf = self.target_critic(o=norm_input_o_2_tf, g=norm_input_g_2_tf, u=self.target_pi_tf)
        if self.use_td3:
            self.main_q2_tf = self.main_critic_twin(o=norm_input_o_tf, g=norm_input_g_tf, u=input_u_tf)
            self.main_q2_pi_tf = self.main_critic_twin(o=norm_input_o_tf, g=norm_input_g_tf, u=self.main_pi_tf)
            self.target_q2_tf = self.target_critic_twin(o=norm_input_o_2_tf, g=norm_input_g_2_tf, u=input_u_tf)
            self.target_q2_pi_tf = self.target_critic_twin(
                o=norm_input_o_2_tf, g=norm_input_g_2_tf, u=self.target_pi_tf
            )

        # Demonstration Inputs
        self.demo_inputs_tf = {}
        self.demo_inputs_tf["o"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimo))
        self.demo_inputs_tf["u"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimu))
        if self.dimg != (0,):
            self.demo_inputs_tf["g"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimg))
        # Policy Inputs
        self.policy_inputs_tf = {}
        self.policy_inputs_tf["o"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimo))
        self.policy_inputs_tf["u"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimu))
        if self.dimg != (0,):
            self.policy_inputs_tf["g"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *self.dimg))

        # Add shaping reward
        # normalizer for goal and observation.
        self.demo_o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        self.demo_g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        # potential function approximator, nf or gan
        if self.demo_strategy == "nf":
            self.demo_shaping = EnsNFDemoShaping(
                demo_inputs_tf=self.demo_inputs_tf,
                gamma=self.gamma,
                o_stats=self.demo_o_stats,
                g_stats=self.demo_g_stats,
                **self.shaping_params["nf"]
            )
        elif self.demo_strategy == "gan":
            self.demo_shaping = EnsGANDemoShaping(
                demo_inputs_tf=self.demo_inputs_tf,
                policy_inputs_tf=self.policy_inputs_tf,
                gamma=self.gamma,
                o_stats=self.demo_o_stats,
                g_stats=self.demo_g_stats,
                **self.shaping_params["gan"]
            )
        else:
            self.demo_shaping = None

        if self.demo_shaping != None:
            self.demo_critic_shaping = self.demo_shaping.reward(
                o=input_o_tf, g=input_g_tf, u=input_u_tf, o_2=input_o_2_tf, g_2=input_g_2_tf, u_2=self.target_pi_tf
            )
            self.demo_actor_shaping = self.demo_shaping.potential(o=input_o_tf, g=input_g_tf, u=self.main_pi_tf)
            self.demo_shaping_check = self.demo_shaping.potential(o=input_o_tf, g=input_g_tf, u=input_u_tf)

        # Critic loss
        # immediate reward
        target_tf = self.inputs_tf["r"]
        # demo shaping reward
        if self.demo_shaping != None:
            target_tf += self.demo_critic_shaping
        # ddpg or td3 target with or without clipping
        if self.use_td3:
            target_tf += self.gamma * tf.minimum(self.target_q_pi_tf, self.target_q2_pi_tf)
        else:
            target_tf += self.gamma * self.target_q_pi_tf
        assert target_tf.shape[1] == 1
        # final ddpg or td3 loss
        if self.use_td3:
            rl_bellman_1_tf = tf.square(tf.stop_gradient(target_tf) - self.main_q_tf)
            rl_bellman_2_tf = tf.square(tf.stop_gradient(target_tf) - self.main_q2_tf)
            rl_bellman_tf = (rl_bellman_1_tf + rl_bellman_2_tf) / 2.0
        else:
            rl_bellman_tf = tf.square(tf.stop_gradient(target_tf) - self.main_q_tf)
        # whether or not to train the critic on demo reward (if sample from demonstration buffer)
        if self.sample_demo_buffer and not self.use_demo_reward:
            # mask off entries from demonstration dataset
            mask = np.concatenate((np.ones(self.batch_size), np.zeros(self.batch_size_demo)), axis=0)
            rl_bellman_tf = tf.boolean_mask(rl_bellman_tf, mask)
        rl_loss_tf = tf.reduce_mean(rl_bellman_tf)
        self.q_loss_tf = rl_loss_tf

        # Actor Loss
        if self.demo_strategy == "bc":
            assert self.sample_demo_buffer, "must sample from the demonstration buffer to use behavior cloning"
            # primary loss scaled by it's respective weight prm_loss_weight
            pi_loss_tf = -self.bc_params["prm_loss_weight"] * tf.reduce_mean(self.main_q_pi_tf)
            # L2 loss on action values scaled by the same weight prm_loss_weight
            pi_loss_tf += (
                self.bc_params["prm_loss_weight"]
                * self.action_l2
                * tf.reduce_mean(tf.square(self.main_pi_tf / self.max_u))
            )
            # define the cloning loss on the actor's actions only on the samples which adhere to the above masks
            mask = np.concatenate((np.zeros(self.batch_size), np.ones(self.batch_size_demo)), axis=0)
            actor_pi_tf = tf.boolean_mask((self.main_pi_tf), mask)
            demo_pi_tf = tf.boolean_mask((input_u_tf), mask)
            if self.bc_params["q_filter"]:
                q_filter_mask = tf.reshape(tf.boolean_mask(self.main_q_tf > self.main_q_pi_tf, mask), [-1])
                # use to be tf.reduce_sum, however, use tf.reduce_mean makes the loss function independent from number
                # of demonstrations
                cloning_loss_tf = tf.reduce_mean(
                    tf.square(
                        tf.boolean_mask(actor_pi_tf, q_filter_mask, axis=0)
                        - tf.boolean_mask(demo_pi_tf, q_filter_mask, axis=0)
                    )
                )
            else:
                # use to be tf.reduce_sum, however, use tf.reduce_mean makes the loss function independent from number
                # of demonstrations
                cloning_loss_tf = tf.reduce_mean(tf.square(actor_pi_tf - demo_pi_tf))
            # adding the cloning loss to the actor loss as an auxilliary loss scaled by its weight aux_loss_weight
            pi_loss_tf += self.bc_params["aux_loss_weight"] * cloning_loss_tf
        elif self.demo_shaping != None:  # any type of shaping method
            pi_loss_tf = -tf.reduce_mean(self.main_q_pi_tf)
            pi_loss_tf += -tf.reduce_mean(self.demo_actor_shaping)
            pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main_pi_tf / self.max_u))
        else:  # not training with demonstrations
            pi_loss_tf = -tf.reduce_mean(self.main_q_pi_tf)
            pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main_pi_tf / self.max_u))
        self.pi_loss_tf = pi_loss_tf

        self.q_update_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.q_lr).minimize(
            self.q_loss_tf, var_list=self.main_critic.trainable_variables
        )
        self.pi_update_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.pi_lr).minimize(
            self.pi_loss_tf, var_list=self.main_actor.trainable_variables
        )

        # Polyak averaging
        hard_copy_func = lambda v: v[0].assign(v[1])
        soft_copy_func = lambda v: v[0].assign(self.polyak * v[0] + (1.0 - self.polyak) * v[1])
        self.initialize_target_net_op = (
            list(map(hard_copy_func, zip(self.target_actor.variables, self.main_actor.variables)))
            + list(map(hard_copy_func, zip(self.target_critic.variables, self.main_critic.variables)))
            + (
                list(map(hard_copy_func, zip(self.target_critic_twin.variables, self.main_critic_twin.variables)))
                if self.use_td3
                else []
            )
        )
        self.update_target_net_op = (
            list(map(soft_copy_func, zip(self.target_actor.variables, self.main_actor.variables)))
            + list(map(soft_copy_func, zip(self.target_critic.variables, self.main_critic.variables)))
            + (
                list(map(soft_copy_func, zip(self.target_critic_twin.variables, self.main_critic_twin.variables)))
                if self.use_td3
                else []
            )
        )

        # extra variable for guarding the training
        self.training_step_tf = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.update_training_step_op = self.training_step_tf.assign_add(1)

        # Initialize all variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.initialize_target_net()
        self.training_step = self.sess.run(self.training_step_tf)

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
        state["tf"] = self.sess.run(
            [x for x in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)]
        )
        return state

    def __setstate__(self, state):
        stored_vars = state.pop("tf")
        self.__init__(**state)
        vars = [x for x in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)]
        assert len(vars) == len(stored_vars)
        node = [tf.compat.v1.assign(var, val) for var, val in zip(vars, stored_vars)]
        self.sess.run(node)
