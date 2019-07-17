import numpy as np
import tensorflow as tf

from yw.tool import logger

from yw.ddpg_main.mpi_adam import MpiAdam
from yw.ddpg_main.normalizer import Normalizer
from yw.ddpg_main.actor_critic import ActorCritic
from yw.ddpg_main.replay_buffer import HERReplayBuffer, UniformReplayBuffer
from yw.ddpg_main.demo_shaping import (
    ManualDemoShaping,
    GaussianDemoShaping,
    NormalizingFlowDemoShaping,
    MAFDemoShaping,
)
from yw.util.tf_util import flatten_grads

# for query
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from yw.flow.query import visualize_query


class DDPG(object):
    def __init__(
        self,
        input_dims,
        use_td3,
        layer_sizes,
        polyak,
        buffer_size,
        batch_size,
        Q_lr,
        pi_lr,
        norm_eps,
        norm_clip,
        max_u,
        action_l2,
        clip_obs,
        scope,
        T,
        clip_pos_returns,
        clip_return,
        demo_strategy,
        num_demo,
        q_filter,
        batch_size_demo,
        prm_loss_weight,
        aux_loss_weight,
        replay_strategy,
        gamma,
        info,
        **kwargs
    ):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Added functionality to use demonstrations for training to Overcome exploration problem.

        Args:
            # Environment I/O and Config
            max_u              (float)        - maximum action magnitude, i.e. actions are in [-max_u, max_u]
            clip_obs           (float)        - clip observations before normalization to be in [-clip_obs, clip_obs]
            T                  (int)          - the time horizon for rollouts
            clip_pos_returns   (boolean)      - whether or not positive returns should be clipped (i.e. clip to 0)
            clip_return        (float)        - clip returns to be in [-clip_return, clip_return]
            # Normalizer
            norm_eps           (float)        - a small value used in the normalizer to avoid numerical instabilities
            norm_clip          (float)        - normalized inputs are clipped to be in [-norm_clip, norm_clip]
            # NN Configuration
            scope              (str)          - the scope used for the TensorFlow graph
            input_dims         (dict of ints) - dimensions for the observation (o), the goal (g), and the actions (u)
            layer_sizes        (list of ints) - number of units in each hidden layers
            reuse              (boolean)      - whether or not the networks should be reused
            # Replay Buffer
            buffer_size        (int)          - number of transitions that are stored in the replay buffer
            # HER
            replay_strategy    (dict)         - strategy to use and function that samples from the replay buffer
            # Dual Network Set
            polyak             (float)        - coefficient for Polyak-averaging of the target network
            # Training
            batch_size         (int)          - batch size for training
            Q_lr               (float)        - learning rate for the Q (critic) network
            pi_lr              (float)        - learning rate for the pi (actor) network
            action_l2          (float)        - coefficient for L2 penalty on the actions
            gamma              (float)        - gamma used for Q learning updates
            # Use demonstration to shape critic or actor
            demo_strategy      (str)          - whether or not to use demonstration with different strategies
            num_demo           (int)          - Number of episodes in to be used in the demonstration buffer
            batch_size_demo    (int)          - number of samples to be used from the demonstrations buffer, per mpi thread
            q_filter           (boolean)      - whether or not a filter on the q value update should be used when training with demonstartions
            prm_loss_weight    (float)        - Weight corresponding to the primary loss
            aux_loss_weight    (float)        - Weight corresponding to the auxilliary loss also called the cloning loss

        """
        # Store initial args passed into the function
        self.init_args = locals()

        # Parameters
        self.input_dims = input_dims
        self.use_td3 = use_td3
        self.layer_sizes = layer_sizes
        self.polyak = polyak
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.Q_lr = Q_lr
        self.pi_lr = pi_lr
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.max_u = max_u
        self.action_l2 = action_l2
        self.clip_obs = clip_obs
        self.scope = scope
        self.T = T
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.demo_strategy = demo_strategy
        self.num_demo = num_demo
        self.q_filter = q_filter
        self.batch_size_demo = batch_size_demo
        self.prm_loss_weight = prm_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.replay_strategy = replay_strategy
        self.gamma = gamma
        self.info = info

        # Prepare parameters
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        # Configure the replay buffer
        # buffer shape
        buffer_shapes = {}
        buffer_shapes["o"] = (self.T + 1, self.dimo)
        buffer_shapes["u"] = (self.T, self.dimu)
        buffer_shapes["r"] = (self.T, 1)
        if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
            buffer_shapes["ag"] = (self.T + 1, self.dimg)
            buffer_shapes["g"] = (self.T, self.dimg)
        # add extra information
        for key, val in input_dims.items():
            if key.startswith("info"):
                buffer_shapes[key] = (self.T, *(tuple([val]) if val > 0 else tuple()))
        # initialize replay buffer(s)
        if not self.replay_strategy:
            pass
        elif self.replay_strategy["strategy"] == "her":
            self.replay_buffer = HERReplayBuffer(
                buffer_shapes, self.buffer_size, self.T, **self.replay_strategy["args"]
            )
            if self.demo_strategy != "none":
                self.demo_buffer = HERReplayBuffer(
                    buffer_shapes, self.buffer_size, self.T, **self.replay_strategy["args"]
                )
        else:
            self.replay_buffer = UniformReplayBuffer(
                buffer_shapes, self.buffer_size, self.T, **self.replay_strategy["args"]
            )
            if self.demo_strategy != "none":
                self.demo_buffer = UniformReplayBuffer(
                    buffer_shapes, self.buffer_size, self.T, **self.replay_strategy["args"]
                )

        # Create the DDPG agent
        with tf.variable_scope(self.scope):
            self._create_network()

    def get_actions(self, o, g, noise_eps=0.0, random_eps=0.0, compute_Q=False):

        # values to compute
        vals = [self.main_pi_tf, self.main_q_pi_tf]
        if self.demo_shaping != None:
            vals.append(self.demo_actor_shaping)
        # feed
        feed = {}
        o = self._preprocess_state(o)
        feed[self.inputs_tf["o"]] = o.reshape(-1, self.dimo)
        if self.dimg != 0:
            g = self._preprocess_state(g)
            feed[self.inputs_tf["g"]] = g.reshape(-1, self.dimg)
        # compute
        ret = self.sess.run(vals, feed_dict=feed)

        # action post-processing
        u = ret[0]
        u += noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (
            self._random_action(u.shape[0]) - u
        )  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if self.demo_shaping != None:
            ret[2] = ret[2] + ret[1]
        else:
            ret.append(ret[1])

        if compute_Q:
            return ret
        else:
            return ret[0]

    def init_demo_buffer(self, demo_file, update_stats=True):
        """ Initialize the demonstration buffer.
        """
        logger.info("Initializing demonstration buffer with {} episodes.".format(self.num_demo))

        demo_data = np.load(demo_file)  # load the demonstration data from data file
        assert self.num_demo <= demo_data["u"].shape[0], "No enough demonstration data!"

        episode_batch = {**demo_data}
        for key in episode_batch.keys():
            episode_batch[key] = episode_batch[key][: self.num_demo]
        # the demo buffer should already have: o, u, r, ag, g and necessary infos.
        self.demo_buffer.store_episode(episode_batch)

        # feed demonstration data for norm shaping
        if self.demo_strategy == "norm":
            demo_data = self.demo_buffer.sample_all()
            demo_data["o"] = self._preprocess_state(demo_data["o"])
            if self.dimg != 0:
                demo_data["g"] = self._preprocess_state(demo_data["g"])
            for k in self.demo_inputs_tf.keys():  # o g u
                self.sess.run(tf.assign(self.demo_inputs_tf[k], demo_data[k]))

        if update_stats:
            logger.info("Updating stats using data from demostration buffer.")
            self._update_stats(episode_batch)

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.replay_buffer.store_episode(episode_batch)

        if update_stats:
            self._update_stats(episode_batch)

    def sample_batch(self):
        # use demonstration buffer to sample as well if demo flag is set TRUE
        if self.demo_strategy == "bc":
            transitions = {}
            transition_rollout = self.replay_buffer.sample(self.batch_size - self.batch_size_demo)
            transition_demo = self.demo_buffer.sample(self.batch_size_demo)
            assert transition_rollout.keys() == transition_demo.keys()
            for k in transition_rollout.keys():
                transitions[k] = np.concatenate((transition_rollout[k], transition_demo[k]))
        else:
            transitions = self.replay_buffer.sample(self.batch_size)  # otherwise only sample from primary buffer

        transitions["o"] = self._preprocess_state(transitions["o"])
        transitions["o_2"] = self._preprocess_state(transitions["o_2"])
        if self.dimg != 0:
            transitions["g"] = self._preprocess_state(transitions["g"])
            transitions["g_2"] = self._preprocess_state(transitions["g_2"])

        return transitions

    def train_shaping(self):
        # train normalizing flow
        loss = 0
        if self.demo_strategy == "maf":
            self.sess.run(self.demo_iter_tf.initializer)
            losses = np.empty(0)
            while True:
                try:
                    loss, _ = self.sess.run([self.demo_shaping.loss, self.demo_shaping.train_op])
                    losses = np.append(losses, loss)
                except tf.errors.OutOfRangeError:
                    loss = np.mean(losses)
                    break
        return loss

    def save_shaping_weights(self, path):
        # save the weights of the potential function
        self.demo_shaping.save_weights(self.sess, path)

    def load_shaping_weights(self, path):
        # save the weights of the potential function
        self.demo_shaping.load_weights(self.sess, path)

    def train(self):
        batch = self.sample_batch()
        feed = {self.inputs_tf[k]: batch[k] for k in self.inputs_tf.keys()}

        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run(
            [self.Q_loss_tf, self.pi_loss_tf, self.Q_grad_tf, self.pi_grad_tf], feed_dict=feed
        )

        # update Q every time
        self.Q_adam.update(Q_grad, self.Q_lr)
        # update Pi every other time
        if self.use_td3:
            if self.training_step % 2 == 0:
                self.pi_adam.update(pi_grad, self.pi_lr)
        else:
            self.pi_adam.update(pi_grad, self.pi_lr)

        self.training_step += 1

        # for debugging
        self.current_batch = batch

        return critic_loss, actor_loss

    def check_train(self):
        """ For debugging only
        """
        pass

    def init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def get_current_buffer_size(self):
        return self.replay_buffer.get_current_size()

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()

    def logs(self, prefix=""):
        logs = []
        logs += [("stats_o/mean", np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [("stats_o/std", np.mean(self.sess.run([self.o_stats.std])))]
        if self.dimg != 0:
            logs += [("stats_g/mean", np.mean(self.sess.run([self.g_stats.mean])))]
            logs += [("stats_g/std", np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not "" and not prefix.endswith("/"):
            return [(prefix + "/" + key, val) for key, val in logs]
        else:
            return logs

    def _create_network(self):

        self.sess = tf.get_default_session()
        assert self.sess != None, "must have a default session before creating DDPG"

        # inputs to ddpg
        self.inputs_tf = {}
        self.inputs_tf["o"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.inputs_tf["o_2"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.inputs_tf["u"] = tf.placeholder(tf.float32, shape=(None, self.dimu))
        self.inputs_tf["r"] = tf.placeholder(tf.float32, shape=(None, 1))
        if self.dimg != 0:
            self.inputs_tf["g"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
            self.inputs_tf["g_2"] = tf.placeholder(tf.float32, shape=(None, self.dimg))

        # create a normalizer for goal and observation.
        with tf.variable_scope("o_stats"):
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope("g_stats"):
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # models
        with tf.variable_scope("main"):
            self.main = ActorCritic(
                dimo=self.dimo,
                dimg=self.dimg,
                dimu=self.dimu,
                max_u=self.max_u,
                o_stats=self.o_stats,
                g_stats=self.g_stats,
                layer_sizes=self.layer_sizes,
                add_pi_noise=False,
            )
            # actor output
            self.main_pi_tf = self.main.actor(o=self.inputs_tf["o"], g=self.inputs_tf["g"] if self.dimg != 0 else None)
            # critic output
            self.main_q_tf = self.main.critic1(
                o=self.inputs_tf["o"], g=self.inputs_tf["g"] if self.dimg != 0 else None, u=self.inputs_tf["u"]
            )
            self.main_q_pi_tf = self.main.critic1(
                o=self.inputs_tf["o"], g=self.inputs_tf["g"] if self.dimg != 0 else None, u=self.main_pi_tf
            )
            if self.use_td3:
                self.main_q2_tf = self.main.critic2(
                    o=self.inputs_tf["o"], g=self.inputs_tf["g"] if self.dimg != 0 else None, u=self.inputs_tf["u"]
                )
                self.main_q2_pi_tf = self.main.critic2(
                    o=self.inputs_tf["o"], g=self.inputs_tf["g"] if self.dimg != 0 else None, u=self.main_pi_tf
                )

        with tf.variable_scope("target"):
            self.target = ActorCritic(
                dimo=self.dimo,
                dimg=self.dimg,
                dimu=self.dimu,
                max_u=self.max_u,
                o_stats=self.o_stats,
                g_stats=self.g_stats,
                layer_sizes=self.layer_sizes,
                add_pi_noise=self.use_td3,
            )
            # actor output
            self.target_pi_tf = self.target.actor(
                o=self.inputs_tf["o_2"], g=self.inputs_tf["g_2"] if self.dimg != 0 else None
            )
            # critic output
            self.target_q_tf = self.target.critic1(
                o=self.inputs_tf["o_2"], g=self.inputs_tf["g_2"] if self.dimg != 0 else None, u=self.inputs_tf["u"]
            )
            self.target_q_pi_tf = self.target.critic1(
                o=self.inputs_tf["o_2"], g=self.inputs_tf["g_2"] if self.dimg != 0 else None, u=self.target_pi_tf
            )
            if self.use_td3:
                self.target_q2_tf = self.target.critic2(
                    o=self.inputs_tf["o_2"], g=self.inputs_tf["g_2"] if self.dimg != 0 else None, u=self.inputs_tf["u"]
                )
                self.target_q2_pi_tf = self.target.critic2(
                    o=self.inputs_tf["o_2"], g=self.inputs_tf["g_2"] if self.dimg != 0 else None, u=self.target_pi_tf
                )

        assert len(self._vars("main")) == len(self._vars("target"))

        # Add shaping reward
        with tf.variable_scope("shaping"):
            if self.demo_strategy == "manual":
                self.demo_shaping = ManualDemoShaping(gamma=self.gamma)
            elif self.demo_strategy == "norm":
                num_transitions = self.num_demo * self.T
                self.demo_inputs_tf = {}
                self.demo_inputs_tf["o"] = tf.Variable(
                    initial_value=tf.zeros((num_transitions, self.dimo), dtype=tf.float32),
                    trainable=False,
                    dtype=tf.float32,
                )
                if self.dimg != 0:
                    self.demo_inputs_tf["g"] = tf.Variable(
                        initial_value=tf.zeros((num_transitions, self.dimg), dtype=tf.float32),
                        trainable=False,
                        dtype=tf.float32,
                    )
                self.demo_inputs_tf["u"] = tf.Variable(
                    initial_value=tf.zeros((num_transitions, self.dimu), dtype=tf.float32),
                    trainable=False,
                    dtype=tf.float32,
                )
                self.demo_shaping = GaussianDemoShaping(gamma=self.gamma, demo_inputs_tf=self.demo_inputs_tf)
            elif self.demo_strategy == "maf":
                # input dataset that loads from demo_buffer
                demo_shapes = {}
                demo_shapes["o"] = (self.dimo,)
                if self.dimg != 0:
                    demo_shapes["g"] = (self.dimg,)
                demo_shapes["u"] = (self.dimu,)
                num_transitions = self.num_demo * self.T

                def generate_demo_data():
                    demo_data = self.demo_buffer.sample_all()
                    demo_data["o"] = self._preprocess_state(demo_data["o"])
                    if self.dimg != 0:
                        demo_data["g"] = self._preprocess_state(demo_data["g"])
                    assert all([demo_data[k].shape[0] == num_transitions for k in demo_data.keys()])
                    for i in range(num_transitions):
                        yield {k: demo_data[k][i] for k in demo_shapes.keys()}

                demo_dataset = (
                    tf.data.Dataset.from_generator(
                        generate_demo_data,
                        output_types={k: tf.float32 for k in demo_shapes.keys()},
                        output_shapes=demo_shapes,
                    )
                    .take(num_transitions)
                    .shuffle(num_transitions)
                    .repeat(1)
                    .batch(128)
                )
                self.demo_iter_tf = demo_dataset.make_initializable_iterator()
                self.demo_inputs_tf = self.demo_iter_tf.get_next()

                # self.demo_shaping = NormalizingFlowDemoShaping(gamma=self.gamma, demo_inputs_tf=self.demo_inputs_tf)
                self.demo_shaping = MAFDemoShaping(gamma=self.gamma, demo_inputs_tf=self.demo_inputs_tf)
            else:
                self.demo_shaping = None

            if self.demo_shaping != None:
                self.demo_critic_shaping = self.demo_shaping.reward(
                    o=self.inputs_tf["o"],
                    g=self.inputs_tf["g"] if self.dimg != 0 else None,
                    u=self.inputs_tf["u"],
                    o_2=self.inputs_tf["o_2"],
                    g_2=self.inputs_tf["g_2"] if self.dimg != 0 else None,
                    u_2=self.target_pi_tf,
                )
                self.demo_actor_shaping = self.demo_shaping.potential(
                    o=self.inputs_tf["o"], g=self.inputs_tf["g"] if self.dimg != 0 else None, u=self.main_pi_tf
                )
                self.demo_shaping_check = self.demo_shaping.potential(
                    o=self.inputs_tf["o"], g=self.inputs_tf["g"] if self.dimg != 0 else None, u=self.inputs_tf["u"]
                )

        # Critic loss
        target_tf = self.inputs_tf["r"]
        # demo shaping
        if self.demo_shaping != None:
            target_tf += self.demo_critic_shaping
        # td3 target
        if self.use_td3:
            target_tf += self.gamma * tf.minimum(self.target_q_pi_tf, self.target_q2_pi_tf)
        else:
            target_tf += self.gamma * self.target_q_pi_tf
        # clipping
        clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else self.clip_return)
        target_tf = tf.clip_by_value(target_tf, *clip_range)

        assert target_tf.shape[1] == 1

        # td3 loss
        if self.use_td3:
            rl_bellman_1_tf = tf.square(tf.stop_gradient(target_tf) - self.main_q_tf)
            rl_bellman_2_tf = tf.square(tf.stop_gradient(target_tf) - self.main_q2_tf)
            rl_loss_tf = (tf.reduce_mean(rl_bellman_1_tf) + tf.reduce_mean(rl_bellman_2_tf)) / 2.0
        else:
            rl_bellman_tf = tf.square(tf.stop_gradient(target_tf) - self.main_q_tf)
            rl_loss_tf = tf.reduce_mean(rl_bellman_tf)
        self.Q_loss_tf = rl_loss_tf

        # Actor Loss
        if self.demo_strategy == "bc":
            # primary loss scaled by it's respective weight prm_loss_weight
            pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main_q_pi_tf)
            # L2 loss on action values scaled by the same weight prm_loss_weight
            pi_loss_tf += (
                self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main_pi_tf / self.max_u))
            )
            # define the cloning loss on the actor's actions only on the samples which adhere to the above masks
            mask = np.concatenate(
                (np.zeros(self.batch_size - self.batch_size_demo), np.ones(self.batch_size_demo)), axis=0
            )
            actor_pi_tf = tf.boolean_mask((self.main_pi_tf), mask)
            demo_pi_tf = tf.boolean_mask((self.inputs_tf["u"]), mask)
            if self.q_filter:
                q_filter_mask = tf.reshape(tf.boolean_mask(self.main_q_tf > self.main_q_pi_tf, mask), [-1])
                cloning_loss_tf = tf.reduce_sum(
                    tf.square(
                        tf.boolean_mask(actor_pi_tf, q_filter_mask, axis=0)
                        - tf.boolean_mask(demo_pi_tf, q_filter_mask, axis=0)
                    )
                )
            else:
                cloning_loss_tf = tf.reduce_sum(tf.square(actor_pi_tf - demo_pi_tf))
            # adding the cloning loss to the actor loss as an auxilliary loss scaled by its weight aux_loss_weight
            pi_loss_tf += self.aux_loss_weight * cloning_loss_tf

        elif self.demo_strategy != "none":
            pi_loss_tf = -tf.reduce_mean(self.main_q_pi_tf)
            pi_loss_tf += -tf.reduce_mean(self.demo_actor_shaping)
            pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main_pi_tf / self.max_u))

        else:  # If not training with demonstrations
            pi_loss_tf = -tf.reduce_mean(self.main_q_pi_tf)
            pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main_pi_tf / self.max_u))

        self.pi_loss_tf = pi_loss_tf

        # Gradients
        # gradients of Q
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars("main/Q"))
        assert len(self._vars("main/Q")) == len(Q_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars("main/Q"))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars("main/Q"))
        # gradients of pi
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars("main/pi"))
        assert len(self._vars("main/pi")) == len(pi_grads_tf)
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars("main/pi"))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars("main/pi"))

        # Optimizers
        self.Q_adam = MpiAdam(self._vars("main/Q"), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars("main/pi"), scale_grad_by_procs=False)

        # Polyak averaging
        self.main_vars = self._vars("main/Q") + self._vars("main/pi")
        self.target_vars = self._vars("target/Q") + self._vars("target/pi")
        self.init_target_net_op = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(
                lambda v: v[0].assign(self.polyak * v[0] + (1.0 - self.polyak) * v[1]),
                zip(self.target_vars, self.main_vars),
            )
        )

        # Initialize all variables
        # tf.variables_initializer(self._global_vars("")).run()
        self.sess.run(tf.global_variables_initializer())
        self.Q_adam.sync()
        self.pi_adam.sync()
        self.init_target_net()
        self.training_step = 0  # initialize number of training step

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_state(self, state):
        state = np.clip(state, -self.clip_obs, self.clip_obs)
        return state

    def _global_vars(self, scope=""):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/" + scope)
        return res

    def _vars(self, scope=""):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + "/" + scope)
        assert len(res) > 0
        return res

    def _update_stats(self, episode_batch):
        # add transitions to normalizer
        episode_batch["o_2"] = episode_batch["o"][:, 1:, :]
        if self.dimg != 0:
            episode_batch["ag_2"] = episode_batch["ag"][:, :, :]
            episode_batch["g_2"] = episode_batch["g"][:, :, :]
        num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
        transitions = self.replay_buffer.sample_transitions(episode_batch, num_normalizing_transitions)

        self.o_stats.update(self._preprocess_state(transitions["o"]))
        self.o_stats.recompute_stats()
        if self.dimg != 0:
            self.g_stats.update(self._preprocess_state(transitions["g"]))
            self.g_stats.recompute_stats()

    def __getstate__(self):

        # """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        # """
        excluded_names = ["self", "replay_strategy"]

        state = {k: v for k, v in self.init_args.items() if not k in excluded_names}
        state["tf"] = self.sess.run([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        return state

    def __setstate__(self, state):

        # We don't need the replay strategy for playing the policy.
        assert "replay_strategy" not in state
        state["replay_strategy"] = None

        kwargs = state["kwargs"]
        del state["kwargs"]

        self.__init__(**state, **kwargs)
        vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        assert len(vars) == len(state["tf"])
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    ###########################################
    # Queries
    ###########################################

    # Store the result you want to query in a npz file.

    def query_potential(self, dim1=0, dim2=1, filename=None):
        """Check the output from demo shaping potential function
        """
        if not "Reach" in self.info["env_name"]:
            return

        logger.info("Query: potential -> Plot the potential over (s, a) space of demo shaping.")

        def visualize_training_data(gs, row, dim1, dim2):
            # sample all training data
            demo_data = self.demo_buffer.sample_all()
            # select some dimension of the concatenated data (concatenate the data in the same order as in demo_shaping)
            concat_demo_data = demo_data["o"]
            if self.dimg != 0:
                # for multigoal environments, we have goal as another states
                concat_demo_data = np.concatenate([concat_demo_data, demo_data["g"]], axis=1)
            concat_demo_data = np.concatenate([concat_demo_data, demo_data["u"]], axis=1)

            x = concat_demo_data[:, dim1 : dim1 + 1]
            y = concat_demo_data[:, dim2 : dim2 + 1]
            np_samples = np.concatenate((x, y), axis=1)

            # Plot Training dataset
            ax = pl.subplot(gs[row, 0])
            ax.clear()
            ax.scatter(np_samples[:, 0], np_samples[:, 1], s=10, color="red")
            # ax.set_xlim([-1, 1])
            # ax.set_ylim([-1, 1])
            ax.set_xlabel("state")
            ax.set_ylabel("action")
            ax.set_title("Training samples")

        def sample_flow(base_dist, transformed_dist):
            x = base_dist.sample(512)
            samples = [x]
            names = [base_dist.name]
            for bijector in reversed(transformed_dist.bijector.bijectors):
                x = bijector.forward(x)
                samples.append(x)
                names.append(bijector.name)

            return x, samples, names

        def visualize_flow(gs, row, samples, titles, dim1, dim2):
            X0 = samples[0]

            for i, j in zip([0, len(samples) - 1], [0, 1]):  # range(len(samples)):
                X1 = samples[i]

                ax = pl.subplot(gs[row, j])
                ax.clear()

                idx = np.logical_and(X0[:, dim1] < 0, X0[:, dim2] < 0)
                ax.scatter(X1[idx, dim1], X1[idx, dim2], s=10, color="red")

                idx = np.logical_and(X0[:, dim1] > 0, X0[:, dim2] < 0)
                ax.scatter(X1[idx, dim1], X1[idx, dim2], s=10, color="green")

                idx = np.logical_and(X0[:, dim1] < 0, X0[:, dim2] > 0)
                ax.scatter(X1[idx, dim1], X1[idx, dim2], s=10, color="blue")

                idx = np.logical_and(X0[:, dim1] > 0, X0[:, dim2] > 0)
                ax.scatter(X1[idx, dim1], X1[idx, dim2], s=10, color="black")

                # ax.set_xlim([-1, 1])
                # ax.set_ylim([-1, 1])
                ax.set_xlabel("state")
                ax.set_ylabel("action")
                ax.set_title(titles[j])

        # Turn on interactive mode to see immediate change
        pl.ion()
        # pl.figure() # open a figure and switch to the figure
        gs = gridspec.GridSpec(2, 2)

        # Plot training data distribution
        visualize_training_data(gs, 0, dim1, dim2)

        # Flow after training
        _, samples_with_training, _ = sample_flow(self.demo_shaping.base_dist, self.demo_shaping.nn.dist)
        samples_with_training = self.sess.run(samples_with_training)
        visualize_flow(gs, 1, samples_with_training, ["Base dist", "Samples w/ training"], dim1, dim2)

        if filename != None:
            logger.info("Query: potential -> storing query results to {}".format(filename))
            pl.savefig(filename)
        # pl.show()
        # pl.pause(0.001)

    def query_policy(self, filename=None, fid=0):

        """Only use this function for 1d first order reacher problem
        """

        num_point = 24
        ls = np.linspace(-1.0, 1.0, num_point)
        o_1, o_2 = np.meshgrid(ls, ls)
        o_r = np.concatenate((o_1.reshape(-1, 1), o_2.reshape(-1, 1)), axis=1)
        g_r = 0.0 * np.ones((num_point ** 2, 2))

        ret = self.get_actions(o=o_r, g=g_r)

        res = {"o": o_r, "u": ret}

        if filename:
            logger.info("Query: action over state space -> storing query results to {}".format(filename))
            np.savez_compressed(filename, **res)
        else:
            # plot the result on the fly
            pl.figure(fid)  # create a new figure
            gs = gridspec.GridSpec(1, 1)
            ax = pl.subplot(gs[0, 0])
            ax.clear()
            visualize_query.visualize_action(ax, res)
            pl.show()
            pl.pause(0.001)
