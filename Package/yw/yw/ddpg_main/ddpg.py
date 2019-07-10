import numpy as np
import tensorflow as tf

from yw.tool import logger

from yw.ddpg_main.mpi_adam import MpiAdam
from yw.ddpg_main.normalizer import Normalizer
from yw.ddpg_main.actor_critic import ActorCritic
from yw.ddpg_main.replay_buffer import *
from yw.ddpg_main.demo_shaping import (
    ManualDemoShaping,
    GaussianDemoShaping,
    NormalizingFlowDemoShaping,
    MAFDemoShaping,
)
from yw.util.util import store_args, import_function
from yw.util.tf_util import flatten_grads

# for query
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from yw.flow import visualize_query


class DDPG(object):
    def __init__(
        self,
        input_dims,
        num_sample,
        use_td3,
        hidden,
        layers,
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
        rollout_batch_size,
        clip_pos_returns,
        clip_return,
        demo_critic,
        demo_actor,
        num_demo,
        q_filter,
        batch_size_demo,
        prm_loss_weight,
        aux_loss_weight,
        replay_strategy,
        demo_replay_strategy,
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
            # Rollout Worker
            rollout_batch_size (int)          - number of parallel rollouts per DDPG agent
            # NN Configuration
            scope              (str)          - the scope used for the TensorFlow graph
            input_dims         (dict of ints) - dimensions for the observation (o), the goal (g), and the actions (u)
            hidden             (int)          - number of units in the hidden layers
            layers             (int)          - number of hidden layers
            reuse              (boolean)      - whether or not the networks should be reused
            # Replay Buffer
            buffer_size        (int)          - number of transitions that are stored in the replay buffer
            # HER
            replay_strategy (function)     - function that samples from the replay buffer
            demo_replay_strategy (function)
            # Dual Network Set
            polyak             (float)        - coefficient for Polyak-averaging of the target network
            # Training
            batch_size         (int)          - batch size for training
            Q_lr               (float)        - learning rate for the Q (critic) network
            pi_lr              (float)        - learning rate for the pi (actor) network
            action_l2          (float)        - coefficient for L2 penalty on the actions
            gamma              (float)        - gamma used for Q learning updates
            # Use demonstration to shape critic or actor
            demo_actor         (str)          - whether or not to use demonstration network to shape the critic or actor
            demo_critic        (str)          - whether or not to use demonstration network to shape the critic or actor
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
        self.num_sample = num_sample
        self.use_td3 = use_td3
        self.hidden = hidden
        self.layers = layers
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
        self.rollout_batch_size = rollout_batch_size
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.demo_critic = demo_critic
        self.demo_actor = demo_actor
        self.num_demo = num_demo
        self.q_filter = q_filter
        self.batch_size_demo = batch_size_demo
        self.prm_loss_weight = prm_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.replay_strategy = replay_strategy
        self.demo_replay_strategy = demo_replay_strategy
        self.gamma = gamma
        self.info = info

        # Prepare parameters
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        logger.info("Configuring the replay buffer.")
        buffer_shapes = {}
        buffer_shapes["o"] = (self.T + 1, self.dimo)
        buffer_shapes["u"] = (self.T, self.dimu)
        buffer_shapes["r"] = (self.T, 1)
        if self.dimg != 0:  # for multigoal environment - or states that do not change over episodes.
            buffer_shapes["ag"] = (self.T + 1, self.dimg)
            buffer_shapes["g"] = (self.T, self.dimg)
        # for bootstrapped ensemble of actor critics
        buffer_shapes["mask"] = (self.T, self.num_sample)  # mask for training each head with different dataset
        # add extra information
        for key, val in input_dims.items():
            if key.startswith("info"):
                buffer_shapes[key] = (self.T, *(tuple([val]) if val > 0 else tuple()))
        logger.debug("DDPG.__init__ -> The buffer shapes are: {}".format(buffer_shapes))

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        if not self.replay_strategy:
            pass
        elif self.replay_strategy["strategy"] == "her":
            self.replay_buffer = HERReplayBuffer(buffer_shapes, buffer_size, self.T, **self.replay_strategy["args"])
        else:
            self.replay_buffer = UniformReplayBuffer(
                buffer_shapes, buffer_size, self.T, **self.replay_strategy["args"]
            )

        # Initialize the demo buffer; in the same way as the primary data buffer
        if not self.demo_replay_strategy:
            pass
        elif self.demo_actor != "none" or self.demo_critic != "none":
            self.demo_buffer = UniformReplayBuffer(
                buffer_shapes, buffer_size, self.T, **self.demo_replay_strategy["args"]
            )
            # This does not matter is using demo_critic, since we call sample all
            # self.demo_buffer = HERReplayBuffer(buffer_shapes, buffer_size, self.T, **self.replay_strategy["args"])

        # Build computation core.
        with tf.variable_scope(self.scope):
            logger.info("Creating a DDPG agent with action space %d x %s." % (self.dimu, self.max_u))
            self._create_network()

    def get_actions(self, o, ag, g, noise_eps=0.0, random_eps=0.0, use_target_net=False, compute_Q=False):
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf, policy.Q_pi_mean_tf]
        # feed
        o = self._preprocess_state(o)
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
        }
        if self.dimg != 0:
            g = self._preprocess_state(g)
            feed[policy.g_tf] = g.reshape(-1, self.dimg)
        ret = self.sess.run(vals, feed_dict=feed)

        # action postprocessing
        if use_target_net:  # base on the assumption that target net is only used when testing
            # randomly select the output action of one policy.
            u = np.mean(ret[0], axis=0)
        else:
            u = ret[0][np.random.randint(self.num_sample)]

        # debug only
        logger.debug("DDPG.get_actions -> Dumping out action input and output for debugging.")
        logger.debug("DDPG.get_actions -> The observation shape is: {}".format(o.shape))
        logger.debug("DDPG.get_actions -> The goal shape is: {}".format(g.shape))
        logger.debug("DDPG.get_actions -> The estimated q value shape is: {}".format(ret[1].shape))
        logger.debug("DDPG.get_actions -> The action array shape is: {} x {}".format(len(ret[0]), ret[0][0].shape))
        logger.debug("DDPG.get_actions -> The selected action shape is: {}".format(u.shape))
        # value debug: check the q from rl.
        # logger.info("DDPG.get_actions -> The estimated q value is: {}".format(ret[1]))
        # logger.info("DDPG.get_actions -> The selected action value is: {}".format(u))

        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (
            self._random_action(u.shape[0]) - u
        )  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if compute_Q:
            return ret
        else:
            return ret[0]

    def init_demo_buffer(self, demo_file, update_stats=True):
        """ Initialize the demonstration buffer. Used for getting demonstration from our own environments only.
        """
        logger.info("Initialized demonstration buffer with {} episodes.".format(self.num_demo))

        demo_data = np.load(demo_file)  # load the demonstration data from data file

        assert self.num_demo <= demo_data["u"].shape[0], "No enough demonstration data!"
        episode_batch = {**demo_data}
        for key in episode_batch.keys():
            episode_batch[key] = episode_batch[key][: self.num_demo]
        # the demo buffer should already have: o, u, r, ag, g and necessary infos.
        # for boot strapped ensemble of actor critics
        episode_batch["mask"] = np.random.binomial(1, 1, (self.num_demo, self.T, self.num_sample))

        self.demo_buffer.store_episode(episode_batch)

        # feed demonstration data for norm shaping
        if self.demo_critic == "norm":
            logger.info("DDPG:init_demo_buffer -> Assign demo data to variables.")
            demo_data = self.demo_buffer.sample_all()
            demo_data["o"] = self._preprocess_state(demo_data["o"])
            if self.dimg != 0:
                demo_data["g"] = self._preprocess_state(demo_data["g"])
            for k in self.demo_inputs_tf.keys():  # o g u
                self.sess.run(tf.assign(self.demo_inputs_tf[k], demo_data[k]))

        # currently we do not update status if using demo_actor==none
        if update_stats:
            logger.info("DDPG:init_demo_buffer -> Updating stats.")
            self._update_stats(episode_batch)

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        for key in episode_batch.keys():
            assert episode_batch[key].shape[0] == self.rollout_batch_size
        # mask transitions for each bootstrapped head
        # select a distribution!
        episode_batch["mask"] = np.float32(
            np.random.binomial(1, 1, (self.rollout_batch_size, self.T, self.num_sample))
        )

        self.replay_buffer.store_episode(episode_batch)

        if update_stats:
            self._update_stats(episode_batch)

    def sample_batch(self):
        # use demonstration buffer to sample as well if demo flag is set TRUE
        if self.demo_actor != "none":
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
        if self.demo_critic == "maf":
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
        logger.debug("DDPG.train -> critic_loss:{}, actor_loss:{}".format(critic_loss, actor_loss))
        self.current_batch = batch

        return critic_loss, actor_loss

    def check_train(self):
        """ For debugging only
        """
        pass
        # feed = {self.inputs_tf[k]: self.current_batch[k] for k in self.inputs_tf.keys()}
        # # feed demonstration data
        # if self.demo_critic == "norm":
        #     demo_data = self.demo_buffer.sample_all()
        #     demo_data["o"] = self._preprocess_state(demo_data["o"])
        #     if self.dimg != 0:
        #         demo_data["g"] = self._preprocess_state(demo_data["g"])
        #     for k in self.demo_inputs_tf.keys():
        #         feed[self.demo_inputs_tf[k]] = demo_data[k]
        # potential, reward = self.sess.run([self.demo_critic_shaping_ls[0].potential, self.demo_critic_shaping_ls[0].reward], feed_dict=feed)
        # print(np.stack((potential, reward), axis=1))

    def init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        logger.debug("DDPG.update_target_net -> updating target net.")
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
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        self.inputs_tf = {}
        self.inputs_tf["o"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.inputs_tf["o_2"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.inputs_tf["u"] = tf.placeholder(tf.float32, shape=(None, self.dimu))
        self.inputs_tf["r"] = tf.placeholder(tf.float32, shape=(None, 1))
        if self.dimg != 0:
            self.inputs_tf["g"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
            self.inputs_tf["g_2"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
        # boot strapped ensemble of actor critics
        self.inputs_tf["mask"] = tf.placeholder(tf.int32, shape=(None, self.num_sample))

        self.target_inputs_tf = self.inputs_tf.copy()
        # The input to the target network has to be the resultant observation and goal!
        self.target_inputs_tf["o"] = self.inputs_tf["o_2"]
        if self.dimg != 0:
            self.target_inputs_tf["g"] = self.inputs_tf["g_2"]

        # Creating a normalizer for goal and observation.
        with tf.variable_scope("o_stats") as vs:
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope("g_stats") as vs:
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # Networks
        with tf.variable_scope("main", reuse=tf.AUTO_REUSE) as vs:
            self.main = ActorCritic(
                inputs_tf=self.inputs_tf,
                dimo=self.dimo,
                dimg=self.dimg,
                dimu=self.dimu,
                max_u=self.max_u,
                o_stats=self.o_stats,
                g_stats=self.g_stats,
                num_sample=self.num_sample,
                use_td3=self.use_td3,
                hidden=self.hidden,
                layers=self.layers,
                add_pi_noise=False,
            )
            self.main_shaping = ActorCritic(
                inputs_tf=self.target_inputs_tf,
                dimo=self.dimo,
                dimg=self.dimg,
                dimu=self.dimu,
                max_u=self.max_u,
                o_stats=self.o_stats,
                g_stats=self.g_stats,
                num_sample=self.num_sample,
                use_td3=self.use_td3,
                hidden=self.hidden,
                layers=self.layers,
                add_pi_noise=False,
            )
        with tf.variable_scope("target") as vs:
            self.target = ActorCritic(
                inputs_tf=self.target_inputs_tf,
                dimo=self.dimo,
                dimg=self.dimg,
                dimu=self.dimu,
                max_u=self.max_u,
                o_stats=self.o_stats,
                g_stats=self.g_stats,
                num_sample=self.num_sample,
                use_td3=self.use_td3,
                hidden=self.hidden,
                layers=self.layers,
                add_pi_noise=True,
            )
        assert len(self._vars("main")) == len(self._vars("target"))

        # Add shaping reward
        if self.demo_critic != "none":
            self.demo_critic_shaping_ls = []
            self.demo_actor_shaping_ls = []
            # for debugging
            self.demo_shaping_check_ls = []

            with tf.variable_scope("shaping") as vs:
                if self.demo_critic == "manual":
                    self.demo_shaping = ManualDemoShaping(gamma=self.gamma)
                elif self.demo_critic == "norm":
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
                elif self.demo_critic == "maf":
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

                for i in range(self.num_sample):
                    self.demo_critic_shaping_ls.append(
                        tf.cast(
                            self.demo_shaping.reward(
                                o=self.inputs_tf["o"],
                                g=self.inputs_tf["g"] if self.dimg != 0 else None,
                                u=self.inputs_tf["u"],
                                o_2=self.inputs_tf["o_2"],
                                g_2=self.inputs_tf["g_2"] if self.dimg != 0 else None,
                                u_2=self.main_shaping.pi_tf[i],
                            ),
                            tf.float32,
                        )
                    )
                    self.demo_shaping_check_ls.append(
                        tf.cast(
                            self.demo_shaping.potential(
                                o=self.inputs_tf["o"],
                                g=self.inputs_tf["g"] if self.dimg != 0 else None,
                                u=self.inputs_tf["u"],
                            ),
                            tf.float32,
                        )
                    )
                    self.demo_actor_shaping_ls.append(
                        tf.cast(
                            self.demo_shaping.potential(
                                o=self.inputs_tf["o"],
                                g=self.inputs_tf["g"] if self.dimg != 0 else None,
                                u=self.main.pi_tf[i],
                            ),
                            tf.float32,
                        )
                    )

                assert all([ele.shape[1] == 1 for ele in self.demo_critic_shaping_ls])
                assert all([ele.shape[1] == 1 for ele in self.demo_actor_shaping_ls])

        # Critic loss
        # clip bellman target return
        clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else self.clip_return)
        self.Q_loss_tf = []
        if self.demo_critic != "none":
            for i in range(self.num_sample):
                if self.use_td3:
                    # calculate bellman target (with shaping reward added)
                    target_tf = tf.clip_by_value(
                        self.inputs_tf["r"]
                        + tf.stop_gradient(self.demo_critic_shaping_ls[i])
                        + self.gamma * tf.minimum(self.target.Q_pi_tf[i], self.target.Q2_pi_tf[i]),
                        *clip_range
                    )
                    assert target_tf.shape[1] == 1
                    rl_bellman_1_tf = tf.boolean_mask(
                        tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf[i]), self.inputs_tf["mask"][:, i]
                    )
                    rl_bellman_2_tf = tf.boolean_mask(
                        tf.square(tf.stop_gradient(target_tf) - self.main.Q2_tf[i]), self.inputs_tf["mask"][:, i]
                    )
                    rl_loss_tf = (tf.reduce_mean(rl_bellman_1_tf) + tf.reduce_mean(rl_bellman_2_tf)) / 2.0

                else:
                    # calculate bellman target (with shaping reward added)
                    target_tf = tf.clip_by_value(
                        self.inputs_tf["r"]
                        + tf.stop_gradient(self.demo_critic_shaping_ls[i])
                        + self.gamma * self.target.Q_pi_tf[i],
                        *clip_range
                    )
                    assert target_tf.shape[1] == 1
                    rl_bellman_tf = tf.boolean_mask(
                        tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf[i]), self.inputs_tf["mask"][:, i]
                    )
                    rl_loss_tf = tf.reduce_mean(rl_bellman_tf)
                self.Q_loss_tf.append(rl_loss_tf)
        else:
            for i in range(self.num_sample):
                if self.use_td3:
                    # calculate bellman target (with shaping reward added)
                    target_tf = tf.clip_by_value(
                        self.inputs_tf["r"] + self.gamma * tf.minimum(self.target.Q_pi_tf[i], self.target.Q2_pi_tf[i]),
                        *clip_range
                    )
                    assert target_tf.shape[1] == 1
                    rl_bellman_1_tf = tf.boolean_mask(
                        tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf[i]), self.inputs_tf["mask"][:, i]
                    )
                    rl_bellman_2_tf = tf.boolean_mask(
                        tf.square(tf.stop_gradient(target_tf) - self.main.Q2_tf[i]), self.inputs_tf["mask"][:, i]
                    )
                    rl_loss_tf = (tf.reduce_mean(rl_bellman_1_tf) + tf.reduce_mean(rl_bellman_2_tf)) / 2.0
                else:
                    # calculate bellman target (with shaping reward added)
                    target_tf = tf.clip_by_value(
                        self.inputs_tf["r"] + self.gamma * self.target.Q_pi_tf[i], *clip_range
                    )
                    assert target_tf.shape[1] == 1
                    rl_bellman_tf = tf.boolean_mask(
                        tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf[i]), self.inputs_tf["mask"][:, i]
                    )
                    rl_loss_tf = tf.reduce_mean(rl_bellman_tf)
                self.Q_loss_tf.append(rl_loss_tf)

        # Actor Loss
        if self.demo_actor != "none" and self.q_filter == 1:
            # train with demonstrations and use demo and q_filter both
            # where is the demonstrator action better than actor action according to the critic? choose those sample only
            mask = np.concatenate(
                (np.zeros(self.batch_size - self.batch_size_demo), np.ones(self.batch_size_demo)), axis=0
            )
            self.pi_loss_tf = []
            for i in range(self.num_sample):
                # primary loss scaled by it's respective weight prm_loss_weight
                pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf[i])
                # L2 loss on action values scaled by the same weight prm_loss_weight
                pi_loss_tf += (
                    self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf[i] / self.max_u))
                )
                # define the cloning loss on the actor's actions only on the samples which adhere to the above masks
                maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf[i] > self.main.Q_pi_tf[i], mask), [-1])
                cloning_loss_tf = tf.reduce_sum(
                    tf.square(
                        tf.boolean_mask(tf.boolean_mask((self.main.pi_tf[i]), mask), maskMain, axis=0)
                        - tf.boolean_mask(tf.boolean_mask((self.inputs_tf["u"]), mask), maskMain, axis=0)
                    )
                )
                # adding the cloning loss to the actor loss as an auxilliary loss scaled by its weight aux_loss_weight
                pi_loss_tf += self.aux_loss_weight * cloning_loss_tf
                self.pi_loss_tf.append(pi_loss_tf)

        elif self.demo_actor != "none" and not self.q_filter:  # train with demonstrations without q_filter
            # choose only the demo buffer samples
            mask = np.concatenate(
                (np.zeros(self.batch_size - self.batch_size_demo), np.ones(self.batch_size_demo)), axis=0
            )
            self.pi_loss_tf = []
            for i in range(self.num_sample):
                pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf[i])
                pi_loss_tf += (
                    self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf[i] / self.max_u))
                )
                cloning_loss_tf = tf.reduce_sum(
                    tf.square(
                        tf.boolean_mask((self.main.pi_tf[i]), mask) - tf.boolean_mask((self.inputs_tf["u"]), mask)
                    )
                )
                pi_loss_tf += self.aux_loss_weight * cloning_loss_tf
                self.pi_loss_tf.append(pi_loss_tf)

        elif self.demo_critic != "none":
            self.pi_loss_tf = [
                -tf.reduce_mean(self.main.Q_pi_tf[i])
                - tf.reduce_mean(self.demo_actor_shaping_ls[i])
                + self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf[i] / self.max_u))
                for i in range(self.num_sample)
            ]

        else:  # If not training with demonstrations
            self.pi_loss_tf = [
                -tf.reduce_mean(self.main.Q_pi_tf[i])
                + self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf[i] / self.max_u))
                for i in range(self.num_sample)
            ]

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

        # Debug info
        logger.debug("Demo.__init__ -> Global variables are: {}".format(self._global_vars()))
        logger.debug("Demo.__init__ -> Trainable variables are: {}".format(self._vars()))

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
        excluded_names = ["self", "demo_replay_strategy", "replay_strategy"]

        state = {k: v for k, v in self.init_args.items() if not k in excluded_names}
        state["tf"] = self.sess.run([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        return state

    def __setstate__(self, state):

        # We don't need these for playing the policy.
        assert "replay_strategy" not in state
        assert "demo_replay_strategy" not in state
        state["replay_strategy"] = None
        state["demo_replay_strategy"] = None

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

    # Currently you can query the value of:
    #     staging_tf          -> staged values
    #     buffer_ph_tf        -> placeholders
    #     demo_q_mean_tf      ->
    #     demo_q_var_tf       ->
    #     demo_q_sample_tf    ->
    #     max_q_tf            ->
    #     Q_loss_tf           -> a list of losses for each critic sample
    #     pi_loss_tf          -> a list of losses for each pi sample
    #     policy.pi_tf        -> list of output from actor
    #     policy.Q_tf         ->
    #     policy.Q_pi_tf      ->
    #     policy.Q_pi_mean_tf ->
    #     policy.Q_pi_var_tf  ->
    #     policy._input_Q     ->

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

    def query_potential_surface(self, filename=None, fid=0):
        """Check the output from demo shaping potential function
        """
        if not "Reach" in self.info["env_name"] or self.demo_critic == "none":
            return

        potential = self.demo_shaping_check_ls[0]

        num_point = 24
        ls = np.linspace(-1.0, 1.0, num_point)
        ls2 = ls * 2
        o_1, o_2 = np.meshgrid(ls, ls)
        u_1, u_2 = np.meshgrid(ls2, ls2)
        o_r = np.concatenate((o_1.reshape(-1, 1), o_2.reshape(-1, 1)), axis=1)
        u_r = np.concatenate((u_1.reshape(-1, 1), u_2.reshape(-1, 1)), axis=1)
        g_r = np.zeros((num_point ** 2, 2))

        feed = {self.inputs_tf["o"]: o_r, self.inputs_tf["u"]: u_r, self.inputs_tf["g"]: g_r}
        ret = self.sess.run(potential, feed_dict=feed)
        ret = ret.reshape((num_point, num_point))

        res = {"o": (o_1, o_2), "potential": ret}

        if filename:
            logger.info("Query: action over state space -> storing query results to {}".format(filename))
            np.savez_compressed(filename, **res)
        else:
            # plot the result on the fly
            pl.figure(fid)
            gs = gridspec.GridSpec(1, 1)
            ax = pl.subplot(gs[0, 0], projection="3d")
            ax.clear()
            visualize_query.visualize_potential_surface(ax, res)
            pl.show()
            pl.pause(0.001)

    def query_potential_based_policy(self, filename=None, fid=0):

        """Create a policy that only optimize over the potential function
        """

        from yw.util.tf_util import nn

        # input data
        num_point = 24
        ls = np.linspace(-1.0, 1.0, num_point)
        o_1, o_2 = np.meshgrid(ls, ls)
        o_r = np.concatenate((o_1.reshape(-1, 1), o_2.reshape(-1, 1)), axis=1)
        g_r = 0.0 * np.ones((num_point ** 2, 2))

        # inputs
        inputs_tf = {}
        inputs_tf["o"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        inputs_tf["g"] = tf.placeholder(tf.float32, shape=(None, self.dimg))

        # add neural network and call neural network
        with tf.variable_scope("potential_based_policy"):
            state_tf = tf.concat((inputs_tf["o"], inputs_tf["g"]), axis=1)
            pi_tf = self.max_u * tf.tanh(nn(state_tf, [self.hidden] * self.layers + [self.dimu]))

            # add loss function and trainer
            potential = tf.cast(self.demo_shaping.potential(o=inputs_tf["o"], g=inputs_tf["g"], u=pi_tf), tf.float32)
            loss_tf = -tf.reduce_mean(potential)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(
                loss_tf, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="potential_based_policy")
            )
            init = tf.initializers.variables(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="potential_based_policy")
            )

        self.sess.run(init)
        num_epochs = 100
        for i in range(num_epochs):
            loss, _ = self.sess.run([loss_tf, train_op], feed_dict={inputs_tf["o"]: o_r, inputs_tf["g"]: g_r})
            if i % (num_epochs / 10) == (num_epochs / 10 - 1):
                logger.info("Query: policy on potential only -> epoch: {} loss: {}".format(i, loss))

        ret = self.sess.run(pi_tf, feed_dict={inputs_tf["o"]: o_r, inputs_tf["g"]: g_r})
        res = {"o": o_r, "u": ret}

        if filename:
            logger.info("Query: policy on potential only -> storing query results to {}".format(filename))
            np.savez_compressed(filename, **res)
        else:
            # plot the result on the fly
            pl.figure(fid)  # create a new figure
            gs = gridspec.GridSpec(1, 1)
            ax = pl.subplot(gs[0, 0])
            ax.clear()
            visualize_query.visualize_action(ax, res)
            pl.show()
            pl.pause(10)
    
        exit() # this function adds extra nodes to the graph

    def query_action(self, filename=None, fid=0):

        """Only use this function for 1d first order reacher problem
        """

        num_point = 24
        ls = np.linspace(-1.0, 1.0, num_point)
        o_1, o_2 = np.meshgrid(ls, ls)
        o_r = np.concatenate((o_1.reshape(-1, 1), o_2.reshape(-1, 1)), axis=1)
        g_r = 0.0 * np.ones((num_point ** 2, 2))

        ret = self.get_actions(o=o_r, ag=None, g=g_r)

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

    def query_uncertainty(self, filename=None):
        """Check the output from demonstration NN when the state is fixed and the action forms a 2d space.

        This check can only be used for the Reach2DFirstOrder environment.
        """
        if not "Reach" in self.info["env_name"]:
            return

        logger.info("Query: uncertainty -> Plot the uncertainty over (s, a) space of critic.")

        num_point = 24
        ls = np.linspace(-1.0, 1.0, num_point)
        ls2 = ls * 2
        o, u = np.meshgrid(ls, ls2)
        g = 0.0 * np.ones((pow(num_point, 2), 1))
        o_r = o.reshape((-1, 1))
        u_r = u.reshape((-1, 1))
        g_r = g.reshape((-1, 1))

        policy = self.main
        name = ["o", "u", "g", "q_var", "q_mean"]
        feed = {policy.o_tf: o_r, policy.g_tf: g_r, policy.u_tf: u_r}
        # feed demonstration data
        if self.demo_critic != "none":
            demo_data = self.demo_buffer.sample_all()
            for k in self.demo_inputs_tf.keys():
                feed[self.demo_inputs_tf[k]] = demo_data[k]
        queries = [policy.Q_var_tf, policy.Q_mean_tf]
        res = self.sess.run(queries, feed_dict=feed)

        if self.demo_critic != "none":
            temp = self.sess.run(self.demo_shaping_check_ls[0], feed_dict=feed)
            temp = temp.reshape(-1)
            res[1] = temp

        o_r = o.reshape((num_point, num_point))
        u_r = u.reshape((num_point, num_point))
        g_r = g.reshape((num_point, num_point))
        for i in range(len(res)):
            res[i] = res[i].reshape((num_point, num_point))

        ret = [o_r, u_r, g_r] + res

        if filename:
            logger.info("Query: uncertainty -> storing query results to {}".format(filename))
            np.savez_compressed(filename, **dict(zip(name, ret)))
        else:
            logger.info("Query: uncertainty -> ", ret)
