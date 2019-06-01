import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from collections import OrderedDict

from yw.tool import logger

from yw.ddpg_main.mpi_adam import MpiAdam
from yw.ddpg_main.normalizer import Normalizer
from yw.ddpg_main.actor_critic import ActorCritic
from yw.ddpg_main.replay_buffer import ReplayBuffer
from yw.util.util import store_args, import_function
from yw.util.tf_util import flatten_grads


class DDPG(object):
    @store_args
    def __init__(
        self,
        input_dims,
        num_sample,
        ca_ratio,
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
        sample_rl_transitions,
        sample_demo_transitions,
        gamma,
        reuse=False,
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
            sample_rl_transitions (function)     - function that samples from the replay buffer
            sample_demo_transitions (function)
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
            demo_policy        (cls Demo)     - trained demonstration nn
            num_demo           (int)          - Number of episodes in to be used in the demonstration buffer
            batch_size_demo    (int)          - number of samples to be used from the demonstrations buffer, per mpi thread
            q_filter           (boolean)      - whether or not a filter on the q value update should be used when training with demonstartions
            prm_loss_weight    (float)        - Weight corresponding to the primary loss
            aux_loss_weight    (float)        - Weight corresponding to the auxilliary loss also called the cloning loss

        """
        # Prepare parameters
        self.dimo = self.input_dims["o"]
        self.dimg = self.input_dims["g"]
        self.dimu = self.input_dims["u"]

        logger.info("Configuring the replay buffer.")
        buffer_shapes = {}
        buffer_shapes["o"] = (self.T + 1, self.dimo)
        buffer_shapes["u"] = (self.T, self.dimu)
        buffer_shapes["r"] = (self.T, 1)
        # for multigoal environment - or states that do not change over episodes.
        if self.dimg != 0:
            buffer_shapes["ag"] = (self.T + 1, self.dimg)
            buffer_shapes["g"] = (self.T, self.dimg)
        # for bootstrapped ensemble of actor critics
        buffer_shapes["mask"] = (self.T, self.num_sample)  # mask for training each head with different dataset
        # for demonstrations
        buffer_shapes["n"] = (self.T, 1)  # n step return
        buffer_shapes["q"] = (self.T, 1)  # expected q value
        # add extra information
        for key, val in input_dims.items():
            if key.startswith("info"):
                buffer_shapes["n"] = (self.T, *(tuple([val]) if val > 0 else tuple()))
        logger.debug("DDPG.__init__ -> The buffer shapes are: {}".format(buffer_shapes))

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        self.replay_buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_rl_transitions)
        if self.demo_actor != "none" or self.demo_critic == "rb":
            # initialize the demo buffer; in the same way as the primary data buffer
            self.demo_buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_demo_transitions)

        # Build computation core.
        with tf.variable_scope(self.scope):
            logger.info("Preparing staging area for feeding data to the model.")
            stage_shapes = OrderedDict()
            stage_shapes["o"] = (None, self.dimo)
            stage_shapes["o_2"] = (None, self.dimo)
            stage_shapes["u"] = (None, self.dimu)
            stage_shapes["r"] = (None, 1)
            # for multigoal environments
            if self.dimg != 0:
                stage_shapes["g"] = (None, self.dimg)
                stage_shapes["g_2"] = (None, self.dimg)
            # for bootstrapped DQN, to add mask
            stage_shapes["mask"] = (None, self.num_sample)
            # for demonstrations
            stage_shapes["q"] = (None, 1)
            stage_shapes["n"] = (None, 1)
            self.stage_shapes = stage_shapes  # feeding data into model
            logger.info("DDPG.__init__ -> The staging shapes are: {}".format(self.stage_shapes))

            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()], shapes=list(self.stage_shapes.values())
            )
            self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            logger.info("Creating a DDPG agent with action space %d x %s." % (self.dimu, self.max_u))
            self._create_network(reuse=reuse)

    def get_q_value(self, batch):
        policy = self.target
        # values to compute
        vals = [policy.Q_sample_tf, policy.Q_mean_tf, policy.Q_var_tf]
        # feed
        feed = {
            policy.o_tf: batch["o"].reshape(-1, self.dimo),
            policy.u_tf: batch["u"].reshape(-1, self.dimu),
        }
        if self.dimg != 0:
            feed[policy.g_tf] = batch["g"].reshape(-1, self.dimg)

        return self.sess.run(vals, feed_dict=feed)

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
        logger.debug("DDPG.get_actions -> The achieved goal shape is: {}".format(ag.shape))
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
        episode_batch = OrderedDict(**demo_data)
        # mask transitions for each bootstrapped head
        # select a distribution!
        for key in episode_batch.keys():
            episode_batch[key] = episode_batch[key][: self.num_demo]

        episode_batch["mask"] = np.random.binomial(1, 1, (self.num_demo, self.T, self.num_sample))
        episode_batch["n"] = np.ones((self.num_demo, self.T, 1), dtype=np.float32)
        if self.demo_critic != "rb":
            # fill in the minimal value of q for rollout data.
            episode_batch["q"] = -100 * np.ones((self.num_demo, self.T, 1), dtype=np.float32)
        self.demo_buffer.store_episode(episode_batch)

        if update_stats:
            self._update_stats(episode_batch)

    # def init_demo_buffer_2(self, demo_data_file, update_stats=True):
    #     """ Another function for initializing the demonstration buffer. Used for getting demonstration from OpenAI
    #         environments only.
    #     """
    #     demo_data = np.load(demo_data_file)  # load the demonstration data from data file
    #     info_keys = [key.replace("info_", "") for key in self.input_dims.keys() if key.startswith("info_")]
    #     info_values = [np.empty((self.T, 1, self.input_dims["info_" + key]), np.float32) for key in info_keys]

    #     for eps in range(self.num_demo):  # we initialize the whole demo buffer at the start of the training
    #         obs, acts, goals, achieved_goals = [], [], [], []
    #         i = 0
    #         for transition in range(self.T):
    #             obs.append([demo_data["obs"][eps][transition].get("observation")])
    #             acts.append([demo_data["acs"][eps][transition]])
    #             goals.append([demo_data["obs"][eps][transition].get("desired_goal")])
    #             achieved_goals.append([demo_data["obs"][eps][transition].get("achieved_goal")])
    #             for idx, key in enumerate(info_keys):
    #                 info_values[idx][transition, i] = demo_data["info"][eps][transition][key]

    #         obs.append([demo_data["obs"][eps][self.T].get("observation")])
    #         achieved_goals.append([demo_data["obs"][eps][self.T].get("achieved_goal")])

    #         episode = dict(o=obs, u=acts, g=goals, ag=achieved_goals)
    #         for key, value in zip(info_keys, info_values):
    #             episode["info_{}".format(key)] = value
    #         episode = convert_episode_to_batch_major(episode)
    #         # create the observation dict and append them into the demonstration buffer
    #         self.demo_buffer.store_episode(episode)
    #         if update_stats:
    #             self._update_stats(episode)
    #         episode.clear()
    #     logger.info("Demo buffer size: ", self.demo_buffer.get_current_size())

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
        episode_batch["n"] = np.ones((self.rollout_batch_size, self.T, 1), dtype=np.float32)
        # fill in the minimal value of q for rollout data.
        episode_batch["q"] = -100 * np.ones((self.rollout_batch_size, self.T, 1), dtype=np.float32)
        self.replay_buffer.store_episode(episode_batch)

        if update_stats:
            self._update_stats(episode_batch)

    def sample_batch(self):
        # use demonstration buffer to sample as well if demo flag is set TRUE
        if self.demo_actor != "none" or self.demo_critic == "rb":
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

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
            batch = [batch[key] for key in self.stage_shapes.keys()]
            self.current_batch = batch
        assert len(self.buffer_ph_tf) == len(batch)
        input_data = dict(zip(self.buffer_ph_tf, batch))
        self.sess.run(self.stage_op, feed_dict=input_data)

        # Debugging
        for k in input_data.keys():
            logger.debug("DDPG.stage_batch -> input of ", k, "is ", input_data[k].shape)
        logger.debug("DDPG.stage_batch -> Order should match stage_shapes.")

    def pre_train(self, stage=True):
        if stage:
            self.stage_batch()
        demo_loss, demo_grad = self.sess.run([self.demo_loss_tf, self.demo_grad_tf])
        self.demo_adam.update(demo_grad, self.Q_lr)
        return demo_loss

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        logger.debug("DDPG.train -> critic_loss:{}, actor_loss:{}".format(critic_loss, actor_loss))
        return critic_loss, actor_loss

    def check_train(self):
        """ For debugging only
        """
        self.stage_batch(self.current_batch)
        # method 0 rl only
        rl_q_var = self.sess.run(self.main.Q_var_tf)
        logger.info("DDPG.check_train -> rl variance {}".format(np.mean(rl_q_var)))

    def update_target_net(self):
        logger.debug("DDPG.update_target_net -> updating target net.")
        self.sess.run(self.update_target_net_op)

    def update_global_step(self):
        step = self.sess.run(self.global_step_inc_op)  # increase global step
        logger.debug("DDPG.train -> global step at {}".format(step))

    def get_current_buffer_size(self):
        return self.replay_buffer.get_current_size()

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()

    def logs(self, prefix=""):
        logs = []
        logs += [("stats_o/mean", np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [("stats_o/std", np.mean(self.sess.run([self.o_stats.std])))]
        logs += [("stats_g/mean", np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [("stats_g/std", np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not "" and not prefix.endswith("/"):
            return [(prefix + "/" + key, val) for key, val in logs]
        else:
            return logs

    def _create_network(self, reuse=False):
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # Mini-batch sampling.
        # Note: batch_tf == {"o":(None, *int), "o_2":(None, *int), "g":(None, *int), "g_2":(None, *int), "u":(None, *int), "r":(None)}
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        logger.debug("DDPG._create_network -> self.stage_shapes.keys() are {}".format(self.stage_shapes.keys()))

        # Creating a normalizer for goal and observation.
        with tf.variable_scope("o_stats") as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope("g_stats") as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # Networks
        target_batch_tf = batch_tf.copy()
        # The input to the target network has to be the resultant observation and goal!
        target_batch_tf["o"] = batch_tf["o_2"]
        if self.dimg != 0:
            target_batch_tf["g"] = batch_tf["g_2"]

        with tf.variable_scope("main") as vs:
            if reuse:
                vs.reuse_variables()
            self.main = ActorCritic(batch_tf, net_type="main", **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope("target") as vs:
            if reuse:
                vs.reuse_variables()
            self.target = ActorCritic(target_batch_tf, net_type="target", **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # Critic loss
        clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else self.clip_return)
        target_list_tf = [
            tf.clip_by_value(batch_tf["r"] + np.power(self.gamma, batch_tf["n"]) * target_Q_pi_tf, *clip_range)
            for target_Q_pi_tf in self.target.Q_pi_tf
        ]
        # target_min_tf = tf.reduce_min(target_list_tf, 0) # used for td3 training.
        self.Q_loss_tf = []
        self.demo_loss_tf = []
        self.global_step_tf = tf.get_variable("global_step", initializer=0.0, trainable=False, dtype=tf.float32)
        self.global_step_inc_op = tf.assign_add(self.global_step_tf, 1.0)
        if self.demo_critic in ["rb"]:
            # Method 1 choose only the demo buffer samples
            demo_mask = np.concatenate(
                (np.zeros(self.batch_size - self.batch_size_demo), np.ones(self.batch_size_demo)), axis=0
            ).reshape(-1, 1)
            rl_mask = np.concatenate(
                (np.ones(self.batch_size - self.batch_size_demo), np.zeros(self.batch_size_demo)), axis=0
            ).reshape(-1, 1)
            for i in range(self.num_sample):
                # rl loss
                rl_mse_tf = tf.boolean_mask(
                    tf.square(tf.stop_gradient(target_list_tf[i]) - self.main.Q_tf[i])
                    * tf.reshape(batch_tf["mask"][:, i], [-1, 1]),
                    rl_mask,
                )
                rl_loss_tf = tf.reduce_mean(rl_mse_tf)
                # demo loss
                demo_mse_tf = tf.boolean_mask(
                    # tf.square(batch_tf["q"] - self.main.Q_tf[i])
                    tf.square(tf.stop_gradient(target_list_tf[i]) - self.main.Q_tf[i])
                    * tf.reshape(batch_tf["mask"][:, i], [-1, 1]),
                    demo_mask,
                )
                demo_loss_tf = tf.reduce_mean(demo_mse_tf)

                # loss_tf = rl_loss_tf + demo_loss_tf
                self.Q_loss_tf.append(rl_loss_tf)
                self.demo_loss_tf.append(demo_loss_tf)
        else:
            for i in range(self.num_sample):
                self.max_q_tf = tf.stop_gradient(target_list_tf[i])
                loss_tf = tf.reduce_mean(
                    tf.square(self.max_q_tf - self.main.Q_tf[i]) * tf.reshape(batch_tf["mask"][:, i], [-1, 1])
                )
                self.Q_loss_tf.append(loss_tf)

        # Actor Loss
        if self.demo_actor != "none" and self.q_filter == 1:
            # train with demonstrations and use demo and q_filter both
            # where is the demonstrator action better than actor action according to the critic? choose those sample only
            mask = np.concatenate(
                (np.zeros(self.batch_size - self.batch_size_demo), np.ones(self.batch_size_demo)), axis=0
            )
            maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf > self.main.Q_pi_tf, mask), [-1])
            # define the cloning loss on the actor's actions only on the samples which adhere to the above masks
            self.cloning_loss_tf = tf.reduce_sum(
                tf.square(
                    tf.boolean_mask(tf.boolean_mask((self.main.pi_tf), mask), maskMain, axis=0)
                    - tf.boolean_mask(tf.boolean_mask((batch_tf["u"]), mask), maskMain, axis=0)
                )
            )
            # primary loss scaled by it's respective weight prm_loss_weight
            self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf)
            # L2 loss on action values scaled by the same weight prm_loss_weight
            self.pi_loss_tf += (
                self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
            )
            # adding the cloning loss to the actor loss as an auxilliary loss scaled by its weight aux_loss_weight
            self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf

        elif self.demo_actor != "none" and not self.q_filter:  # train with demonstrations without q_filter
            # choose only the demo buffer samples
            mask = np.concatenate(
                (np.zeros(self.batch_size - self.batch_size_demo), np.ones(self.batch_size_demo)), axis=0
            )
            self.cloning_loss_tf = tf.reduce_sum(
                tf.square(tf.boolean_mask((self.main.pi_tf), mask) - tf.boolean_mask((batch_tf["u"]), mask))
            )
            self.pi_loss_tf = -self.prm_loss_weight * tf.reduce_mean(self.main.Q_pi_tf[0])
            self.pi_loss_tf += (
                self.prm_loss_weight * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
            )
            self.pi_loss_tf += self.aux_loss_weight * self.cloning_loss_tf

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
        # gradients of demo
        if self.demo_critic == "rb":
            demo_grads_tf = tf.gradients(self.demo_loss_tf, self._vars("main/Q"))
            assert len(self._vars("main/pi")) == len(demo_grads_tf)
            self.demo_grads_vars_tf = zip(demo_grads_tf, self._vars("main/Q"))
            self.demo_grad_tf = flatten_grads(grads=demo_grads_tf, var_list=self._vars("main/Q"))

        # Optimizers
        self.Q_adam = MpiAdam(self._vars("main/Q"), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars("main/pi"), scale_grad_by_procs=False)
        if self.demo_critic == "rb":
            self.demo_adam = MpiAdam(self._vars("main/Q"), scale_grad_by_procs=False)

        # Polyak averaging
        self.main_vars = self._vars("main/Q") + self._vars("main/pi")
        self.target_vars = self._vars("target/Q") + self._vars("target/pi")
        self.stats_vars = self._global_vars("o_stats") + self._global_vars("g_stats")  # what is this used for?
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

        # initialize all variables
        tf.variables_initializer(self._global_vars("")).run()
        self._sync_optimizers()
        self._init_target_net()

    def _global_vars(self, scope=""):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/" + scope)
        return res

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run(
            [self.Q_loss_tf, self.pi_loss_tf, self.Q_grad_tf, self.pi_grad_tf]
        )
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def _preprocess_state(self, state):
        state = np.clip(state, -self.clip_obs, self.clip_obs)
        return state

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def _update_stats(self, episode_batch):
        # add transitions to normalizer
        episode_batch["o_2"] = episode_batch["o"][:, 1:, :]
        if self.dimg != 0:
            episode_batch["ag_2"] = episode_batch["ag"][:, 1:, :]
            episode_batch["g_2"] = episode_batch["g"][:, :, :]
        num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
        transitions = self.replay_buffer.sample_transitions(episode_batch, num_normalizing_transitions)

        transitions["o"] = self._preprocess_state(transitions["o"])
        self.o_stats.update(transitions["o"])
        self.o_stats.recompute_stats()
        if self.dimg != 0:
            transitions["g"] = self._preprocess_state(transitions["g"])
            self.g_stats.update(transitions["g"])
            self.g_stats.recompute_stats()

    def _vars(self, scope=""):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + "/" + scope)
        assert len(res) > 0
        return res

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = [
            "_tf",
            "_op",
            "_vars",
            "_adam",
            "_stats",
            "_buffer",
            "_transitions",  # sample
            "sess",
            "main",
            "target",
            "lock",
            "env",
            "stage_shapes",
        ]

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state["tf"] = self.sess.run([x for x in self._global_vars("") if "buffer" not in x.name])
        return state

    def __setstate__(self, state):
        # We don't need these for playing the policy.
        assert "sample_rl_transitions" not in state
        assert "sample_demo_transitions" not in state
        state["sample_rl_transitions"] = None
        state["sample_demo_transitions"] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == "_stats":
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars("") if "buffer" not in x.name]
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

    def query_ac_output(self, filename=None):
        """Check the output from actor and critic based on the action selected by the actor. Also check the output from
        the demonstration neural net if exists.

        This check can only be used for the Reach2D environment and its variants.
        """
        logger.info("Query: ac_output: getting Q value and actions over epochs.")

        num_point = 32
        ls = np.linspace(-1.2, 1.2, num_point)
        z1 = np.zeros((num_point * num_point, 1))
        z2 = np.zeros((num_point * num_point, 2))
        x, y = np.meshgrid(ls, ls)
        xr = x.reshape((-1, 1))
        yr = y.reshape((-1, 1))
        if self.dimo == 2:
            o = np.concatenate((xr, yr), 1)  # This has to be set accordingly
        elif self.dimo == 4:
            o = np.concatenate((z2, xr, yr), 1)
        else:
            assert False, "Observation dimension does not match!"
        g = z2
        ag = o
        o, g = self._preprocess_og(o, ag, g)
        u = z2
        q = z1

        policy = self.target
        feed = {policy.o_tf: o, policy.g_tf: g, policy.u_tf: u}
        # values to compute
        name = ["x", "y", "action", "rl_q"]
        queries = [policy.pi_tf, policy.Q_pi_mean_tf]
        ret = self.sess.run(queries, feed_dict=feed)
        ret[0] = np.mean(ret[0], axis=0)
        ret = [x, y] + ret
        if self.demo_policy is not None:
            transitions = {"u": ret[2], "o": o, "g": g, "q": q}
            demo_sample, demo_mean, demo_var = self.demo_policy.get_q_value(transitions)
            name = name + ["demo_q"]
            ret = ret + [demo_mean]

        if filename:
            logger.info("Query: ac_output: storing query results to {}".format(filename))
            np.savez_compressed(filename, **dict(zip(name, ret)))
        else:
            logger.info("Query: ac_output: ", ret)

    def query_critic_q(self, filename=None):
        """Given some (s,a) pairs, get the output from the current critic. This is used to generate some training data
        for the demonstration neutal net.

        This check can only be used for the Reach2D environment and its variants.
        """
        logger.info("Query: critic_q: getting Q output from target network for (s,a) pairs over epochs.")

        num_point = 8
        z1 = np.zeros((pow(num_point, 4), 1))
        z2 = np.zeros((pow(num_point, 4), 2))
        # Method 1
        ls = np.linspace(-1.0, 1.0, num_point)
        x, y, ax, ay = np.meshgrid(ls, ls, ls, ls)
        xr = x.reshape((-1, 1))
        yr = y.reshape((-1, 1))
        axr = ax.reshape((-1, 1))
        ayr = ay.reshape((-1, 1))
        if self.dimo == 2:
            o = np.concatenate((xr, yr), 1)  # This has to be set accordingly
        elif self.dimo == 4:
            o = np.concatenate((z2, xr, yr), 1)  # This has to be set accordingly
        else:
            assert False, "Observation dimension does not match!"
        g = z2
        ag = o
        o, g = self._preprocess_og(o, ag, g)
        u = np.concatenate((axr, ayr), axis=1)
        # Method 2
        # max_o = 1.0
        # max_g = 1.0
        # o = np.random.rand(pow(num_point, 4), self.dimo) * 2 * max_o - max_o
        # g = np.random.rand(pow(num_point, 4), self.dimg) * 2 * max_g - max_g
        # u = np.random.rand(pow(num_point, 4), self.dimu) * 2 * self.max_u - self.max_u
        # ag = o
        # o, g = self._preprocess_og(o, ag, g)

        policy = self.target
        feed = {policy.o_tf: o, policy.g_tf: g, policy.u_tf: u}
        name = ["o", "g", "u", "q"]
        queries = [policy.Q_mean_tf]
        ret = self.sess.run(queries, feed_dict=feed)
        ret[0] = ret[0].reshape((-1, 1))
        ret = [o, g, u] + ret

        logger.info("Query: critic_q: number of transition pairs is {}.".format(ret[0].shape[0]))

        if filename:
            logger.info("Query: critic_q: storing query results to {}".format(filename))
            np.savez_compressed(filename, **dict(zip(name, ret)))
        else:
            logger.info("Query: critic_q: ", ret)

    def query_uncertainty(self, filename=None):
        """Check the output from demonstration NN when the state is fixed and the action forms a 2d space.

        This check can only be used for the Reach2DFirstOrder environment.
        """
        logger.info("Query: uncertainty: Plot the uncertainty of demonstration NN.")

        num_point = 12
        z1 = np.zeros((pow(num_point, 2), 1))
        z2 = np.zeros((pow(num_point, 2), 2))
        ls = np.linspace(-2.0, 2.0, num_point)
        x, y = np.meshgrid(ls, ls)
        xr = x.reshape((-1, 1))
        yr = y.reshape((-1, 1))
        u = np.concatenate((xr, yr), 1)  # This has to be set accordingly
        q = z1
        g = z2

        policy = self.target
        name = ["x", "y", "o", "g", "rl_q"]
        if self.demo_policy is not None:
            name = name + ["demo_q_mean", "demo_q_var"]
        o1, o2 = np.meshgrid(np.linspace(-0.8, 0.8, 4), np.linspace(-0.8, 0.8, 4))
        o1 = o1.reshape((-1, 1))
        o2 = o2.reshape((-1, 1))
        os = np.concatenate((o1, o2), 1)
        rl_q = []
        demo_q_mean = []
        demo_q_var = []
        o_sample = []
        for o in os:
            o = np.repeat(o.reshape((1, -1)), [z1.shape[0]], axis=0)
            ag = o
            o, g = self._preprocess_og(o, ag, g)
            o_sample.append(o)
            feed = {policy.o_tf: o, policy.g_tf: g, policy.u_tf: u}
            queries = policy.Q_mean_tf
            rl_q.append(self.sess.run(queries, feed_dict=feed).reshape((1, -1, 1)))
            if self.demo_policy is not None:
                transitions = {"u": u, "o": o, "g": g, "q": q}
                demo_sample, demo_mean, demo_var = self.demo_policy.get_q_value(transitions)
                demo_q_mean.append(demo_mean.reshape((1, -1, 1)))
                demo_q_var.append(demo_var.reshape((1, -1, 1)))
        ret = [x, y, o_sample, g] + [np.concatenate((rl_q), axis=0)]
        if self.demo_policy is not None:
            ret = ret + [np.concatenate((demo_q_mean), axis=0), np.concatenate((demo_q_var), axis=0)]

        if filename:
            logger.info("Query: uncertainty: storing query results to {}".format(filename))
            np.savez_compressed(filename, **dict(zip(name, ret)))
        else:
            logger.info("Query: uncertainty: ", ret)
