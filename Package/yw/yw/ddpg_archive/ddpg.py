import numpy as np
import tensorflow as tf

from yw.tool import logger

from yw.ddpg_main.mpi_adam import MpiAdam
from yw.ddpg_main.normalizer import Normalizer
from yw.ddpg_main.actor_critic import Actor, Critic, ActorCritic
from yw.ddpg_main.replay_buffer import *
from yw.util.util import store_args, import_function
from yw.util.tf_util import flatten_grads, flatgrad


class DDPG(object):
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
        info,
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
        # Store initial args passed into the function
        self.init_args = locals()

        # Parameters
        self.input_dims = input_dims
        self.num_sample = num_sample
        self.ca_ratio = ca_ratio
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
        self.sample_rl_transitions = sample_rl_transitions
        self.sample_demo_transitions = sample_demo_transitions
        self.gamma = gamma
        self.info = info
        self.reuse = reuse

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
        # for demonstrations
        buffer_shapes["n"] = (self.T, 1)  # n step return
        buffer_shapes["q"] = (self.T, 1)  # expected q value
        # for bootstrapped ensemble of actor critics
        buffer_shapes["mask"] = (self.T, self.num_sample)  # mask for training each head with different dataset
        # add extra information
        for key, val in input_dims.items():
            if key.startswith("info"):
                buffer_shapes[key] = (self.T, *(tuple([val]) if val > 0 else tuple()))
        logger.debug("DDPG.__init__ -> The buffer shapes are: {}".format(buffer_shapes))

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        if not self.sample_rl_transitions:
            pass
        elif self.sample_rl_transitions["strategy"] == "her":
            self.replay_buffer = HERReplayBuffer(buffer_shapes, buffer_size, self.T, **self.sample_rl_transitions["args"])
        else:
            self.replay_buffer = UniformReplayBuffer(buffer_shapes, buffer_size, self.T, **self.sample_rl_transitions["args"])

        # initialize the demo buffer; in the same way as the primary data buffer
        if not self.sample_demo_transitions:
            pass
        elif self.demo_actor != "none" or self.demo_critic == "rb":
            self.demo_buffer = NStepReplayBuffer(buffer_shapes, buffer_size, self.T, **self.sample_demo_transitions)

        logger.info("Creating a DDPG agent with action space %d x %s." % (self.dimu, self.max_u))
        # Build computation core.
        with tf.variable_scope(self.scope):
            self._create_network()



    def get_actions(self, o, ag, g, noise_eps=0.0, random_eps=0.0, use_target_net=False, compute_Q=False):

        policy = self.main

        # values to compute
        vals = [policy.pi_tf, policy.Q_pi_mean_tf]
        # feed
        feed = {self.inputs["o"]: self._preprocess_state(o)}
        if self.dimg != 0:
            feed[self.inputs["g"]] = self._preprocess_state(g)
        ret = self.sess.run(vals, feed_dict=feed)

        # action postprocessing
        if use_target_net:  # base on the assumption that target net is only used when testing
            # randomly select the output action of one policy.
            u = np.mean(ret[0], axis=0) * self.max_u
        else:
            u = ret[0][np.random.randint(self.num_sample)] * self.max_u

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

    def pre_train(self, stage=True):

        batch = self.sample_batch()

        # Get all gradients and perform a synced update.
        ops = [self.demo_critic_loss, self.demo_critic_grads]
        feeds = {self.inputs[k]: batch[k] for k in self.inputs.keys()}
        demo_loss, demo_grad = self.sess.run(ops, feed_dict=feeds)
        self.demo_critic_optimizer.update(demo_grad, stepsize=self.Q_lr)

        return demo_loss

    def train(self, stage=True):

        batch = self.sample_batch()

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        feeds = {self.inputs[k]: batch[k] for k in self.inputs.keys()}
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict=feeds)
        self.actor_optimizer.update(actor_grads, stepsize=self.pi_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.Q_lr)

        return critic_loss, actor_loss

    def check_train(self):
        """ For debugging only
        """
        pass

    def init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        logger.debug("DDPG.update_target_net -> updating target net.")
        self.sess.run(self.update_target_net_op)

    def update_global_step(self):
        pass
        # step = self.sess.run(self.global_step_inc_op)  # increase global step
        # logger.debug("DDPG.train -> global step at {}".format(step))

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


    def _create_network(self):
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        self.inputs = {}
        self.inputs["o"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.inputs["o_2"] = tf.placeholder(tf.float32, shape=(None, self.dimo))
        self.inputs["u"] = tf.placeholder(tf.float32, shape=(None, self.dimu))
        self.inputs["r"] = tf.placeholder(tf.float32, shape=(None, 1))
        # for multigoal environments, should be merged with inputs o
        if self.dimg != 0:
            self.inputs["g"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
            self.inputs["g_2"] = tf.placeholder(tf.float32, shape=(None, self.dimg))
        # for demonstrations -> ideally this should also go into replay buffer
        self.inputs["q"] = tf.placeholder(tf.float32, shape=(None, 1))
        self.inputs["n"] = tf.placeholder(tf.float32, shape=(None, 1))
        # for bootstrapped DQN, to add mask TODO move this to another scope
        self.inputs["mask"] = tf.placeholder(tf.float32, shape=(None, self.num_sample))

        # Creating a normalizer for goal and observation.
        with tf.variable_scope("o_stats") as vs:
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope("g_stats") as vs:
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # Prepare inputs for actor and critic.
        normalized_o = self.o_stats.normalize(self.inputs["o"])
        normalized_o_2 = self.o_stats.normalize(self.inputs["o_2"])
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            normalized_g = self.g_stats.normalize(self.inputs["g"])
            normalized_g_2 = self.g_stats.normalize(self.inputs["g_2"])
            normalized_o = tf.concat(axis=1, values=[normalized_o, normalized_g])
            normalized_o_2 = tf.concat(axis=1, values=[normalized_o_2, normalized_g_2])
        normalized_u = self.inputs["u"] / self.max_u

        # Main networks
        with tf.variable_scope("main") as vs:
            self.main = ActorCritic(normalized_o,normalized_u,num_sample=self.num_sample,hidden=self.hidden,layers=self.layers)
        with tf.variable_scope("target") as vs:
            self.target = ActorCritic(normalized_o_2,normalized_u,num_sample=self.num_sample,hidden=self.hidden,layers=self.layers)
        assert len(self._vars("main")) == len(self._vars("target"))

        # Critic loss
        clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else self.clip_return)
        target_Q = [
            tf.clip_by_value(self.inputs["r"] + tf.pow(self.gamma, self.inputs["n"]) * target_Q_pi_tf, *clip_range)
            for target_Q_pi_tf in self.target.Q_pi_tf
        ]
        self.critic_loss = []
        self.demo_critic_loss = []
        # self.global_step_tf = tf.get_variable("global_step", initializer=0.0, trainable=False, dtype=tf.float32)
        # self.global_step_inc_op = tf.assign_add(self.global_step_tf, 1.0)
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
                    tf.square(tf.stop_gradient(target_Q[i]) - self.main.Q_tf[i])
                    * tf.reshape(self.inputs["mask"][:, i], [-1, 1]),
                    rl_mask,
                )
                rl_loss_tf = tf.reduce_mean(rl_mse_tf)
                # demo loss
                demo_mse_tf = tf.boolean_mask(
                    tf.square(tf.stop_gradient(target_Q[i]) - self.main.Q_tf[i])
                    * tf.reshape(self.inputs["mask"][:, i], [-1, 1]),
                    demo_mask,
                )
                demo_loss_tf = tf.reduce_mean(demo_mse_tf)

                # loss_tf = rl_loss_tf + demo_loss_tf
                self.critic_loss.append(rl_loss_tf)
                self.demo_critic_loss.append(demo_loss_tf)
        else:
            for i in range(self.num_sample):
                self.max_q_tf = tf.stop_gradient(target_Q[i])
                loss_tf = tf.reduce_mean(
                    tf.square(self.max_q_tf - self.main.Q_tf[i]) * tf.reshape(self.inputs["mask"][:, i], [-1, 1])
                )
                self.critic_loss.append(loss_tf)

        # Actor Loss
        self.actor_loss = [
            -tf.reduce_mean(self.main.Q_pi_tf[i])
            + self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf[i] / self.max_u))
            for i in range(self.num_sample)
        ]

        # Gradients
        # gradients of Q
        critic_grads = tf.gradients(self.critic_loss, self._vars("main/Q"))
        assert len(self._vars("main/Q")) == len(critic_grads)
        self.critic_grads_vars = zip(critic_grads, self._vars("main/Q"))
        self.critic_grads = flatten_grads(grads=critic_grads, var_list=self._vars("main/Q"))
        # gradients of pi
        actor_grads = tf.gradients(self.actor_loss, self._vars("main/pi"))
        assert len(self._vars("main/pi")) == len(actor_grads)
        self.actor_grads_vars = zip(actor_grads, self._vars("main/pi"))
        self.actor_grads = flatten_grads(grads=actor_grads, var_list=self._vars("main/pi"))
        # gradients of demo
        if self.demo_critic == "rb":
            demo_critic_grads = tf.gradients(self.demo_critic_loss, self._vars("main/Q"))
            assert len(self._vars("main/pi")) == len(demo_critic_grads)
            self.demo_critic_grads_vars = zip(demo_critic_grads, self._vars("main/Q"))
            self.demo_critic_grads = flatten_grads(grads=demo_critic_grads, var_list=self._vars("main/Q"))

        # Optimizers
        self.critic_optimizer = MpiAdam(self._vars("main/Q"), scale_grad_by_procs=False)
        self.actor_optimizer = MpiAdam(self._vars("main/pi"), scale_grad_by_procs=False)
        if self.demo_critic == "rb":
            self.demo_critic_optimizer = MpiAdam(self._vars("main/Q"), scale_grad_by_procs=False)

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
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.init_target_net_op)

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
            episode_batch["ag_2"] = episode_batch["ag"][:, 1:, :]
            episode_batch["g_2"] = episode_batch["g"][:, :, :]
        num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
        transitions = self.replay_buffer._sample_transitions(episode_batch, num_normalizing_transitions)

        self.o_stats.update(self._preprocess_state(transitions["o"]))
        self.o_stats.recompute_stats()
        if self.dimg != 0:
            self.g_stats.update(self._preprocess_state(transitions["g"]))
            self.g_stats.recompute_stats()

    def __getstate__(self):

        # """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        # """
        excluded_names = ["self", "sample_demo_transitions", "sample_rl_transitions"]

        state = {k: v for k, v in self.init_args.items() if not k in excluded_names}
        state["tf"] = self.sess.run([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        return state

    def __setstate__(self, state):

        # We don't need these for playing the policy.
        assert "sample_rl_transitions" not in state
        assert "sample_demo_transitions" not in state
        state["sample_rl_transitions"] = None
        state["sample_demo_transitions"] = None

        kwargs = state["kwargs"]
        del state["kwargs"]

        self.__init__(**state, **kwargs)
        vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
        assert len(vars) == len(state["tf"])
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
