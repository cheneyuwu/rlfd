import numpy as np
import tensorflow as tf

# from tensorflow.contrib.staging import StagingArea
from collections import OrderedDict

from yw.tool import logger

from yw.ddpg_tf2.mpi_adam import MpiAdam
from yw.ddpg_tf2.normalizer import Normalizer
from yw.ddpg_main.replay_buffer import ReplayBuffer
from yw.ddpg_tf2.actor_critic import ActorCritic
from yw.util.util import store_args, import_function


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
        subtract_goals,
        relative_goals,
        clip_pos_returns,
        clip_return,
        demo_critic,
        demo_actor,
        demo_policy,
        num_demo,
        q_filter,
        batch_size_demo,
        prm_loss_weight,
        aux_loss_weight,
        sample_transitions,
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
            subtract_goals     (function)     - (NOT USED) function that subtracts goals from each other
            relative_goals     (boolean)      - (NOT USED) whether or not relative goals should be fed into the network
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
            sample_transitions (function)     - function that samples from the replay buffer
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
        input_shapes = {key: tuple([val]) if val > 0 else tuple() for key, val in self.input_dims.items()}

        logger.info("Configuring the replay buffer.")
        buffer_shapes = {
            key: (self.T if key != "o" else self.T + 1, *input_shapes[key]) for key, val in input_shapes.items()
        }
        buffer_shapes["g"] = (buffer_shapes["g"][0], self.dimg)
        buffer_shapes["ag"] = (self.T + 1, self.dimg)
        buffer_shapes["mask"] = (self.T, self.num_sample)  # mask for training each head with different dataset
        buffer_shapes["q"] = (self.T, 1)  # expected q value
        logger.debug("DDPG.__init__ -> The buffer shapes are: {}".format(buffer_shapes))
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        logger.debug("DDPG.__init__ -> The buffer size is: {}".format(buffer_size))
        self.replay_buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)
        if self.demo_actor != "none":
            # initialize the demo buffer; in the same way as the primary data buffer
            self.demo_buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

        logger.info("Creating a DDPG agent with action space %d x %s." % (self.dimu, self.max_u))

        # Creating a normalizer for goal and observation.
        self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip)
        self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip)

        # Create actor critic
        self.main_tf = ActorCritic(**vars(self))
        self.target_tf = ActorCritic(**vars(self))

        # Loss functions
        def compute_critic_loss(input):
            clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else self.clip_return)
            target_value = tf.clip_by_value(
                input["r"] + self.gamma * self.target_tf.get_q_pi({"o": input["o_2"], "g": input["g_2"]}), *clip_range
            )
            critic_loss = tf.losses.mean_squared_error(
                target_value, self.main_tf.get_q({"o": input["o"], "g": input["g"], "u": input["u"]})
            )

            return tf.reduce_mean(critic_loss)

        self.critic_loss_op = compute_critic_loss

        def compute_actor_loss(input):
            actor_loss = -tf.reduce_mean(
                self.main_tf.get_q_pi({"o": input["o"], "g": input["g"]})
            ) + self.action_l2 * tf.reduce_mean(
                tf.square(self.main_tf.get_pi({"o": input["o"], "g": input["g"]}) / self.max_u)
            )
            return actor_loss

        self.actor_loss_op = compute_actor_loss

        # Optimizers
        self.Q_adam = MpiAdam(self.main_tf.get_critic_vars(), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self.main_tf.get_actor_vars(), scale_grad_by_procs=False)
        self.Q_adam.sync()
        self.pi_adam.sync()

        # Sync Target net
        self.target_tf.set_weights(self.main_tf.get_weights())

        # # Loss functions
        # clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else self.clip_return)
        # target_list_tf = [
        #     tf.clip_by_value(batch_tf["r"] + self.gamma * target_Q_pi_tf, *clip_range)
        #     for target_Q_pi_tf in self.target.Q_pi_tf
        # ]
        # # target_min_tf = tf.reduce_min(target_list_tf, 0) # used for td3 training.
        # self.Q_loss_tf = []

        # for i in range(self.num_sample):
        #     self.max_q_tf = tf.stop_gradient(target_list_tf[i])
        #     loss_tf = tf.reduce_mean(
        #         tf.square(self.max_q_tf - self.main.Q_tf[i]) * tf.reshape(batch_tf["mask"][:, i], [-1, 1])
        #     )
        #     self.Q_loss_tf.append(loss_tf)

        # self.pi_loss_tf = [
        #     -tf.reduce_mean(self.main.Q_pi_tf[i])
        #     + self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf[i] / self.max_u))
        #     for i in range(self.num_sample)
        # ]

    # def get_q_value(self, batch):
    #     policy = self.target
    #     # values to compute
    #     vals = [policy.Q_sample_tf, policy.Q_mean_tf, policy.Q_var_tf]
    #     # feed
    #     feed = {
    #         policy.o_tf: batch["o"].reshape(-1, self.dimo),
    #         policy.g_tf: batch["g"].reshape(-1, self.dimg),
    #         policy.u_tf: batch["u"].reshape(-1, self.dimu),
    #     }
    #     return self.sess.run(vals, feed_dict=feed)

    def get_actions(self, o, ag, g, noise_eps=0.0, random_eps=0.0, use_target_net=False, compute_Q=False):

        o, g = self._preprocess_og(o, ag, g)

        policy = self.target_tf if use_target_net else self.main_tf

        u = policy.get_pi({"o": o, "g": g}).numpy()
        exp_q = policy.get_q_pi({"o": o, "g": g}).numpy()

        ret = [u, exp_q]

        # # action postprocessing
        # if use_target_net:  # base on the assumption that target net is only used when testing
        #     # randomly select the output action of one policy.
        #     u = np.mean(ret[0], axis=0)
        # else:
        #     u = ret[0][np.random.randint(self.num_sample)]

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

        if self.demo_policy is not None:
            transitions = {"u": u, "o": o, "g": g, "q": np.zeros(ret[1].shape).reshape((-1, 1))}
            demo_sample, demo_mean, demo_var = self.demo_policy.get_q_value(transitions)
            logger.debug("DDPG.get_actions -> The shape of demo mean is: {}".format(demo_mean.shape))
            logger.debug("DDPG.get_actions -> The shape of demo variance is: {}".format(demo_var.shape))
            # value debug: check the Q from demo and rl are similar
            # logger.info("DDPG.get_actions -> The output from demo mean is: {}".format(demo_mean-demo_var))

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

        mask = np.random.binomial(1, 1, (self.num_demo, self.T, self.num_sample))
        episode_batch["mask"] = mask
        # expected_q = np.zeros((self.num_demo, self.T, 1)) # fill in the minimal value of q for rollout data.
        # episode_batch["q"] = expected_q
        self.demo_buffer.store_episode(episode_batch)

        if update_stats:
            self._update_stats(episode_batch)

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        # mask transitions for each bootstrapped head
        # select a distribution!
        for key in episode_batch.keys():
            assert episode_batch[key].shape[0] == self.rollout_batch_size
        mask = np.random.binomial(
            1, 1, (self.rollout_batch_size, self.T, self.num_sample)
        )  # choose your own distribution!
        episode_batch["mask"] = mask
        # fill in the minimal value of q for rollout data.
        expected_q = -100 * np.ones((self.rollout_batch_size, self.T, 1))
        episode_batch["q"] = expected_q
        self.replay_buffer.store_episode(episode_batch)

        if update_stats:
            self._update_stats(episode_batch)

    def sample_batch(self):
        if self.demo_actor != "none":  # use demonstration buffer to sample as well if demo flag is set TRUE
            transitions = {}
            transition_rollout = self.replay_buffer.sample(self.batch_size - self.batch_size_demo)
            transition_demo = self.demo_buffer.sample(self.batch_size_demo)
            assert transition_rollout.keys() == transition_demo.keys()
            for k in transition_rollout.keys():
                transitions[k] = np.concatenate((transition_rollout[k], transition_demo[k]))
        else:
            transitions = self.replay_buffer.sample(self.batch_size)  # otherwise only sample from primary buffer

        o, o_2, g = transitions["o"], transitions["o_2"], transitions["g"]
        ag, ag_2 = transitions["ag"], transitions["ag_2"]
        transitions["o"], transitions["g"] = self._preprocess_og(o, ag, g)
        transitions["o_2"], transitions["g_2"] = self._preprocess_og(o_2, ag_2, g)
        if self.demo_critic != "none":

            # Method 1 calculate q_d directly
            output_sample, output_mean, output_var = self.demo_policy.get_q_value(transitions)

            # Method 2 calculate q_d for the next state and then
            # policy = self.target
            # # feed
            # feed = {policy.o_tf: transitions["o_2"], policy.g_tf: transitions["g"], policy.u_tf: transitions["u"]}
            # # values to compute
            # ret = self.sess.run(policy.pi_tf, feed_dict=feed)
            # u = np.mean(ret, axis=0)
            # demo_input = transitions.copy()
            # demo_input["u"] = u
            # output_sample, output_mean, output_var = self.demo_policy.get_q_value(demo_input)
            # output_sample = output_sample.reshape((-1, self.demo_policy.num_sample)) * self.gamma + transitions["r"]
            # output_mean = output_mean.reshape((-1, 1)) * self.gamma + transitions["r"]

            transitions["q_sample"] = output_sample.reshape((-1, self.demo_policy.num_sample))
            transitions["q_mean"] = output_mean.reshape((-1, 1))
            transitions["q_var"] = output_var.reshape((-1, 1))

        # transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        # logger.debug(
        #     "DDPG.sample_batch -> The sampled batch shape is: {}".format(
        #         {k: transitions[k].shape for k in self.stage_shapes.keys()}
        #     )
        # )

        return transitions

    def train(self):

        # Sample a batch
        batch = self.sample_batch()

        # Update
        self.Q_adam.update(self.critic_loss_op, input={"input":batch})
        self.pi_adam.update(self.actor_loss_op, input={"input":batch})

    # def check_train(self):
    #     """ For debugging only
    #     """
    #     self.stage_batch(self.current_batch)
    #     # method 0 rl only
    #     rl_q_var = self.sess.run(
    #         self.main.Q_var_tf
    #     )
    #     logger.info("DDPG.check_train -> rl variance {}".format(np.mean(rl_q_var)))

    #     # method 1 using both rl and demo uncertainty
    #     # critic_loss, actor_loss, weight, rl_q, demo_q, rl_q_sample, demo_q_sample = self.sess.run(
    #     #     [self.Q_loss_tf, self.pi_loss_tf, self.weight_tf, self.rl_certainty_tf, self.demo_certainty_tf, self.rl_q_sample_tf, self.demo_q_sample_tf]
    #     # )

    #     # method 2 using demo uncertainty only
    #     # critic_loss, actor_loss, weight, demo_q = self.sess.run(
    #     #     [self.Q_loss_tf, self.pi_loss_tf, self.weight_tf, self.demo_q_var_tf]
    #     # )
    #     # logger.debug("DDPG.check_train -> critic_loss:{}, actor_loss:{}".format(critic_loss, actor_loss))
    #     # for i in range(10):
    #     #     logger.info("DDPG.check_train -> weight:{}, demo_var: {}".format(np.mean(weight), demo_q[i]))
    #         # logger.info("DDPG.check_train -> rl_q_sample: {}".format(rl_q_sample[i]))
    #         # logger.info("DDPG.check_train -> demo_q_sample: {}".format(demo_q_sample[i]))
    #         # logger.info(
    #         #     "DDPG.check_train -> {}: rl_q, demo_q: {} {} {}".format(i, rl_q[i], demo_q[i], rl_q[i] > demo_q[i])
    #         # )

    def update_target_net(self):
        logger.debug("DDPG.update_target_net -> updating target net.")
        main_weights = self.main_tf.get_weights()
        target_weights = self.target_tf.get_weights()
        # polyak averaging
        avg_weights = [self.polyak * v[0] + (1.0 - self.polyak) * v[1] for v in zip(target_weights, main_weights)]
        self.target_tf.set_weights(avg_weights)

    def update_global_step(self):
        pass

    def get_current_buffer_size(self):
        return self.replay_buffer.get_current_size()

    def clear_buffer(self):
        self.replay_buffer.clear_buffer()

    def logs(self, prefix=""):
        logs = []
        # logs += [("stats_o/mean", np.mean(self.sess.run([self.o_stats.mean])))]
        # logs += [("stats_o/std", np.mean(self.sess.run([self.o_stats.std])))]
        # logs += [("stats_g/mean", np.mean(self.sess.run([self.g_stats.mean])))]
        # logs += [("stats_g/std", np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not "" and not prefix.endswith("/"):
            return [(prefix + "/" + key, val) for key, val in logs]
        else:
            return logs

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:  # note: this is never used
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _update_stats(self, episode_batch):
        # add transitions to normalizer
        episode_batch["o_2"] = episode_batch["o"][:, 1:, :]
        episode_batch["ag_2"] = episode_batch["ag"][:, 1:, :]
        num_normalizing_transitions = episode_batch["u"].shape[0] * episode_batch["u"].shape[1]
        transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

        o, o_2, g, ag = transitions["o"], transitions["o_2"], transitions["g"], transitions["ag"]
        transitions["o"], transitions["g"] = self._preprocess_og(o, ag, g)
        # No need to preprocess the o_2 and g_2 since this is only used for stats

        # self.o_stats.update(transitions["o"])
        # self.g_stats.update(transitions["g"])

        # self.o_stats.recompute_stats()
        # self.g_stats.recompute_stats()

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = [
            "_tf",
            "_op",
            "_adam",
            "_stats",
            "_buffer",
            "sample_transitions",
        ]

        state = {k: v for k, v in vars(self).items() if all([not subname in k for subname in excluded_subnames])}
        state["tf"] = {"main": self.main_tf.get_weights(), "target": self.target_tf.get_weights()}
        return state

    def __setstate__(self, state):

        # We don't need these for playing the policy.
        if "sample_transitions" not in state:
            state["sample_transitions"] = None

        self.__init__(**state)

        # load TF variables
        self.main_tf.set_weights(state["tf"]["main"])
        self.target_tf.set_weights(state["tf"]["target"])

    # ###########################################
    # # Queries
    # ###########################################

    # # Store the result you want to query in a npz file.

    # # Currently you can query the value of:
    # #     staging_tf          -> staged values
    # #     buffer_ph_tf        -> placeholders
    # #     demo_q_mean_tf      ->
    # #     demo_q_var_tf       ->
    # #     demo_q_sample_tf    ->
    # #     max_q_tf            ->
    # #     Q_loss_tf           -> a list of losses for each critic sample
    # #     pi_loss_tf          -> a list of losses for each pi sample
    # #     policy.pi_tf        -> list of output from actor
    # #     policy.Q_tf         ->
    # #     policy.Q_pi_tf      ->
    # #     policy.Q_pi_mean_tf ->
    # #     policy.Q_pi_var_tf  ->
    # #     policy._input_Q     ->

    # def query_ac_output(self, filename=None):
    #     """Check the output from actor and critic based on the action selected by the actor. Also check the output from
    #     the demonstration neural net if exists.

    #     This check can only be used for the Reach2D environment and its variants.
    #     """
    #     logger.info("Query: ac_output: getting Q value and actions over epochs.")

    #     num_point = 32
    #     ls = np.linspace(-1.2, 1.2, num_point)
    #     z1 = np.zeros((num_point * num_point, 1))
    #     z2 = np.zeros((num_point * num_point, 2))
    #     x, y = np.meshgrid(ls, ls)
    #     xr = x.reshape((-1, 1))
    #     yr = y.reshape((-1, 1))
    #     if self.dimo == 2:
    #         o = np.concatenate((xr, yr), 1)  # This has to be set accordingly
    #     elif self.dimo == 4:
    #         o = np.concatenate((z2, xr, yr), 1)
    #     else:
    #         assert False, "Observation dimension does not match!"
    #     g = z2
    #     ag = o
    #     o, g = self._preprocess_og(o, ag, g)
    #     u = z2
    #     q = z1

    #     policy = self.target
    #     feed = {policy.o_tf: o, policy.g_tf: g, policy.u_tf: u}
    #     # values to compute
    #     name = ["x", "y", "action", "rl_q"]
    #     queries = [policy.pi_tf, policy.Q_pi_mean_tf]
    #     ret = self.sess.run(queries, feed_dict=feed)
    #     ret[0] = np.mean(ret[0], axis=0)
    #     ret = [x, y] + ret
    #     if self.demo_policy is not None:
    #         transitions = {"u": ret[2], "o": o, "g": g, "q": q}
    #         demo_sample, demo_mean, demo_var = self.demo_policy.get_q_value(transitions)
    #         name = name + ["demo_q"]
    #         ret = ret + [demo_mean]

    #     if filename:
    #         logger.info("Query: ac_output: storing query results to {}".format(filename))
    #         np.savez_compressed(filename, **dict(zip(name, ret)))
    #     else:
    #         logger.info("Query: ac_output: ", ret)

    # def query_critic_q(self, filename=None):
    #     """Given some (s,a) pairs, get the output from the current critic. This is used to generate some training data
    #     for the demonstration neutal net.

    #     This check can only be used for the Reach2D environment and its variants.
    #     """
    #     logger.info("Query: critic_q: getting Q output from target network for (s,a) pairs over epochs.")

    #     num_point = 8
    #     z1 = np.zeros((pow(num_point, 4), 1))
    #     z2 = np.zeros((pow(num_point, 4), 2))
    #     # Method 1
    #     ls = np.linspace(-1.0, 1.0, num_point)
    #     x, y, ax, ay = np.meshgrid(ls, ls, ls, ls)
    #     xr = x.reshape((-1, 1))
    #     yr = y.reshape((-1, 1))
    #     axr = ax.reshape((-1, 1))
    #     ayr = ay.reshape((-1, 1))
    #     if self.dimo == 2:
    #         o = np.concatenate((xr, yr), 1)  # This has to be set accordingly
    #     elif self.dimo == 4:
    #         o = np.concatenate((z2, xr, yr), 1)  # This has to be set accordingly
    #     else:
    #         assert False, "Observation dimension does not match!"
    #     g = z2
    #     ag = o
    #     o, g = self._preprocess_og(o, ag, g)
    #     u = np.concatenate((axr, ayr), axis=1)
    #     # Method 2
    #     # max_o = 1.0
    #     # max_g = 1.0
    #     # o = np.random.rand(pow(num_point, 4), self.dimo) * 2 * max_o - max_o
    #     # g = np.random.rand(pow(num_point, 4), self.dimg) * 2 * max_g - max_g
    #     # u = np.random.rand(pow(num_point, 4), self.dimu) * 2 * self.max_u - self.max_u
    #     # ag = o
    #     # o, g = self._preprocess_og(o, ag, g)

    #     policy = self.target
    #     feed = {policy.o_tf: o, policy.g_tf: g, policy.u_tf: u}
    #     name = ["o", "g", "u", "q"]
    #     queries = [policy.Q_mean_tf]
    #     ret = self.sess.run(queries, feed_dict=feed)
    #     ret[0] = ret[0].reshape((-1, 1))
    #     ret = [o, g, u] + ret

    #     logger.info("Query: critic_q: number of transition pairs is {}.".format(ret[0].shape[0]))

    #     if filename:
    #         logger.info("Query: critic_q: storing query results to {}".format(filename))
    #         np.savez_compressed(filename, **dict(zip(name, ret)))
    #     else:
    #         logger.info("Query: critic_q: ", ret)

    # def query_uncertainty(self, filename=None):
    #     """Check the output from demonstration NN when the state is fixed and the action forms a 2d space.

    #     This check can only be used for the Reach2DFirstOrder environment.
    #     """
    #     logger.info("Query: uncertainty: Plot the uncertainty of demonstration NN.")

    #     num_point = 12
    #     z1 = np.zeros((pow(num_point, 2), 1))
    #     z2 = np.zeros((pow(num_point, 2), 2))
    #     ls = np.linspace(-2.0, 2.0, num_point)
    #     x, y = np.meshgrid(ls, ls)
    #     xr = x.reshape((-1, 1))
    #     yr = y.reshape((-1, 1))
    #     u = np.concatenate((xr, yr), 1)  # This has to be set accordingly
    #     q = z1
    #     g = z2

    #     policy = self.target
    #     name = ["x", "y", "o", "g", "rl_q"]
    #     if self.demo_policy is not None:
    #         name = name + ["demo_q_mean", "demo_q_var"]
    #     o1, o2 = np.meshgrid(np.linspace(-0.8, 0.8, 4), np.linspace(-0.8, 0.8, 4))
    #     o1 = o1.reshape((-1, 1))
    #     o2 = o2.reshape((-1, 1))
    #     os = np.concatenate((o1, o2), 1)
    #     rl_q = []
    #     demo_q_mean = []
    #     demo_q_var = []
    #     o_sample = []
    #     for o in os:
    #         o = np.repeat(o.reshape((1, -1)), [z1.shape[0]], axis=0)
    #         ag = o
    #         o, g = self._preprocess_og(o, ag, g)
    #         o_sample.append(o)
    #         feed = {policy.o_tf: o, policy.g_tf: g, policy.u_tf: u}
    #         queries = policy.Q_mean_tf
    #         rl_q.append(self.sess.run(queries, feed_dict=feed).reshape((1, -1, 1)))
    #         if self.demo_policy is not None:
    #             transitions = {"u": u, "o": o, "g": g, "q": q}
    #             demo_sample, demo_mean, demo_var = self.demo_policy.get_q_value(transitions)
    #             demo_q_mean.append(demo_mean.reshape((1, -1, 1)))
    #             demo_q_var.append(demo_var.reshape((1, -1, 1)))
    #     ret = [x, y, o_sample, g] + [np.concatenate((rl_q), axis=0)]
    #     if self.demo_policy is not None:
    #         ret = ret + [np.concatenate((demo_q_mean), axis=0), np.concatenate((demo_q_var), axis=0)]

    #     if filename:
    #         logger.info("Query: uncertainty: storing query results to {}".format(filename))
    #         np.savez_compressed(filename, **dict(zip(name, ret)))
    #     else:
    #         logger.info("Query: uncertainty: ", ret)
