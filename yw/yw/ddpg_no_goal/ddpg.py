# =============================================================================
# Import
# =============================================================================
# Infra import
import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from collections import OrderedDict

# DDPG Package import
from yw import logger
from yw.ddpg.util import (
    import_function,
    reshape_for_broadcasting,
    store_args,
    transitions_in_episode_batch,
    dims_to_shapes,
)
from yw.ddpg.mpi_adam import MpiAdam
from yw.ddpg_no_goal.replay_buffer import ReplayBuffer

from yw.util.tf_util import flatten_grads


class DDPG(object):
    @store_args
    def __init__(
        self,
        input_dims,
        buffer_size,
        hidden,
        layers,
        network_class,
        polyak,
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
        sample_transitions,
        gamma,
        reuse=False,
        **kwargs
    ):
        """
        Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.

        Args:
            # Environment I/O and Config
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            T (int): the time horizon for rollouts
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network

            # Normalizer
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]

            # Rollout Worker
            rollout_batch_size (int): number of parallel rollouts per DDPG agent

            # NN Configuration
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            scope (str): the scope used for the TensorFlow graph
            reuse (boolean): whether or not the networks should be reused

            # Replay Buffer
            buffer_size (int): number of transitions that are stored in the replay buffer

            # HER
            sample_transitions (function) function that samples from the replay buffer

            # Dual Network Set
            polyak (float): coefficient for Polyak-averaging of the target network

            # Loss functions
            action_l2 (float): coefficient for L2 penalty on the actions

            # Training
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            gamma (float): gamma used for Q learning updates
            
            # Others
            bc_loss: whether or not the behavior cloning loss should be used as an auxilliary loss
            q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            num_demo: Number of episodes in to be used in the demonstration buffer
            demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            prm_loss_weight: Weight corresponding to the primary loss
            aux_loss_weight: Weight corresponding to the auxilliary loss also called the cloning loss
        """
        # Prepare parameters
        self.create_actor_critic = import_function(self.network_class)
        self.dimo = self.input_dims["o"]
        self.dimu = self.input_dims["u"]
        input_shapes = dims_to_shapes(self.input_dims)

        logger.info("Preparing staging area for feeding data to the model.")
        # -----------------------------------------------------------------------------
        # Note: self.stage_shapes == {"o":(None, *int), "o_2":(None, *int), "g":(None, *int), "g_2":(None, *int), "u":(None, *int), "r":(None)}
        # None is for batch size which is not determined
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith("info_"):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ["o"]:
            stage_shapes[key + "_2"] = stage_shapes[key]
        stage_shapes["r"] = (None,1)
        self.stage_shapes = stage_shapes  # feeding data into model
        logger.debug("DDPG.__init__ -> The staging shapes are: {}".format(self.stage_shapes))
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()], shapes=list(self.stage_shapes.values())
            )
            self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        # -----------------------------------------------------------------------------
        with tf.variable_scope(self.scope):
            self._create_network(reuse=reuse)

        logger.info("Configure the replay buffer.")
        # -----------------------------------------------------------------------------
        # Note: buffer_shapes: {'o': (T+1, 10), 'u': (T, 4), 'r': (T, 1)}
        buffer_shapes = {
            key: (self.T if key != "o" else self.T + 1, *input_shapes[key]) for key, val in input_shapes.items()
        }
        buffer_shapes["r"] = (self.T, 1)
        logger.debug("DDPG.__init__ -> The buffer shapes are: {}".format(buffer_shapes))

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        logger.debug("DDPG.__init__ -> The buffer size is: {}".format(buffer_size))
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def get_actions(self, o, noise_eps=0.0, random_eps=0.0, use_target_net=False, compute_Q=False):
        o = self._preprocess_og(o)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32),
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
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

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def logs(self, prefix=""):
        logs = []
        
        if prefix is not "" and not prefix.endswith("/"):
            return [(prefix + "/" + key, val) for key, val in logs]
        else:
            return logs

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)  # otherwise only sample from primary buffer

        o, o_2 = transitions["o"], transitions["o_2"]
        transitions["o"]= self._preprocess_og(o)
        transitions["o_2"]= self._preprocess_og(o_2)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
            self.current_batch = batch
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        logger.debug("DDPG.train -> critic_loss:{}, actor_loss:{}".format(critic_loss, actor_loss))
        return critic_loss, actor_loss

    def check_train(self):
        """
        For debugging only
        """
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, self.current_batch)))
        critic_loss, actor_loss, _, _ = self._grads()
        logger.debug("DDPG.check_train -> current_buffer_size:{}".format(self.get_current_buffer_size()))
        logger.debug("DDPG.check_train -> critic_loss:{}, actor_loss:{}".format(critic_loss, actor_loss))

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _create_network(self, reuse=False):
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # mini-batch sampling.
        # Note: batch_tf == {"o":(None, *int), "o_2":(None, *int), "u":(None, *int), "r":(None)}
        # None is for batch size which is not determined
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        # Now we do this in her.py when re-computing the reward
        # batch_tf["r"] = tf.reshape(batch_tf["r"], [-1, 1])

        # networks
        with tf.variable_scope("main") as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type="main", **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope("target") as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            # The input to the target network has to be the resultant observation and goal!
            target_batch_tf["o"] = batch_tf["o_2"]
            self.target = self.create_actor_critic(target_batch_tf, net_type="target", **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0.0 if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf["r"] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))

        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars("main/Q"))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars("main/pi"))
        assert len(self._vars("main/Q")) == len(Q_grads_tf)
        assert len(self._vars("main/pi")) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars("main/Q"))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars("main/pi"))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars("main/Q"))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars("main/pi"))

        # optimizers
        self.Q_adam = MpiAdam(self._vars("main/Q"), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars("main/pi"), scale_grad_by_procs=False)

        # polyak averaging
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

        # initialize all variables
        tf.variables_initializer(self._global_vars("")).run()
        self._sync_optimizers()
        self._init_target_net()

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + "/" + scope)
        return res

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run(
            [self.Q_loss_tf, self.main.Q_pi_tf, self.Q_grad_tf, self.pi_grad_tf]
        )
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def _preprocess_og(self, o):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        return o

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def _vars(self, scope):
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
            "buffer",
            "sess",
            "_stats",
            "main",
            "target",
            "lock",
            "env",
            "sample_transitions",
            "stage_shapes",
            "create_actor_critic",
        ]

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state["buffer_size"] = self.buffer_size
        state["tf"] = self.sess.run([x for x in self._global_vars("") if "buffer" not in x.name])
        return state

    def __setstate__(self, state):
        if "sample_transitions" not in state:
            # We don't need this for playing the policy.
            state["sample_transitions"] = None

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
