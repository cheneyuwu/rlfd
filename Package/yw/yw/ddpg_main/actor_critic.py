import tensorflow as tf

from yw.tool import logger
from yw.util.util import store_args
from yw.util.tf_util import nn


class ActorCritic:
    @store_args
    def __init__(
        self,
        inputs_tf,
        dimo,
        dimg,
        dimu,
        max_u,
        o_stats,
        g_stats,
        num_sample,
        ca_ratio,
        hidden,
        layers,
        **kwargs
    ):
        """The actor-critic network and related training code.

        Args:
            inputs_tf  (dict of tensor) - all necessary inputs for the network: theobservation (o), the goal (g), and the action (u)
            dimo       (int)            - the dimension of the observations
            dimg       (int)            - the dimension of the goals
            dimu       (int)            - the dimension of the actions
            max_u      (float)          - the maximum magnitude of actions; action outputs will be scaled accordingly
            o_stats    (Normalizer)     - normalizer for observations
            g_stats    (Normalizer)     - normalizer for goals
            num_sample (int)            - number of ensemble actor critic pairs
            ca_ratio   (int)            - number of critic for each actor, refer to td3 algorithm
            hidden     (int)            - number of hidden units that should be used in hidden layers
            layers     (int)            - number of hidden layers
            net_type   (str)
        """
        # Prepare inputs for actor and critic.
        self.o_tf = inputs_tf["o"]
        state = self.o_stats.normalize(self.o_tf) # o = self.o_tf
        self.u_tf = inputs_tf["u"]
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            self.g_tf = inputs_tf["g"]
            goal = self.g_stats.normalize(self.g_tf) # g = self.g_tf
            state = tf.concat(axis=1, values=[state, goal])

        self.pi_tf = []  # output of actor
        with tf.variable_scope("pi"):
            for i in range(self.num_sample):
                with tf.variable_scope("pi" + str(i)):
                    self.pi_tf.append(self.max_u * tf.tanh(nn(state, [self.hidden] * self.layers + [self.dimu])))
        with tf.variable_scope("Q"):
            self._input_Q = []
            self.Q_pi_tf = []
            self.Q_tf = []
            for i in range(self.num_sample):
                with tf.variable_scope("Q" + str(i)):
                    # for policy training
                    input_Q = tf.concat(axis=1, values=[state, self.pi_tf[i] / self.max_u])
                    self.Q_pi_tf.append(nn(input_Q, [self.hidden] * self.layers + [1]))
                    # for critic training
                    input_Q = tf.concat(axis=1, values=[state, self.u_tf / self.max_u])
                    self._input_Q.append(input_Q)  # exposed for tests
                    self.Q_tf.append(nn(input_Q, [self.hidden] * self.layers + [1], reuse=True))
            self.Q_sample_tf = tf.concat(values=self.Q_tf, axis=1)
            self.Q_mean_tf, self.Q_var_tf = tf.nn.moments(self.Q_sample_tf, 1)
            self.Q_pi_sample_tf = tf.concat(values=self.Q_pi_tf, axis=1)
            self.Q_pi_mean_tf, self.Q_pi_var_tf = tf.nn.moments(self.Q_pi_sample_tf, 1)

