import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from yw.util.tf_util import nn


class ActorCritic:
    def __init__(self, dimo, dimg, dimu, max_u, o_stats, g_stats, add_pi_noise, hidden, layers, **kwargs):
        """The actor-critic network and related training code.

        Args:
            dimo         (int)            - the dimension of the observations
            dimg         (int)            - the dimension of the goals
            dimu         (int)            - the dimension of the actions
            max_u        (float)          - the maximum magnitude of actions; action outputs will be scaled accordingly
            o_stats      (Normalizer)     - normalizer for observations
            g_stats      (Normalizer)     - normalizer for goals
            hidden       (int)            - number of hidden units that should be used in hidden layers
            add_pi_noise (bool)
            layers       (int)            - number of hidden layers
        """

        # Params
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u
        self.o_stats = o_stats
        self.g_stats = g_stats
        self.add_pi_noise = add_pi_noise
        self.hidden = hidden
        self.layers = layers

    def actor(self, o, g):

        state = self._normalize_concat_state(o, g)

        with tf.variable_scope("pi"):
            nn_pi_tf = tf.tanh(nn(state, [self.hidden] * self.layers + [self.dimu]))
            if self.add_pi_noise:  # for td3, add noise!
                nn_pi_tf += tfd.Normal(loc=[0.0] * self.dimu, scale=1.0).sample([tf.shape(o)[0]])
                nn_pi_tf = tf.clip_by_value(nn_pi_tf, -1.0, 1.0)
            pi_tf = self.max_u * nn_pi_tf

        return pi_tf

    def critic1(self, o, g, u):

        state = self._normalize_concat_state(o, g)

        with tf.variable_scope("Q"):
            with tf.variable_scope("Q1"):
                input_q = tf.concat(axis=1, values=[state, u / self.max_u])
                q_tf = nn(input_q, [self.hidden] * self.layers + [1])

        return q_tf

    def critic2(self, o, g, u):

        state = self._normalize_concat_state(o, g)

        with tf.variable_scope("Q"):
            with tf.variable_scope("Q2"):
                input_q = tf.concat(axis=1, values=[state, u / self.max_u])
                q_tf = nn(input_q, [self.hidden] * self.layers + [1])

        return q_tf

    def _normalize_concat_state(self, o, g):
        state = self.o_stats.normalize(o)
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            goal = self.g_stats.normalize(g)
            state = tf.concat(axis=1, values=[state, goal])
        return state

