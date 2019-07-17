import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from yw.util.tf_util import MLP


class ActorCritic:
    def __init__(self, dimo, dimg, dimu, max_u, o_stats, g_stats, add_pi_noise, layer_sizes, **kwargs):
        """The actor-critic network and related training code.

        Args:
            dimo         (int)            - the dimension of the observations
            dimg         (int)            - the dimension of the goals
            dimu         (int)            - the dimension of the actions
            max_u        (float)          - the maximum magnitude of actions; action outputs will be scaled accordingly
            o_stats      (Normalizer)     - normalizer for observations
            g_stats      (Normalizer)     - normalizer for goals
            layer_sizes  (list of ints)   - number of hidden units in each layer
            add_pi_noise (bool)
        """

        # Params
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u
        self.o_stats = o_stats
        self.g_stats = g_stats
        self.add_pi_noise = add_pi_noise

        # actor
        with tf.variable_scope("pi"):
            input_shape = (None, self.dimo + self.dimg)
            self.pi_nn = MLP(input_shape=input_shape, layers_sizes=layer_sizes + [self.dimu])

        # critic
        with tf.variable_scope("Q"):
            input_shape = (None, self.dimo + self.dimg + self.dimu)
            with tf.variable_scope("Q1"):
                self.q1_nn = MLP(input_shape=input_shape, layers_sizes=layer_sizes + [1])
            with tf.variable_scope("Q2"):
                self.q2_nn = MLP(input_shape=input_shape, layers_sizes=layer_sizes + [1])

    def actor(self, o, g):
        state = self._normalize_concat_state(o, g)
        nn_pi_tf = tf.tanh(self.pi_nn(state))
        if self.add_pi_noise:  # for td3, add noise!
            nn_pi_tf += tfd.Normal(loc=[0.0] * self.dimu, scale=1.0).sample([tf.shape(o)[0]])
            nn_pi_tf = tf.clip_by_value(nn_pi_tf, -1.0, 1.0)
        pi_tf = self.max_u * nn_pi_tf
        return pi_tf

    def critic1(self, o, g, u):
        state = self._normalize_concat_state(o, g)
        input_q = tf.concat(axis=1, values=[state, u / self.max_u])
        q_tf = self.q1_nn(input_q)

        return q_tf

    def critic2(self, o, g, u):
        state = self._normalize_concat_state(o, g)
        input_q = tf.concat(axis=1, values=[state, u / self.max_u])
        q_tf = self.q2_nn(input_q)

        return q_tf

    def _normalize_concat_state(self, o, g):
        state = self.o_stats.normalize(o)
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            goal = self.g_stats.normalize(g)
            state = tf.concat(axis=1, values=[state, goal])
        return state
