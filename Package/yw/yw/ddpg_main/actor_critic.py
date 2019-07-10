import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from yw.util.tf_util import nn


class ActorCritic:
    def __init__(
        self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, use_td3, add_pi_noise, hidden, layers, **kwargs
    ):
        """The actor-critic network and related training code.

        Args:
            inputs_tf    (dict of tensor) - all necessary inputs for the network: theobservation (o), the goal (g), and the action (u)
            dimo         (int)            - the dimension of the observations
            dimg         (int)            - the dimension of the goals
            dimu         (int)            - the dimension of the actions
            max_u        (float)          - the maximum magnitude of actions; action outputs will be scaled accordingly
            o_stats      (Normalizer)     - normalizer for observations
            g_stats      (Normalizer)     - normalizer for goals
            use_td3      (int)            - number of critic for each actor, refer to td3 algorithm
            hidden       (int)            - number of hidden units that should be used in hidden layers
            add_pi_noise (bool)
            layers       (int)            - number of hidden layers
        """

        # Prepare inputs for actor and critic.
        self.o_tf = inputs_tf["o"]
        state = o_stats.normalize(self.o_tf)
        # state = self.o_tf
        # for multigoal environments, we have goal as another states
        if dimg != 0:
            self.g_tf = inputs_tf["g"]
            goal = g_stats.normalize(self.g_tf)
            # goal = self.g_tf
            state = tf.concat(axis=1, values=[state, goal])
        self.u_tf = inputs_tf["u"]

        with tf.variable_scope("pi"):
            nn_pi_tf = tf.tanh(nn(state, [hidden] * layers + [dimu]))
            if use_td3 and add_pi_noise:  # for td3, add noise!
                nn_pi_tf += tfd.Normal(loc=[0.0] * dimu, scale=1.0).sample([tf.shape(self.o_tf)[0]])
                nn_pi_tf = tf.clip_by_value(nn_pi_tf, -1.0, 1.0)
            self.pi_tf = max_u * nn_pi_tf

        with tf.variable_scope("Q"):
            with tf.variable_scope("Q1"):
                # for policy training
                input_Q = tf.concat(axis=1, values=[state, self.pi_tf / max_u])
                self.Q_pi_tf = nn(input_Q, [hidden] * layers + [1])
                # for critic training
                input_Q = tf.concat(axis=1, values=[state, self.u_tf / max_u])
                self.Q_tf = nn(input_Q, [hidden] * layers + [1], reuse=True)

            if use_td3:
                with tf.variable_scope("Q2"):
                    # for policy training
                    input_Q = tf.concat(axis=1, values=[state, self.pi_tf / max_u])
                    self.Q2_pi_tf = nn(input_Q, [hidden] * layers + [1])
                    # for critic training
                    input_Q = tf.concat(axis=1, values=[state, self.u_tf / max_u])
                    self.Q2_tf = nn(input_Q, [hidden] * layers + [1], reuse=True)
