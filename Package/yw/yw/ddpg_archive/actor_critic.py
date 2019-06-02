import tensorflow as tf

from yw.tool import logger
from yw.util.tf_util import get_network_builder, nn


class Model(object):
    def __init__(self, name, network="mlp", **network_kwargs):
        self.name = name
        self.network_builder = get_network_builder(network)(**network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if "LayerNorm" not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name="actor", network="mlp", **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.network_builder(obs)
            x = tf.layers.dense(
                x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
            )
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name="critic", network="mlp", **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = tf.concat([obs, action], axis=-1)  # this assumes observation and action can be concatenated
            x = self.network_builder(x)
            x = tf.layers.dense(
                x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name="output"
            )
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if "output" in var.name]
        return output_vars


class ActorCritic:
    def __init__(
        self,
        input_o,
        input_u,
        num_sample,
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
            num_sample (int)            - number of ensemble actor critic pairs
            ca_ratio   (int)            - number of critic for each actor, refer to td3 algorithm
            hidden     (int)            - number of hidden units that should be used in hidden layers
            layers     (int)            - number of hidden layers
            net_type   (str)
        """
        with tf.variable_scope("pi"):
            self.pi_tf = []  # output of actor
            for i in range(num_sample):
                with tf.variable_scope("pi" + str(i)):
                    self.pi_tf.append(tf.tanh(nn(input_o, [hidden] * layers + [input_u.shape[1]])))
        with tf.variable_scope("Q"):
            self.Q_pi_tf = []
            self.Q_tf = []
            for i in range(num_sample):
                with tf.variable_scope("Q" + str(i)):
                    # for policy training
                    input_Q = tf.concat(axis=1, values=[input_o, self.pi_tf[i]])
                    self.Q_pi_tf.append(nn(input_Q, [hidden] * layers + [1]))
                    # for critic training
                    input_Q = tf.concat(axis=1, values=[input_o, input_u])
                    self.Q_tf.append(nn(input_Q, [hidden] * layers + [1], reuse=True))
            self.Q_sample_tf = tf.concat(values=self.Q_tf, axis=1)
            self.Q_mean_tf, self.Q_var_tf = tf.nn.moments(self.Q_sample_tf, 1)
            self.Q_pi_sample_tf = tf.concat(values=self.Q_pi_tf, axis=1)
            self.Q_pi_mean_tf, self.Q_pi_var_tf = tf.nn.moments(self.Q_pi_sample_tf, 1)


