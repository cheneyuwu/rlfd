import tensorflow as tf

from yw.tool import logger
from yw.ddpg_tf2.normalizer import Normalizer
from yw.util.util import store_args
from yw.util.tf2_util import nn


class Actor(tf.keras.Model):
    def __init__(self, max_u, hidden, layers, dimu, **kwargs):
        super(Actor, self).__init__()
        self.nn = nn([hidden] * layers + [dimu])
        self.max_u = max_u

    def call(self, input):
        concat_input = tf.concat(axis=1, values=input)
        return self.max_u * tf.tanh(self.nn(concat_input))


class Critic(tf.keras.Model):
    def __init__(self, max_u, hidden, layers, **kwargs):
        super(Critic, self).__init__()
        self.max_u = max_u
        self.nn = nn([hidden] * layers + [1])

    def call(self, input):
        concat_input = tf.concat(axis=1, values=[input[0], input[1], input[2] / self.max_u])
        return self.nn(concat_input)


class ActorCritic:
    @store_args
    def __init__(self, max_u, hidden, layers, dimo, dimg, dimu, o_stats, g_stats, **kwargs):

        self.critic = Critic(**vars(self))
        self.critic.build([(None, dimo), (None, dimg), (None, dimu)])
        self.critic_num_weights = len(self.critic.get_weights())

        self.actor = Actor(**vars(self))
        self.actor.build([(None, dimo), (None, dimg)])
        self.actor_num_weights = len(self.actor.get_weights())

        self.variables = self.actor.variables + self.critic.variables

    @tf.function
    def get_pi(self, input):
        input = self._normalize_og(input)
        return self.actor([input["o"], input["g"]])

    @tf.function
    def get_q(self, input):
        input = self._normalize_og(input)
        return self.critic([input["o"], input["g"], input["u"]])

    @tf.function
    def get_q_pi(self, input):
        u = self.get_pi(input)
        input = self._normalize_og(input)
        return self.critic([input["o"], input["g"], u])

    def get_actor_vars(self):
        return self.actor.trainable_variables

    def get_critic_vars(self):
        return self.critic.trainable_variables

    @tf.function
    def set_variables(self, weights):
        for i in range(len(self.variables)):
            self.variables[i].assign(weights[i])

    def _normalize_og(self, input):
        result = input.copy()
        result["o"] = self.o_stats.normalize(input["o"])
        result["g"] = self.g_stats.normalize(input["g"])
        return result


class EnsambleActorCritic:
    @store_args
    def __init__(self, num_sample, max_u, hidden, layers, dimu, **kwargs):
        self.ensembles = [ActorCritic(**vars(self)) for _ in range(num_sample)]
