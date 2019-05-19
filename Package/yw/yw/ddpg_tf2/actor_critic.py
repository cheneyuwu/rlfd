import tensorflow as tf

from yw.tool import logger
from yw.util.util import store_args
from yw.util.tf2_util import nn


class Actor(tf.keras.Model):
    def __init__(self, max_u, hidden, layers, dimu, **kwargs):
        super(Actor, self).__init__()
        self.nn = nn([hidden] * layers + [dimu])
        self.max_u = max_u

    def call(self, input):
        return self.max_u * tf.tanh(self.nn(input))


class Critic(tf.keras.Model):
    def __init__(self, hidden, layers, **kwargs):
        super(Critic, self).__init__()
        self.nn = nn([hidden] * layers + [1])

    def call(self, input):
        return self.nn(input)


class ActorCritic:
    @store_args
    def __init__(self, max_u, hidden, layers, dimo, dimg, dimu, **kwargs):

        self.critic = Critic(**vars(self))
        critic_input_shape = dimo + dimu + dimg
        self.critic.build(tuple([None, dimo + dimu + dimg]))
        self.critic_num_weights = len(self.critic.get_weights())

        self.actor = Actor(**vars(self))
        actor_input_shape = dimo + dimg
        self.actor.build(tuple([None, dimo + dimg]))
        self.actor_num_weights = len(self.actor.get_weights())

    def get_pi(self, input):
        concat_input = tf.concat(axis=1, values=[input["o"], input["g"]])
        return self.actor(concat_input)

    def get_q(self, input):
        concat_input = tf.concat(axis=1, values=[input["o"], input["g"], input["u"] / self.max_u])
        return self.critic(concat_input)

    def get_q_pi(self, input):
        u = self.get_pi(input)
        concat_input = tf.concat(axis=1, values=[input["o"], input["g"], u])
        return self.critic(concat_input)

    def get_actor_vars(self):
        return self.actor.trainable_variables

    def get_critic_vars(self):
        return self.critic.trainable_variables

    def get_weights(self):
        weights = []
        weights += self.actor.get_weights()
        weights += self.critic.get_weights()
        return weights

    def set_weights(self, weights):
        size = 0
        self.actor.set_weights(weights[size : size + self.actor_num_weights])
        size += self.actor_num_weights
        self.critic.set_weights(weights[size : size + self.critic_num_weights])


class EnsambleActorCritic:
    @store_args
    def __init__(self, num_sample, max_u, hidden, layers, dimu, **kwargs):
        self.ensembles = [ActorCritic(**vars(self)) for _ in range(num_sample)]
