import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Actor(tf.keras.Model):
    def __init__(self, dimo, dimg, dimu, max_u, layer_sizes):
        super().__init__()
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u
        # build layers
        self.model_layers = []
        for size in layer_sizes:
            layer = tf.keras.layers.Dense(
                units=size,
                activation="relu",
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                bias_initializer=None,
            )
            self.model_layers.append(layer)
        self.model_layers.append(
            tf.keras.layers.Dense(
                units=self.dimu[0],
                activation="tanh",
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                bias_initializer=None,
            )
        )

    @tf.function
    def call(self, inputs):
        o, g = inputs
        res = tf.concat([o, g], axis=1)
        for l in self.model_layers:
            res = l(res)
        res = tf.clip_by_value(res, -1.0, 1.0) * self.max_u
        return res


class Critic(tf.keras.Model):
    def __init__(self, dimo, dimg, dimu, max_u, layer_sizes):
        super().__init__()
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u
        # build layers
        self.model_layers = []
        for size in layer_sizes:
            layer = tf.keras.layers.Dense(
                units=size, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal()
            )
            self.model_layers.append(layer)
        self.model_layers.append(
            tf.keras.layers.Dense(units=1, kernel_initializer=tf.keras.initializers.glorot_normal())
        )

    @tf.function
    def call(self, inputs):
        o, g, u = inputs
        res = tf.concat([o, g, u / self.max_u], axis=1)
        for l in self.model_layers:
            res = l(res)
        return res


def test_actor_critic():
    import numpy as np

    dimo = (2,)
    dimg = (2,)
    dimu = (2,)
    max_u = 2.0
    layer_sizes = [16]

    input_o = tf.zeros((1, *dimo))
    input_g = tf.zeros((1, *dimg))
    input_u = tf.zeros((1, *dimu))

    actor = Actor(dimo, dimg, dimu, max_u, layer_sizes)
    critic = Critic(dimo, dimg, dimu, max_u, layer_sizes)
    actor([input_o, input_g])
    critic([input_o, input_g, input_u])
    actor.summary()
    critic.summary()


if __name__ == "__main__":
    test_actor_critic()
