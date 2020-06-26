import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfl = tf.keras.layers


class Generator(tf.keras.Model):
    def __init__(
        self, layer_sizes,
    ):
        super(Generator, self).__init__()

        self.network_layers = []
        for size in layer_sizes[:-1]:
            layer = tfl.Dense(units=size, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal())
            self.network_layers.append(layer)
        self.network_layers.append(
            tfl.Dense(units=layer_sizes[-1], kernel_initializer=tf.keras.initializers.glorot_normal())
        )

    def call(self, inputs):
        res = inputs
        for l in self.network_layers:
            res = l(res)
        return res


class Discriminator(tf.keras.Model):
    def __init__(
        self, layer_sizes,
    ):
        super(Discriminator, self).__init__()

        self.network_layers = []
        for size in layer_sizes[:-1]:
            layer = tfl.Dense(units=size, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal())
            self.network_layers.append(layer)
        self.network_layers.append(
            tfl.Dense(units=layer_sizes[-1], kernel_initializer=tf.keras.initializers.glorot_normal())
        )

    def call(self, inputs):
        res = inputs
        for l in self.network_layers:
            res = l(res)
        return res
