import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfl = tf.keras.layers


class Generator(tf.keras.Model):
    def __init__(
        self,
        input_tensor_spec=None,
        conv_layer_params=None,
        fc_layer_params=None,
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu,
        kernel_initializer=None,
        # batch_squash=True,
        dtype=tf.float32,
        name="Generator",
        conv_type="2d",  # "1d" or "2d"
    ):
        super(Generator, self).__init__()

        self.network_layers = []
        for size in fc_layer_params[:-1]:
            layer = tfl.Dense(units=size, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal())
            self.network_layers.append(layer)
        self.network_layers.append(
            tfl.Dense(units=fc_layer_params[-1], kernel_initializer=tf.keras.initializers.glorot_normal())
        )

    def call(self, inputs):
        res = inputs
        for l in self.network_layers:
            res = l(res)
        return res


class Discriminator(tf.keras.Model):
    def __init__(
        self,
        input_tensor_spec=None,
        conv_layer_params=None,
        fc_layer_params=None,
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu,
        kernel_initializer=None,
        # batch_squash=True,
        dtype=tf.float32,
        name="Discriminator",
        conv_type="2d",  # "1d" or "2d"
    ):
        super(Discriminator, self).__init__()

        self.network_layers = []
        for size in fc_layer_params[:-1]:
            layer = tfl.Dense(units=size, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal())
            self.network_layers.append(layer)
        self.network_layers.append(
            tfl.Dense(units=fc_layer_params[-1], kernel_initializer=tf.keras.initializers.glorot_normal())
        )

    def call(self, inputs, training=True):
        res = inputs
        for l in self.network_layers:
            res = l(res)
        return res


if __name__ == "__main__":
    pass
