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
        # TODO: for pixel input
        # self.network_layers = [
        #     tfl.Dense(units=8 * 8 * 8 * 64, use_bias=False),
        #     tfl.BatchNormalization(),
        #     tfl.LeakyReLU(),
        #     tfl.Reshape(target_shape=(8, 8, 512)),
        #     tfl.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),
        #     tfl.BatchNormalization(),
        #     tfl.LeakyReLU(),
        #     tfl.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False),
        #     tfl.BatchNormalization(),
        #     tfl.LeakyReLU(),
        #     tfl.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", use_bias=False),
        #     tfl.BatchNormalization(),
        #     tfl.LeakyReLU(),
        #     tfl.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"),
        # ]
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
        # TODO: for images
        # self.network_layers = [
        #     tfl.Conv2D(32, (5, 5), strides=(2, 2), padding="same"),
        #     tfl.LeakyReLU(),
        #     tfl.Dropout(0.3),
        #     tfl.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
        #     tfl.LeakyReLU(),
        #     tfl.Dropout(0.3),
        #     tfl.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        #     tfl.LeakyReLU(),
        #     tfl.Dropout(0.3),
        #     tfl.Flatten(),
        #     tfl.Dense(1),
        # ]
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
