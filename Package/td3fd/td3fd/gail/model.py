import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ValueNet(tf.Module):
    def __init__(self, dimo, dimg, layer_sizes):
        super().__init__()
        self.dimo = dimo
        self.dimg = dimg
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

    def __call__(self, o, g):
        state = o
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            state = tf.concat([state, g], axis=1)
        else:
            assert g is None
        res = state
        for l in self.model_layers:
            res = l(res)
        return res


class Generator(tf.Module):
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
            tf.keras.layers.Dense(
                units=self.dimu, activation="tanh", kernel_initializer=tf.keras.initializers.glorot_normal()
            )
        )
        self.logstd_tf = tf.Variable(tf.zeros(self.dimu), dtype=tf.float32)

    def __call__(self, o, g):
        dist = self.get_output_dist(o, g)
        res = dist.sample()
        res = tf.clip_by_value(res, -1.0, 1.0) * self.max_u
        return res

    def get_output_dist(self, o, g):
        state = o
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            state = tf.concat([state, g], axis=1)
        else:
            assert g is None
        res = state
        for l in self.model_layers:
            res = l(res)
        res = tfd.Normal(loc=res, scale=tf.exp(self.logstd_tf))
        return res


class Discriminator(tf.Module):
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

    def __call__(self, o, g, u):
        state = o
        # for multigoal environments, we have goal as another states
        if self.dimg != 0:
            state = tf.concat([state, g], axis=1)
        else:
            assert g is None
        res = tf.concat([state, u / self.max_u], axis=1)
        for l in self.model_layers:
            res = l(res)
        return res
