import tensorflow as tf
import tensorflow_probability as tfp

from td3fd.util.tf_util import MLP

tfd = tfp.distributions


class Actor(tf.Module):
    def __init__(self, dimo, dimg, dimu, max_u, noise, layer_sizes):
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
                units=self.dimu,
                activation="tanh",
                kernel_initializer=tf.keras.initializers.glorot_normal(),
                bias_initializer=None,
            )
        )
        self.noise = tfd.Normal(loc=0.0, scale=0.1 if noise else 0.0)

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
        res = res + self.noise.sample(tf.shape(res))
        res = tf.clip_by_value(res, -1.0, 1.0) * self.max_u
        return res


class Critic(tf.Module):
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


def test_actor_critic():
    import numpy as np

    dimo = 2
    dimg = 2
    dimu = 2
    max_u = 2.0
    noise = True
    layer_sizes = [16]
    o = tf.placeholder(tf.float32, shape=(None, dimo))
    g = tf.placeholder(tf.float32, shape=(None, dimg))

    actor = Actor(dimo, dimg, dimu, max_u, noise, layer_sizes)
    critic = Critic(dimo, dimg, dimu, max_u, layer_sizes)
    u = actor(o, g)
    q = critic(o, g, u)
    actor.summary()
    critic.summary()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    u, q = sess.run(
        [u, q], feed_dict={o: np.array([1.0, 1.0]).reshape(-1, dimo), g: np.array([1.0, 1.0]).reshape(-1, dimg)}
    )
    print(u, q)


if __name__ == "__main__":
    test_actor_critic()
