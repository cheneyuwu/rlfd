"""adopted from openai baseline code base
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Normalizer(tf.Module):
    def __init__(self, shape, eps=1e-2, clip_range=np.inf):
        """
        A normalizer that ensures that observations are approximately distributed according to a standard Normal
        distribution (i.e. have mean zero and variance one).

        Args:
            shape      (tuple)  - the size of the observation to be normalized
            eps        (float)  - a small constant that avoids underflows
            clip_range (float)  - normalized observations are clipped to be in [-clip_range, clip_range]
        """
        super().__init__()

        self.shape = shape
        self.clip_range = clip_range
        self.dtype = tf.float32

        self.eps = tf.constant(eps, dtype=self.dtype)
        self.sum_tf = tf.zeros(self.shape, dtype=self.dtype)
        self.sumsq_tf = tf.zeros(self.shape, dtype=self.dtype)
        self.count_tf = tf.zeros(1, dtype=self.dtype)

        self.mean_tf = tf.zeros(self.shape, dtype=self.dtype)
        self.std_tf = tf.ones(self.shape, dtype=self.dtype)

    def update(self, v):
        v = tf.reshape(v, [-1] + list(self.shape))
        self.count_tf += v.shape[0]
        self.sum_tf += tf.reduce_sum(v, axis=0)
        self.sumsq_tf += tf.reduce_sum(tf.square(v), axis=0)

        self.mean_tf = self.sum_tf / self.count_tf
        self.std_tf = tf.sqrt(
            tf.maximum(tf.square(self.eps), self.sumsq_tf / self.count_tf - tf.square(self.sum_tf / self.count_tf))
        )

    @tf.function
    def normalize(self, v):
        print("asdklfjas;dlfkjasdf")
        mean_tf, std_tf = self._reshape_for_broadcasting(v)
        return tf.clip_by_value((v - mean_tf) / std_tf, -self.clip_range, self.clip_range)

    @tf.function
    def denormalize(self, v):
        mean_tf, std_tf = self._reshape_for_broadcasting(v)
        return mean_tf + v * std_tf

    def _reshape_for_broadcasting(self, v):
        dim = len(v.shape) - len(self.shape)
        mean_tf = tf.reshape(self.mean_tf, [1] * dim + list(self.shape))
        std_tf = tf.reshape(self.std_tf, [1] * dim + list(self.shape))
        return mean_tf, std_tf


def test_normalizer():
    normalizer = Normalizer((2, 2))
    dist = tfp.distributions.Normal(loc=[[1.0, 2.0], [3.0, 4.0]], scale=2.0)
    train_data = dist.sample([1000000])
    test_data = dist.sample([1000000])
    normalizer.update(train_data)
    output = normalizer.normalize(test_data)
    revert_output = normalizer.denormalize(output)
    print(np.round(np.mean(output, axis=0)), "\n", np.round(np.std(output, axis=0)), "\n")
    print(np.round(np.mean(revert_output, axis=0)), "\n", np.round(np.std(revert_output, axis=0)), "\n")


if __name__ == "__main__":
    import time

    t = time.time()
    test_normalizer()
    t = time.time() - t
    print("Time: ", t)
