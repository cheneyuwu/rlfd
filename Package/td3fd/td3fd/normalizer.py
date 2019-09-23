"""adopted from openai baseline code base
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Normalizer(tf.Module):
    def __init__(self, size, eps=1e-2, clip_range=np.inf, sess=None):
        """
        A normalizer that ensures that observations are approximately distributed according to a standard Normal
        distribution (i.e. have mean zero and variance one).

        Args:
            size       (int)    - the size of the observation to be normalized
            eps        (float)  - a small constant that avoids underflows
            clip_range (float)  - normalized observations are clipped to be in [-clip_range, clip_range]
        """
        super().__init__()

        self.size = size
        self.clip_range = clip_range
        self.dtype = tf.float32
        self.sess = sess if sess is not None else tf.get_default_session()

        self.eps = tf.constant(eps, dtype=self.dtype)
        self.sum_tf = tf.Variable(tf.zeros(self.size), dtype=self.dtype)
        self.sumsq_tf = tf.Variable(tf.zeros(self.size), dtype=self.dtype)
        self.count_tf = tf.Variable(0.0, dtype=self.dtype)

        self.mean_tf = tf.Variable(tf.zeros(self.size), dtype=self.dtype)
        self.std_tf = tf.Variable(tf.ones(self.size), dtype=self.dtype)

        # update
        self.input = tf.placeholder(self.dtype, shape=(None, self.size))
        self.update_op = tf.group(
            self.count_tf.assign_add(tf.cast(tf.shape(self.input)[0], tf.float32)),
            self.sum_tf.assign_add(tf.reduce_sum(self.input, axis=0)),
            self.sumsq_tf.assign_add(tf.reduce_sum(tf.square(self.input), axis=0)),
        )
        self.recompute_op = tf.group(
            tf.assign(self.mean_tf, self.sum_tf / self.count_tf),
            tf.assign(
                self.std_tf,
                tf.sqrt(
                    tf.maximum(
                        tf.square(self.eps), self.sumsq_tf / self.count_tf - tf.square(self.sum_tf / self.count_tf)
                    )
                ),
            ),
        )

    def update(self, v):
        self.sess.run(self.update_op, feed_dict={self.input: v.reshape(-1, self.size)})
        self.sess.run(self.recompute_op)

    def normalize(self, v):
        mean_tf, std_tf = self._reshape_for_broadcasting(v)
        return tf.clip_by_value((v - mean_tf) / std_tf, -self.clip_range, self.clip_range)

    def denormalize(self, v):
        mean_tf, std_tf = self._reshape_for_broadcasting(v)
        return mean_tf + v * std_tf

    def _reshape_for_broadcasting(self, v):
        dim = len(v.shape) - 1
        mean_tf = tf.reshape(self.mean_tf, [1] * dim + [self.size])
        std_tf = tf.reshape(self.std_tf, [1] * dim + [self.size])
        return mean_tf, std_tf


def test_normalizer():
    sess = tf.Session()
    normalizer = Normalizer(1, sess=sess)
    dist_tf = tfp.distributions.Normal(loc=1.0, scale=2.0)
    train_data_tf = dist_tf.sample([1000000])
    test_data_tf = dist_tf.sample([1000000])

    train_data = sess.run(train_data_tf)
    test_data = sess.run(test_data_tf)
    normalized_tf = normalizer.normalize(test_data_tf)
    denormalized_tf = normalizer.denormalize(normalized_tf)

    sess.run(tf.global_variables_initializer())
    normalizer.update(train_data)

    mean, std = sess.run([normalizer.mean_tf, normalizer.std_tf])
    output, revert_output = sess.run([normalized_tf, denormalized_tf])
    
    print(mean, std)
    print(np.mean(test_data), np.std(test_data))
    print(np.mean(output), np.std(output))
    print(np.mean(revert_output), np.std(revert_output))


if __name__ == "__main__":
    import time

    t = time.time()
    test_normalizer()
    t = time.time() - t
    print("Time: ", t)
