import threading

import numpy as np
from mpi4py import MPI
import tensorflow as tf


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf, sess=None):
        """A normalizer that ensures that observations are approximately distributed according to
        a standard Normal distribution (i.e. have mean zero and variance one).

        Args:
            size               (int)    - the size of the observation to be normalized
            eps                (float)  - a small constant that avoids underflows
            default_clip_range (float)  - normalized observations are clipped to be in [-default_clip_range, default_clip_range]
            sess               (object) - the TensorFlow session to be used
        """
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        self.sum_tf = tf.Variable(
            initial_value=np.zeros_like(self.local_sum), name="sum", trainable=False, dtype=tf.float32
        )
        self.sumsq_tf = tf.Variable(
            initial_value=np.zeros_like(self.local_sumsq), name="sumsq", trainable=False, dtype=tf.float32
        )
        self.count_tf = tf.Variable(
            initial_value=np.ones_like(self.local_count), name="count", trainable=False, dtype=tf.float32
        )
        self.mean = tf.Variable(initial_value=np.ones((self.size,)), name="mean", trainable=False, dtype=tf.float32)
        self.std = tf.Variable(initial_value=np.ones((self.size,)), name="std", trainable=False, dtype=tf.float32)

        def update_op(synced_count, synced_sum, synced_sumsq):
            self.count_tf.assign_add(synced_count)
            self.sum_tf.assign_add(synced_sum)
            self.sumsq_tf.assign_add(synced_sumsq)

        self.update_op = update_op

        def recompute_op():
            self.mean.assign(self.sum_tf / self.count_tf)
            self.std.assign(
                tf.sqrt(
                    tf.maximum(
                        tf.square(self.eps), self.sumsq_tf / self.count_tf - tf.square(self.sum_tf / self.count_tf)
                    )
                )
            )

        self.recompute_op = recompute_op

        self.lock = threading.Lock()

    def update(self, v):
        v = v.reshape(-1, self.size)

        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = Normalizer.reshape_for_broadcasting(self.mean, v)
        std = Normalizer.reshape_for_broadcasting(self.std, v)
        return tf.clip_by_value((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        mean = Normalizer.reshape_for_broadcasting(self.mean, v)
        std = Normalizer.reshape_for_broadcasting(self.std, v)
        return mean + v * std

    def synchronize(self, local_sum, local_sumsq, local_count, root=None):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            # Copy over results.
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()

            # Reset.
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0

        # We perform the synchronization outside of the lock to keep the critical section as short
        # as possible.
        synced_sum, synced_sumsq, synced_count = self.synchronize(
            local_sum=local_sum, local_sumsq=local_sumsq, local_count=local_count
        )

        self.update_op(synced_count, synced_sum, synced_sumsq)
        self.recompute_op()

    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    @staticmethod
    def reshape_for_broadcasting(source, target):
        """Reshapes a tensor (source) to have the correct shape and dtype of the target
        before broadcasting it with MPI.
        """
        dim = len(target.get_shape())
        shape = ([1] * (dim - 1)) + [-1]
        return tf.reshape(tf.cast(source, target.dtype), shape)