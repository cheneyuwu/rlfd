import tensorflow as tf
import numpy as np
from yw.util.util import store_args
from yw.util.tf2_util import get_flat, set_from_flat

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class MpiAdam:
    @store_args
    def __init__(self, var_list, *, learning_rate=0.01, scale_grad_by_procs=True, comm=None):
        self.comm = MPI.COMM_WORLD if comm is None and MPI is not None else comm
        self.adam = tf.optimizers.Adam(learning_rate=learning_rate)
        self.sync_grad = [tf.Variable(tf.zeros_like(i), trainable=False, name="mpi_grad") for i in var_list]
        self.global_step = 0

    def update(self, loss, input={}):
        # check
        self.global_step += 1
        if self.global_step % 100 == 0:
            self.check_synced()
        # get gradient
        with tf.GradientTape() as g:
            l = loss(**input)
        local_grad = g.gradient(l, self.var_list)
        # sync
        if self.comm is not None:
            local_grad_flat = get_flat(local_grad)
            global_grad_flat = np.zeros_like(local_grad_flat)
            self.comm.Allreduce(local_grad_flat, global_grad_flat, op=MPI.SUM)
            if self.scale_grad_by_procs:
                global_grad_flat /= self.comm.Get_size()
            set_from_flat(self.sync_grad, global_grad_flat)
            grad = self.sync_grad
        # apply gradient
        self.adam.apply_gradients(zip(grad, self.var_list))

    def sync(self):
        if self.comm is None:
            return
        theta = get_flat(self.var_list)
        self.comm.Bcast(theta, root=0)
        set_from_flat(self.var_list, theta)

    def check_synced(self):
        if self.comm is None:
            return
        if self.comm.Get_rank() == 0:  # this is root
            theta = get_flat(self.var_list)
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = get_flat(self.var_list)
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

def test_MpiAdam():
    np.random.seed(0)
    tf.random.set_seed(0)
    learning_rate = 1e-2

    a_init = np.random.randn(3).astype("float32")
    b_init = np.random.randn(2, 5).astype("float32")

    a = tf.Variable(np.zeros_like(a_init))
    b = tf.Variable(np.zeros_like(b_init))
    loss = lambda: tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    a.assign(a_init)
    b.assign(b_init)
    adam = tf.optimizers.Adam(learning_rate)
    losslist_ref = []
    for i in range(100):
        adam.minimize(loss, [a, b])
        l = loss()
        losslist_ref.append(l)

    a.assign(a_init)
    b.assign(b_init)
    adam = MpiAdam([a,b], learning_rate=learning_rate)
    adam.sync()
    losslist_test = []
    for i in range(100):
        adam.update(loss)
        adam.check_synced()
        l = loss()
        losslist_test.append(l)

    np.testing.assert_allclose(np.array(losslist_ref), np.array(losslist_test), atol=1e-4)


if __name__ == "__main__":
    test_MpiAdam()
