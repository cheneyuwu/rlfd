import os

import numpy as np
import copy
import functools
import collections
import multiprocessing

import tensorflow as tf  # pylint: ignore-module
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def get_session(config=None):
    """
    Get default session or create one with a given config
    """
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess


def in_session(f):
    @functools.wraps(f)
    def newfunc(*args, **kwargs):
        with tf.Session():
            f(*args, **kwargs)

    return newfunc


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """
    Returns a session that will use <num_cpu> CPU's only
    """
    if num_cpu is None:
        num_cpu = int(os.getenv("RCALL_NUM_CPU", multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True, inter_op_parallelism_threads=num_cpu, intra_op_parallelism_threads=num_cpu
        )
        # Yuchen: re-enable when GPU allowed
        # config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)


def single_threaded_session():
    """
    Returns a session which will only use a single CPU
    """
    return make_session(num_cpu=1)


# Flat vectors
# =====================================


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), "shape function assumes that shape is fully known"
    return out


def numel(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(
        axis=0,
        values=[
            tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
            for (v, grad) in zip(var_list, grads)
        ],
    )


def flatten_grads(var_list, grads):
    """Flattens a variables and their gradients.
    """
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start : start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.get_default_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return tf.get_default_session().run(self.op)


# Shape adjustment for feeding into tf placeholders
# =============================================================================
def adjust_shape(placeholder, data):
    """
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown

    Parameters:
        placeholder     tensorflow input placeholder

        data            input data to be (potentially) reshaped to be fed into placeholder

    Returns:
        reshaped data
    """

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    return np.reshape(data, placeholder_shape)


# Build neural net works
# =============================================================================


def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(
            inputs=input,
            units=size,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            reuse=reuse,
            name=name + "_" + str(i),
        )
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """

    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = tf.layers.dense(
                inputs=h,
                units=num_hidden,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="mlp_fc_" + str(i),
            )
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError("Unknown network type: {}".format(name))


# Build normalizing flows
# =============================================================================

# quite easy to interpret - multiplying by alpha causes a contraction in volume.
class LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="leaky_relu"):
        super(LeakyReLU, self).__init__(
            forward_min_event_ndims=1, inverse_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1.0 / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        event_dims = (
            1
        )  # self._event_dims_tensor(y)  ## Note: this wil break for objects of size other than N x dim(vector)

        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=event_dims)


class NormalizingFlow:
    def __init__(self, base_dist, num_layers=6, d=2, r=2):
        self.bijectors = []

        for i in range(num_layers):
            with tf.variable_scope("bijector_%d" % i):
                V = tf.get_variable("V", [d, r], dtype=tf.float32)  # factor loading
                shift = tf.get_variable("shift", [d], dtype=tf.float32)  # affine shift
                L = tf.get_variable("L", [d * (d + 1) / 2], dtype=tf.float32)  # lower triangular

                self.bijectors.append(
                    tfb.Affine(scale_tril=tfd.fill_triangular(L), scale_perturb_factor=V, shift=shift)
                )

                alpha = tf.abs(tf.get_variable("alpha", [], dtype=tf.float32)) + 0.01
                self.bijectors.append(LeakyReLU(alpha=alpha))

        # Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
        # that they are applied..
        mlp_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])), name="2d_mlp_bijector")

        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=mlp_bijector)

    def __call__(self, input):
        return self.dist.log_prob(input)


# MAF
# =============================================================================

class MAF:
    def __init__(self, base_dist, dim=2, num_layers=6):
        self.bijectors = []

        for _ in range(num_layers):
            self.bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(hidden_layers=[512, 512])
                )
            )
            # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
            # self.bijectors.append(tfb.BatchNormalization())
            self.bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))

        # Discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])))

        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)

    def __call__(self, input):
        return self.dist.log_prob(input)


if __name__ == "__main__":

    import matplotlib.pylab as pl
    import matplotlib.gridspec as gridspec

    batch_size = 512

    def sample_target_halfmoon():
        x2_dist = tfd.Normal(loc=tf.cast(0.0, tf.float32), scale=tf.cast(4.0, tf.float32))
        x2_samples = x2_dist.sample(batch_size)
        x1 = tfd.Normal(loc=0.25 * tf.square(x2_samples), scale=tf.ones(batch_size, dtype=tf.float32))
        x1_samples = x1.sample()
        x_samples = tf.stack([x1_samples, x2_samples], axis=1)
        return x_samples  # shape (batch_size, 2)

    def sample_flow(base_dist, transformed_dist):
        x = base_dist.sample(512)
        samples = [x]
        names = [base_dist.name]
        for bijector in reversed(transformed_dist.bijector.bijectors):
            x = bijector.forward(x)
            samples.append(x)
            names.append(bijector.name)

        return x, samples, names

    def visualize_flow(gs, row, samples, titles):
        X0 = samples[0]

        for i, j in zip([0, len(samples) - 1], [0, 1]):  # range(len(samples)):
            X1 = samples[i]

            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            ax = pl.subplot(gs[row, j])
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="red")

            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="green")

            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="blue")

            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            pl.scatter(X1[idx, 0], X1[idx, 1], s=10, color="black")
            pl.xlim([-5, 30])
            pl.ylim([-10, 10])
            pl.title(titles[j])

    tf.set_random_seed(6)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    x_samples = sample_target_halfmoon()
    np_samples = sess.run(x_samples)

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype=tf.float32))
    nn = NormalizingFlow(base_dist)
    transformed_dist = nn.dist
    _, samples_no_training, _ = sample_flow(base_dist, transformed_dist)

    sess.run(tf.global_variables_initializer())
    samples_no_training = sess.run(samples_no_training)

    pl.figure()
    gs = gridspec.GridSpec(3, 3)

    # Training dataset
    ax = pl.subplot(gs[0, 0])
    pl.scatter(np_samples[:, 0], np_samples[:, 1], s=10, color="red")
    pl.xlim([-5, 30])
    pl.ylim([-10, 10])
    pl.title("Training samples")

    # Flow before training
    visualize_flow(gs, 1, samples_no_training, ["Base dist", "Samples w/o training"])

    loss = -tf.reduce_mean(nn(x_samples))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    sess.run(tf.global_variables_initializer())

    NUM_STEPS = 50000
    global_step = []
    np_losses = []
    for i in range(NUM_STEPS):
        _, np_loss = sess.run([train_op, loss])
        if i % 1000 == 0:
            global_step.append(i)
            np_losses.append(np_loss)

        if i % int(1e4) == 0:
            print(i, np_loss)

    # start = 10
    # pl.plot(np_losses[start:])

    _, samples_with_training, _ = sample_flow(base_dist, transformed_dist)
    samples_with_training = sess.run(samples_with_training)

    # Flow after training
    visualize_flow(gs, 2, samples_with_training, ["Base dist", "Samples w/ training"])

    pl.show()
