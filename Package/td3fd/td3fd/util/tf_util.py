"""Mostly adopted from OpenAI baselines: https://github.com/openai/baselines
"""
import collections
import copy
import functools
import multiprocessing
import os

import numpy as np
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
        config.gpu_options.allow_growth = True

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


# NNs
# =============================================================================
class MLP:
    def __init__(self, input_shape, layers_sizes, initializer_type="glorot", name=""):
        # choose initializer
        if initializer_type == "zero":
            kernel_initializer = tf.initializers.zeros()
            bias_initializer = tf.initializers.constant(0.01)
        elif initializer_type == "glorot":
            kernel_initializer = tf.initializers.glorot_normal()
            # kernel_initializer = tf.initializers.glorot_uniform()
            bias_initializer = None
        else:
            assert False, "unsupported initializer type"
        # build layers
        self.layers = []
        for i, size in enumerate(layers_sizes):
            activation = "relu" if i < len(layers_sizes) - 1 else None
            layer = tf.layers.Dense(
                units=size,
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=name + "_" + str(i),
            )
            layer.build(input_shape=input_shape)
            input_shape = layer.compute_output_shape(input_shape)
            self.layers.append(layer)

    def __call__(self, input):
        res = input
        for l in self.layers:
            res = l(res)
        return res


def nn(input, layers_sizes, reuse=None, flatten=False, initializer_type="glorot", name=""):
    """Creates a simple neural network
    """
    # choose initializer
    if initializer_type == "zero":
        kernel_initializer = tf.initializers.zeros()
        bias_initializer = tf.initializers.constant(0.01)
    elif initializer_type == "glorot":
        kernel_initializer = tf.initializers.glorot_normal()
        # kernel_initializer = tf.initializers.glorot_uniform()
        bias_initializer = None
    else:
        assert False, "unsupported initializer type"
    # connect layers
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(
            inputs=input,
            units=size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            reuse=reuse,
            name=name + "_" + str(i),
        )
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


"""Adopted from Eric Jang's normalizing flow tutorial: https://blog.evjang.com/2018/01/nf1.html
"""
# Simple Normalizing Flow
# =============================================================================
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


class ToyNF:
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


# Modern Normalizing Flows
# =============================================================================
class MAF:
    def __init__(self, base_dist, dim, num_bijectors=6, layer_sizes=[512, 512], initializer_type="glorot"):
        # choose initializer
        if initializer_type == "zero":
            kernel_initializer = tf.initializers.zeros()
            bias_initializer = tf.initializers.constant(0.01)
        elif initializer_type == "glorot":
            kernel_initializer = tf.initializers.glorot_normal()
            bias_initializer = None
        else:
            assert False, "unsupported initializer type"
        # build layers
        self.bijectors = []
        for _ in range(num_bijectors):
            self.bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=layer_sizes,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                    )
                )
            )
            self.bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))
        # discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])))
        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
        # output
        self.log_prob = self.dist.log_prob
        self.prob = self.dist.prob


class RealNVP:
    def __init__(self, base_dist, dim, num_masked, num_bijectors=6, layer_sizes=[512, 512], initializer_type="glorot"):
        # choose initializer
        if initializer_type == "zero":
            kernel_initializer = tf.initializers.zeros()
            bias_initializer = tf.initializers.constant(0.01)
        elif initializer_type == "glorot":
            kernel_initializer = tf.initializers.glorot_normal()
            bias_initializer = None
        else:
            assert False, "unsupported initializer type"
        # build layers
        self.bijectors = []
        for _ in range(num_bijectors):
            self.bijectors.append(
                tfb.RealNVP(
                    num_masked=num_masked,
                    shift_and_log_scale_fn=tfb.real_nvp_default_template(
                        hidden_layers=layer_sizes,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                    ),
                )
            )
            self.bijectors.append(tfb.Permute(permutation=list(range(0, dim))[::-1]))
        # discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(self.bijectors[:-1])))
        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
        # output
        self.log_prob = self.dist.log_prob
        self.prob = self.dist.prob
