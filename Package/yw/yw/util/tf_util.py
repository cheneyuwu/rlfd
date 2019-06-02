import os

import numpy as np
import copy
import functools
import collections
import multiprocessing

import tensorflow as tf  # pylint: ignore-module


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
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
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
        raise ValueError('Unknown network type: {}'.format(name))
