import os

import numpy as np
import copy
import functools
import collections
import multiprocessing

import tensorflow as tf  # pylint: ignore-module


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

def get_flat(var_list):
    return tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list]).numpy()

def set_from_flat(var_list, theta):
    shapes = list(map(var_shape, var_list))
    start = 0
    for (shape, v) in zip(shapes, var_list):
        size = intprod(shape)
        v.assign(tf.reshape(theta[start : start + size], shape))
        start += size

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

def nn(layers_sizes):
    """Creates a simple neural network model
    """
    model = tf.keras.Sequential()
    for i, size in enumerate(layers_sizes):
        model.add(tf.keras.layers.Dense(
            units=size,
            activation=tf.keras.layers.ReLU() if i < len(layers_sizes) - 1 else None,
        ))
    return model

class Actor(tf.keras.Model):

  def __init__(self):
    super(Actor, self).__init__()
    self.nn = nn([3,16])
    self.nn2 = nn([3,1])

  def call(self, inputs):
    x = self.nn(inputs)
    self.result = self.nn2(x)

if __name__ == "__main__":
    model1 = nn([3,8,9,1])
    model2 = nn([3,8,9,1])
    model3 = Actor()

    input = np.array((1.,2.,3.,4.)).reshape(-1,1)

    # print(model(input))
    # print(model2(input))
    print(model3.nn.trainable_variables)
    model3(input)
    print(model3.nn.summary())
    # print(model.trainable_variables)
    # print(model2.trainable_variables)
    # print(model3.trainable_variables)




