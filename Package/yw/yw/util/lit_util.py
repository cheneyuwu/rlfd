"""RL Project

This file contains basic utility functions for creating test data.

"""
import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def build_input_pipeline(x, y, batch_size):
    """Build a Dataset iterator for supervised classification.

    Args:
        x: Numpy `array` of features, indexed by the first dimension.
        y: Numpy `array` of labels, with the same first dimension as `x`.
        batch_size: Number of elements in each training batch.

    Returns:
        batch_features: `Tensor` feed  features, of shape
        `[batch_size] + x.shape[1:]`.
        batch_labels: `Tensor` feed of labels, of shape
        `[batch_size] + y.shape[1:]`.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    training_iterator = training_dataset.repeat().shuffle(100).batch(batch_size).make_one_shot_iterator()
    batch_features, batch_labels = training_iterator.get_next()
    return batch_features, batch_labels


def toy_logistic_data(num_examples, input_size=2, weights_prior_stddev=5.0):
    """Generates synthetic data for binary classification.

    Args:
        num_examples: The number of samples to generate (scalar Python `int`).
        input_size: The input space dimension (scalar Python `int`).
        weights_prior_stddev: The prior standard deviation of the weight
        vector. (scalar Python `float`).

    Returns:
        random_weights: Sampled weights as a Numpy `array` of shape
        `[input_size]`.
        random_bias: Sampled bias as a scalar Python `float`.
        design_matrix: Points sampled uniformly from the cube `[-1,
        1]^{input_size}`, as a Numpy `array` of shape `(num_examples,
        input_size)`.
        labels: Labels sampled from the logistic model `p(label=1) =
        logistic(dot(features, random_weights) + random_bias)`, as a Numpy
        `int32` `array` of shape `(num_examples, 1)`.
    """
    random_weights = weights_prior_stddev * np.random.randn(input_size)
    random_bias = np.random.randn()
    design_matrix = np.random.rand(num_examples, input_size) * 2 - 1
    logits = np.reshape(np.dot(design_matrix, random_weights) + random_bias, (-1, 1))
    p_labels = 1.0 / (1 + np.exp(-logits))
    labels = np.int32(p_labels > np.random.rand(num_examples, 1))
    return random_weights, random_bias, np.float32(design_matrix), labels


def toy_regression_data(num_examples, input_range=np.pi, random=False):
    """Generates synthetic data for regression.

    Args:
        num_examples         (int)   The number of samples to generate
        weights_prior_stddev (float) The prior standard deviation of the weight vector.

    Returns:
        x (float) - Data uniformly distributed on (-input_range, +inputrange)
        y (float) - Desired value sampled from a function
    """
    if random:
        x = np.random.uniform(-input_range, input_range, num_examples).reshpae(-1,1)
    else:
        x = np.linspace(-input_range, input_range, num_examples).reshape(-1, 1)
    w = 2  + np.random.randn(1)
    y = (
        2 * (np.sin(x * w)).reshape((num_examples, 1))
         + 2 * (np.cos(x * (w + 3))).reshape((num_examples, 1))
         + 0.1 * np.random.randn(num_examples, 1)
    )
    return x, y


def visualize_decision(features, labels, true_w_b, candidate_w_bs):
    """Utility method to visualize decision boundaries in R^2.

    Args:
      features:       Input points, as a Numpy `array` of shape `[num_examples, 2]`.
      labels:         Numpy `float`-like array of shape `[num_examples, 1]` giving a label for each point.
      true_w_b:       A `tuple` `(w, b)` where `w` is a Numpy array of shape `[2]` and `b` is a scalar `float`, interpreted as a
                      decision rule of the form `dot(features, w) + b > 0`.
      candidate_w_bs: Python `iterable` containing tuples of the same form as true_w_b.
      fname:          The filename to save the plot as a PNG image (Python `str`).

    """
    plt.scatter(features[:, 0], features[:, 1], c=np.float32(labels[:, 0]), edgecolors="k")

    def plot_weights(w, b, **kwargs):
        w1, w2 = w
        x1s = np.linspace(-1, 1, 100)
        x2s = -(w1 * x1s + b) / w2
        plt.plot(x1s, x2s, **kwargs)

    for w, b in candidate_w_bs:
        plot_weights(w, b, alpha=1.0 / np.sqrt(len(candidate_w_bs)), lw=1, color="blue")

    if true_w_b is not None:
        plot_weights(*true_w_b, lw=4, color="green", label="true separator")

    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.show()

    # canvas.print_figure(fname, format="png")
    # print("saved {}".format(fname))
