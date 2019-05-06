"""RL Project

Lit tests for demo_policy

"""
import os
import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from yw.util.lit_util import build_input_pipeline, toy_logistic_data, toy_regression_data, visualize_decision
from yw.example_code.optimizer import Adam
from yw.util.tf_util import flatten_grads


def ligistic_regression():
    """BNN implementation of simple logistic regression
    """

    batch_size = 32
    num_examples = 100
    learning_rate = 0.01
    max_steps = 1000
    num_monte_carlo = 50

    w_true, b_true, x, y = toy_logistic_data(num_examples, 2, 5.0)
    features, labels = build_input_pipeline(x, y, batch_size)

    model = tf.keras.Sequential([tfp.layers.DenseReparameterization(128), tfp.layers.DenseReparameterization(1)])
    logits = model(features)
    labels_distribution = tfd.Bernoulli(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(labels))
    kl = sum(model.losses) / num_examples
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.cast(logits > 0, dtype=tf.int32)
    accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = tf.InteractiveSession()
    sess.run(init_op)

    # Fit the model to data.
    for step in range(max_steps):
        _ = sess.run([train_op, accuracy_update_op])
        if step % 100 == 0:
            loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
            print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(step, loss_value, accuracy_value))

    # Visualize some draws from the weights posterior.
    w_draw = model.layers[0].kernel_posterior.sample()
    b_draw = model.layers[0].bias_posterior.sample()
    candidate_w_bs = []
    for _ in range(num_monte_carlo):
        w, b = sess.run((w_draw, b_draw))
        candidate_w_bs.append((w, b))
    visualize_decision(x, y, (w_true, b_true), candidate_w_bs)


def linear_regression():
    """BNN implementation of simple linear regression
    """

    batch_size = 32
    num_examples = 32
    learning_rate = 0.01
    max_steps = 100000

    x, y = toy_regression_data(num_examples, input_size=1)
    z = np.zeros((num_examples,1))
    x_con = np.concatenate((z,z,x,z), axis=1)
    features, labels = build_input_pipeline(x_con, y, batch_size)

    model = tf.keras.Sequential(
        [
            tfp.layers.DenseReparameterization(32, activation=tf.nn.tanh),
            tfp.layers.DenseReparameterization(32, activation=tf.nn.tanh),
            tfp.layers.DenseReparameterization(1),
        ]
    )
    logits = model(features)
    labels_distribution = tfd.Normal(loc=tf.cast(logits, tf.float32), scale=tf.cast(0.1, tf.float32))
    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(input_tensor=labels_distribution.log_prob(tf.cast(labels, tf.float32)))
    kl = sum(model.losses) / num_examples
    elbo_loss = neg_log_likelihood + tf.cast(kl, tf.float32)

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    # accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.minimize(elbo_loss)

    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
    grads_tf = tf.gradients(elbo_loss, trainables)
    grad_tf = flatten_grads(grads=grads_tf, var_list=trainables)
    adam = Adam(trainables,  scale_grad_by_procs=False)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    test_data = tf.placeholder(tf.float32, shape=(None, 4))
    test_output = []
    for i in range(50):
        test_output.append(model(test_data))

    sess = tf.InteractiveSession()
    sess.run(init_op)

    plt.ion()
    # Fit the model to data.
    for step in range(max_steps):
        # _ = sess.run(train_op)
        grad = sess.run(grad_tf)
        adam.update(grad, 0.01)
        if step % 100 == 0:
            x_test = np.linspace(-np.pi, np.pi,num_examples).reshape((-1,1))
            stack = np.zeros((num_examples,1))
            x_test2 = np.concatenate((stack,stack,x_test,stack), axis=1)
            loss_value = sess.run(elbo_loss, feed_dict={test_data: x_test2})
            print("Step: {:>3d} Loss: {:.3f}".format(step, loss_value))
            mean, stddiv = tf.nn.moments(tf.concat(test_output, axis=1), 1)
            m, std = sess.run([mean, stddiv], feed_dict={test_data: x_test2})

            y1 = (m - 1 * std).reshape((-1))
            y2 = (m + 1 * std).reshape((-1))
            plt.fill_between(x_test.reshape((-1)), y1, y2, where=y2 > y1, facecolor="black", alpha=0.5)
            plt.plot(x_test, m)
            plt.scatter(x[:,0], y)
            plt.show()
            plt.pause(0.01)
            plt.cla()


if __name__ == "__main__":
    linear_regression()

