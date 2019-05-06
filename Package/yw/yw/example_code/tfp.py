""" Simple example of variational inference
    See notes on variational inference.
"""
import os
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.examples.tutorials.mnist import input_data

# used for argument parse
FLAGS = None

def extract_samples_Fn(data, num_classes):
    index_list = []
    for sample_index in range(data.shape[0]):
        label = data[sample_index]
        if label < num_classes:
            index_list.append(sample_index)
    return index_list

def process_data(num_classes):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    index_list_train = extract_samples_Fn(y_train, num_classes)
    index_list_test = extract_samples_Fn(y_test, num_classes)
    data = {}
    data['train/image'] = x_train[index_list_train].reshape([x_train[index_list_train].shape[0], -1])
    data['train/label'] = y_train[index_list_train]
    data['test/image'] = x_test[index_list_test].reshape([x_train[index_list_test].shape[0], -1])
    data['test/label'] = y_test[index_list_test]
    return data

def logistic_model(input,num_classes):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    # A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
    logits = tf.contrib.layers.fully_connected(inputs=input, num_outputs = num_classes, scope='fc')
    return logits

def baysian_logistic_model(input, num_classes):
    with tf.variable_scope('model'):
        model = tf.keras.Sequential([
            tfp.layers.DenseReparameterization(num_classes),
        ])
        model2 = tf.keras.Sequential([
            tfp.layers.DenseReparameterization(num_classes),
        ])
        output1 = model(input) 
        output2 = model(input)
        output3 = model([input, input])
    return model, output1, output2, output3


def logit_loss(logits, label_one_hot):
    """
    compute the loss by comparing the predicted value to the actual label.
    :param X: The output from model.
    :param Y: The label.
    :return: The loss over the samples.
    """
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_one_hot))
    return loss

def train(loss, learning_rate = 0.00001):
    with tf.name_scope('train_op'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients_and_variables = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients_and_variables)
    return train_op

def check_accuracy(logits, label_one_hot):
    with tf.name_scope('accuracy'):
        prediction_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))
    return accuracy

def global_vars():
    res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    return res

def main(_):
    # get data
    data = process_data(FLAGS.num_classes)

    # Dimentionality of train
    dimensionality_train = data['train/image'].shape
    # Dimensions
    num_train_samples = dimensionality_train[0]
    num_features = dimensionality_train[1]

    # build graph
    # input data placeholder 
    image_place = tf.placeholder(tf.float32, shape=([None, int(num_features)]), name='image')
    label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
    label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
    # model
    model, logits, logits2, logits3 = baysian_logistic_model(image_place, FLAGS.num_classes)
    vars = global_vars()
    for i in vars:
        print(i)
    # loss
    loss = logit_loss(logits, label_one_hot)  + 0.0001*sum(model.losses)
    # train_op
    # learning_rate = train_helper(loss, num_train_samples, FLAGS.batch_size, FLAGS.num_epochs_per_decay, FLAGS.learning_rate_decay_factor, FLAGS.initial_learning_rate)
    train_op = train(loss, 0.001)
    # Evaluate the model
    accuracy = check_accuracy(logits, label_one_hot)

    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.num_epochs):
            total_batch_training = int(data['train/image'].shape[0] / FLAGS.batch_size)
            # go through the batches
            for batch_num in range(total_batch_training):

                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num + 1) * FLAGS.batch_size

                # Fit training using batch data
                train_batch_data, train_batch_label = data['train/image'][start_idx:end_idx], data['train/label'][
                                                                                            start_idx:end_idx]
                # Run optimization op (backprop) and Calculate batch loss and accuracy
                # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
                batch_loss, _, model_loss, l1, l2, l3 = sess.run([loss, train_op, model.losses, logits, logits2, logits3],
                    feed_dict={image_place: train_batch_data,
                            label_place: train_batch_label})
                print("logits 1 is: " + "{:.5f}".format(l1[0][0]) + "; logits 2 is : " + "{:.5f}".format(l2[0][0]) + "; logits 3 is : " + "{:.5f}".format(l3[0][0][0])+ "{:.5f}".format(l3[1][0][0]) )

            print("Epoch " + str(epoch + 1) + ", Training Loss= " + \
                "{:.5f}".format(batch_loss)+ ", Model Loss= " + "{:.1f}".format(sum(model_loss)))

        # Evaluation of the model
        test_accuracy = 100 * sess.run(accuracy, feed_dict={
            image_place: data['test/image'],
            label_place: data['test/label']})
        print("Final Test Accuracy is %% %.2f" % test_accuracy)

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str,
                        default='.',
                        help='Directory for log data')
    parser.add_argument('--num_classes', type=int,
                        default=3,
                        help='Number of classes')
    parser.add_argument('--batch_size', type=int,
                        default=512,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int,
                        default=10,
                        help='Batch size')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)