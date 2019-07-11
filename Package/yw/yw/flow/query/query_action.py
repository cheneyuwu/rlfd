import sys
import os
import pickle

import numpy as np
import tensorflow as tf

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

# DDPG Package import
from yw.tool import logger
from yw.ddpg_main import config
from yw.util.util import set_global_seeds
from yw.util.tf_util import nn
from yw.flow import visualize_query


def main(directory, save, **kwargs):
    """Generate demo from policy file
    """
    assert directory != None, "Must provide the base directory!"

    # Setup
    policy_file = os.path.join(directory, "rl/policy_latest.pkl")  # assumed to be in this directory
    # rank = MPI.COMM_WORLD.Get_rank() if MPI != None else 0

    # input data - used for both training and test set
    num_point = 24
    ls = np.linspace(-1.0, 1.0, num_point)
    o_1, o_2 = np.meshgrid(ls, ls)
    o_r = np.concatenate((o_1.reshape(-1, 1), o_2.reshape(-1, 1)), axis=1)
    g_r = 0.0 * np.ones((num_point ** 2, 2))

    for loss_mode in ["q_only", "p_only", "p_and_q"]:
        # reset default graph every time this function is called.
        tf.reset_default_graph()
        # Set random seed for the current graph
        set_global_seeds(0)
        # get a default session for the current graph
        tf.InteractiveSession()

        # Load policy.
        with open(policy_file, "rb") as f:
            policy = pickle.load(f)

        # input placeholders
        inputs_tf = {}
        inputs_tf["o"] = tf.placeholder(tf.float32, shape=(None, policy.dimo))
        inputs_tf["g"] = tf.placeholder(tf.float32, shape=(None, policy.dimg))
        state_tf = tf.concat((inputs_tf["o"], inputs_tf["g"]), axis=1)

        with tf.variable_scope("query"):
            pi_tf = policy.max_u * tf.tanh(nn(state_tf, [policy.hidden] * policy.layers + [policy.dimu]))

            # add loss function and trainer
            q_tf = policy.main.critic1(o=inputs_tf["o"], g=inputs_tf["g"], u=pi_tf)
            if "demo_shaping" in policy.__dict__.keys():
                p_tf = policy.demo_shaping.potential(o=inputs_tf["o"], g=inputs_tf["g"], u=pi_tf)
            else:
                p_tf = None

            if loss_mode == "q_only":
                loss_tf = -tf.reduce_mean(q_tf)
            elif loss_mode == "p_only" and p_tf != None:
                loss_tf = -tf.reduce_mean(p_tf)
            elif loss_mode == "p_and_q" and p_tf != None:
                loss_tf = -tf.reduce_mean(p_tf) - tf.reduce_mean(q_tf)
            else:
                continue

            train_op = tf.train.AdamOptimizer(1e-3).minimize(
                loss_tf,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name),
            )

            init = tf.initializers.variables(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            )

        policy.sess.run(init)
        num_epochs = 1000
        for i in range(num_epochs):
            loss, _ = policy.sess.run([loss_tf, train_op], feed_dict={inputs_tf["o"]: o_r, inputs_tf["g"]: g_r})
            if i % (num_epochs / 10) == (num_epochs / 10 - 1):
                logger.info("Query: policy optimized for {} -> epoch: {} loss: {}".format(loss_mode, i, loss))

        ret = policy.sess.run(pi_tf, feed_dict={inputs_tf["o"]: o_r, inputs_tf["g"]: g_r})
        res = {"o": o_r, "u": ret}

        if save:
            store_dir = os.path.join(directory, "query_" + loss_mode)
            logger.info("Query: policy optimized for {}, storing query results to {}".format(loss_mode, save))
            np.savez_compressed(save, **res)
        else:
            # plot the result on the fly
            pl.figure(0)  # create a new figure
            gs = gridspec.GridSpec(1, 1)
            ax = pl.subplot(gs[0, 0])
            ax.clear()
            visualize_query.visualize_action(ax, res)
            pl.show()
            pl.pause(1)

        # Close the default session to prevent memory leaking
        tf.get_default_session().close()


from yw.util.cmd_util import ArgParser

ap = ArgParser()
ap.parser.add_argument("--directory", help="directory for load and store", type=str, default=None)
ap.parser.add_argument("--save", help="save for later plotting", type=int, default=0)

if __name__ == "__main__":
    ap.parse(sys.argv)

    main(**ap.get_dict())
