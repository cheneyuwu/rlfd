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
from yw.flow.query import visualize_query
from yw.flow.plot import load_results


def query(result, save):
    """Generate demo from policy file
    """
    # Setup
    config = result["params"]["config"]
    directory = result["dirname"]
    policy_file = os.path.join(directory, "rl/policy_latest.pkl")  # assumed to be in this directory

    # Load policy.
    # get a default session for the current graph
    tf.reset_default_graph()
    tf.InteractiveSession()
    with open(policy_file, "rb") as f:
        policy = pickle.load(f)

    # input data - used for both training and test set
    dim1 = 0
    dim2 = 1
    num_point = 24
    ls = np.linspace(-1.0, 1.0, num_point)
    o_1, o_2 = np.meshgrid(ls, ls)
    o_r = 0.0 * np.ones((num_point ** 2, policy.dimo))
    o_r[..., dim1 : dim1 + 1] = o_1.reshape(-1, 1)
    o_r[..., dim2 : dim2 + 1] = o_2.reshape(-1, 1)
    g_r = 0.0 * np.ones((num_point ** 2, policy.dimg))
    u_r = 2.0 * np.ones((num_point ** 2, policy.dimu))

    # Close the default session to prevent memory leaking
    tf.get_default_session().close()

    for loss_mode in ["surface_q_only", "surface_p_only", "surface_p_plus_q"]:

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
        inputs_tf["g"] = tf.placeholder(tf.float32, shape=(None, policy.dimg)) if policy.dimg != 0 else None
        inputs_tf["u"] = tf.placeholder(tf.float32, shape=(None, policy.dimu))

        q_value = policy.main.critic1(o=inputs_tf["o"], g=inputs_tf["g"], u=inputs_tf["u"])
        if policy.demo_shaping != None:
            potential = policy.demo_shaping.potential(o=inputs_tf["o"], g=inputs_tf["g"], u=inputs_tf["u"])
        else:
            potential = None

        if loss_mode == "surface_q_only":
            val = q_value
        elif loss_mode == "surface_p_only" and potential != None:
            val = potential
        elif loss_mode == "surface_p_plus_q" and potential != None:
            val = q_value + potential
        else:
            tf.get_default_session().close()
            continue

        feed = {inputs_tf["o"]: o_r, inputs_tf["u"]: u_r}
        if policy.dimg != 0:
            feed[inputs_tf["g"]] = g_r
        ret = policy.sess.run(val, feed_dict=feed)

        res = {"o": (o_1, o_2), "surf": ret.reshape((num_point, num_point))}

        if save:
            store_dir = os.path.join(directory, "query_" + loss_mode)
            os.makedirs(store_dir, exist_ok=True)
            filename = os.path.join(store_dir, "query_000.npz")
            logger.info("Query: offline policy {}, storing query results to {}".format(loss_mode, filename))
            np.savez_compressed(filename, **res)
        else:
            # plot the result on the fly
            pl.figure(0)
            gs = gridspec.GridSpec(1, 1)
            ax = pl.subplot(gs[0, 0], projection="3d")
            ax.clear()
            visualize_query.visualize_potential_surface(ax, res)
            pl.show()
            pl.pause(0.1)

        # Close the default session to prevent memory leaking
        tf.get_default_session().close()

    # UNCOMMENT this for simple environments
    return

    for loss_mode in ["optimized_q_only", "optimized_p_only", "optimized_p_plus_q"]:
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
        if policy.dimg != 0:
            inputs_tf["g"] = tf.placeholder(tf.float32, shape=(None, policy.dimg)) if policy.dimg != 0 else None
            state_tf = tf.concat((inputs_tf["o"], inputs_tf["g"]), axis=1)
        else:
            inputs_tf["g"] = None
            state_tf = inputs_tf["o"]

        with tf.variable_scope("query"):
            pi_tf = policy.max_u * tf.tanh(nn(state_tf, policy.layer_sizes + [policy.dimu]))

            # add loss function and trainer
            q_tf = policy.main.critic1(o=inputs_tf["o"], g=inputs_tf["g"], u=pi_tf)
            if policy.demo_shaping != None:
                p_tf = policy.demo_shaping.potential(o=inputs_tf["o"], g=inputs_tf["g"], u=pi_tf)
            else:
                p_tf = None

            if loss_mode == "optimized_q_only":
                loss_tf = -tf.reduce_mean(q_tf)
            elif loss_mode == "optimized_p_only" and p_tf != None:
                loss_tf = -tf.reduce_mean(p_tf)
            elif loss_mode == "optimized_p_plus_q" and p_tf != None:
                loss_tf = -tf.reduce_mean(p_tf) - tf.reduce_mean(q_tf)
            else:
                tf.get_default_session().close()
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
        feed = {inputs_tf["o"]: o_r}
        if policy.dimg != 0:
            feed[inputs_tf["g"]] = g_r
        for i in range(num_epochs):
            loss, _ = policy.sess.run([loss_tf, train_op], feed_dict=feed)
            if i % (num_epochs / 10) == (num_epochs / 10 - 1):
                logger.info("Query: offline policy {} -> epoch: {} loss: {}".format(loss_mode, i, loss))

        ret = policy.sess.run(pi_tf, feed_dict=feed)
        res = {"o": np.concatenate((o_1.reshape(-1, 1), o_2.reshape(-1, 1)), axis=1), "u": ret}

        if save:
            store_dir = os.path.join(directory, "query_" + loss_mode)
            os.makedirs(store_dir, exist_ok=True)
            filename = os.path.join(store_dir, "query_000.npz")
            logger.info("Query: offline policy {}, storing query results to {}".format(loss_mode, filename))
            np.savez_compressed(filename, **res)
        else:
            # plot the result on the fly
            pl.figure(0)  # create a new figure
            gs = gridspec.GridSpec(1, 1)
            ax = pl.subplot(gs[0, 0])
            ax.clear()
            visualize_query.visualize_action(ax, res)
            pl.show()
            pl.pause(0.1)

        # Close the default session to prevent memory leaking
        tf.get_default_session().close()


def main(directories, save, **kwargs):

    logger.configure()
    assert logger.get_dir() is not None

    # Allow load dir to be *
    all_results = load_results(directories)

    for result in all_results:
        query(result, save)


if __name__ == "__main__":
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()
    ap.parser.add_argument(
        "--directory", help="directories to load", type=str, action="append", default=None, dest="directories"
    )
    ap.parser.add_argument("--save", help="save for later plotting", type=int, default=1)
    ap.parse(sys.argv)

    main(**ap.get_dict())
