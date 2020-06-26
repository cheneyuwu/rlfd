import sys
import os
import pickle

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from td3fd import logger


def check_potential(exp_dir, policy):
    """Generate demo from policy file
    """
    # Load policy.
    # get a default session for the current graph
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.InteractiveSession()
    with open(policy, "rb") as f:
        policy = pickle.load(f)

    # input data - used for both training and test set
    demo_data = dict(np.load(os.path.join(exp_dir, "demo_data.npz")))
    num_eps = demo_data["u"].shape[0]
    eps_length = demo_data["u"].shape[1]
    o = demo_data["o"][:num_eps, :eps_length, ...].reshape((num_eps * eps_length, -1))
    g = demo_data["g"][:num_eps, :eps_length, ...].reshape((num_eps * eps_length, -1))
    u = demo_data["u"][:num_eps, :eps_length, ...].reshape((num_eps * eps_length, -1))
    print(o.shape, g.shape, u.shape)

    # input placeholders
    inputs_tf = {}
    inputs_tf["o"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *policy.dimo))
    inputs_tf["g"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *policy.dimg)) if policy.dimg != (0,) else None
    inputs_tf["u"] = tf.compat.v1.placeholder(tf.float32, shape=(None, *policy.dimu))
    # copied inputs
    input_o_tf = inputs_tf["o"]
    input_g_tf = inputs_tf["g"] if policy.dimg != (0,) else None
    input_u_tf = inputs_tf["u"]
    # normalized o g
    norm_input_o_tf = policy.o_stats.normalize(input_o_tf)
    norm_input_g_tf = policy.g_stats.normalize(input_g_tf) if policy.dimg != (0,) else None

    assert policy.demo_shaping != None
    q_tf = policy.main_critic(o=norm_input_o_tf, g=norm_input_g_tf, u=input_u_tf)
    p_tf = policy.demo_shaping.potential(o=input_o_tf, g=input_g_tf, u=input_u_tf)

    x = []
    y = []
    for var in np.arange(0, 1e-2, 1e-4):
        feed = {}
        feed[inputs_tf["o"]] = o + np.random.normal(0.0, var, o.shape)
        feed[inputs_tf["u"]] = u + np.random.normal(0.0, var, u.shape)
        if policy.dimg != (0,):
            feed[inputs_tf["g"]] = g + np.random.normal(0.0, var, g.shape)
        ret = policy.sess.run(p_tf, feed_dict=feed)

        x.append(var)
        y.append(np.mean(ret))

    # Close the default session to prevent memory leaking
    tf.get_default_session().close()

    plt.plot(x, y)
    plt.xlabel("var")
    plt.ylabel("potential")
    save_path = os.path.join(exp_dir, "check_potential.jpg")
    print("Result saved to", save_path)
    plt.savefig(save_path, dpi=200)
    plt.show()


def main(exp_dir, policy, **kwargs):

    logger.configure()
    assert logger.get_dir() is not None

    check_potential(exp_dir, policy)
