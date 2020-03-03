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
    assert policy.shaping != None

    o_tf = tf.convert_to_tensor(o, dtype=tf.float32)
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    u_tf = tf.convert_to_tensor(u, dtype=tf.float32)

    p_tf = policy.shaping.potential(o=o_tf, g=g_tf, u=u_tf)

    x = []
    y = []
    for var in np.arange(0, 1e-2, 1e-4):
        o_tf = tf.convert_to_tensor(o + np.random.normal(0.0, var, o.shape), dtype=tf.float32)
        g_tf = tf.convert_to_tensor(g + np.random.normal(0.0, var, g.shape), dtype=tf.float32)
        u_tf = tf.convert_to_tensor(u + np.random.normal(0.0, var, u.shape), dtype=tf.float32)
        p_tf = policy.shaping.potential(o=o_tf, g=g_tf, u=u_tf)
        x.append(var)
        y.append(np.mean(p_tf.numpy()))

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
