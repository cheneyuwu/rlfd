import json
import os
import os.path as osp
import pickle
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm

from td3fd.plot import load_results
from td3fd.util.cmd_util import ArgParser
from td3fd.util.reader_util import load_csv

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# set sizes
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize


def get_potential(demo_data, vars, policy):

    with open(policy, "rb") as f:
        policy = pickle.load(f)

    num_eps = demo_data["u"].shape[0]
    eps_length = demo_data["u"].shape[1]
    o = demo_data["o"][:num_eps, :eps_length, ...].reshape((num_eps * eps_length, -1))
    g = demo_data["g"][:num_eps, :eps_length, ...].reshape((num_eps * eps_length, -1))
    u = demo_data["u"][:num_eps, :eps_length, ...].reshape((num_eps * eps_length, -1))
    assert policy.shaping != None

    potential = []
    for var in vars:
        o_tf = tf.convert_to_tensor(o + np.random.normal(0.0, var, o.shape), dtype=tf.float32)
        g_tf = tf.convert_to_tensor(g + np.random.normal(0.0, var, g.shape), dtype=tf.float32)
        u_tf = tf.convert_to_tensor(u + np.random.normal(0.0, var, u.shape), dtype=tf.float32)
        p_tf = policy.shaping.potential(o=o_tf, g=g_tf, u=u_tf)
        # potential = tf.sigmoid(potential)
        # potential = tf.where(potential > 10.0, tf.ones_like(potential) ,tf.zeros_like(potential))
        potential.append(np.mean(p_tf.numpy()))

    return potential

def check_potential(exp_dir, results):
    """Generate demo from policy file
    """
    demo_data = dict(np.load(os.path.join(exp_dir, "demo_data.npz")))

    max_var = 1.0
    vars = np.arange(0, max_var, max_var / 100.0)

    configs = {}
    for res in results:
        config = res["params"]["config"]
        policy = os.path.join(res["dirname"], "policies/policy_latest.pkl")
        if not "Shaping" in config:
            print("Skip ", config)
            continue
        print(config)
        if not config in configs.keys():
            configs[config] = []
        y = get_potential(demo_data, vars, policy)
        configs[config].append(y)

    fig = plt.figure()
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9, wspace=0.25, hspace=0.25)
    ax = fig.add_subplot(111)

    colors = ["r", "g", "b", "m", "c", "k", "y"]
    markers = ["o", "v", "s", "d", "p", "h"]

    for j, (config, ys) in enumerate(configs.items()):
        ys = np.array(ys)
        mean_y = np.nanmean(ys, axis=0)
        stddev_y = np.nanstd(ys, axis=0)
        ax.plot(
            vars,
            mean_y,
            label=config,
            color=colors[j % len(colors)],
            lw=2,
            marker=markers[j % len(markers)],
            ms=8,
            markevery=max(1, int(vars.shape[0] / 20)),
        )
        ax.fill_between(
            vars, mean_y - 1.0 * stddev_y, mean_y + 1.0 * stddev_y, alpha=0.2, color=colors[j % len(colors)]
        )

        ax.set_xlabel("Shaping Potential ($\Phi$)")
        ax.set_ylabel("Variance of Gaussian Noise ($\sigma^2$)")

        # ax.set_ylim(-20, -5)

        ax.tick_params(axis="x", pad=5, length=5, width=1)
        ax.tick_params(axis="y", pad=5, length=5, width=1)
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

        # ax.legend()

    # use fig level legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    save_path = os.path.join(exp_dir, "Check_potential.jpg")
    print("Result saved to", save_path)
    plt.savefig(save_path, dpi=200)
    plt.show()


def main(exp_dir, **kwargs):
    results = load_results(exp_dir)
    check_potential(exp_dir, results)