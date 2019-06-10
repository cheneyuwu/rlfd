import click
import os
import numpy as np
import pickle

import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.animation as animation


def animate(frame, fig, data):
    res = data[frame]

    # Get the x axis and y axis
    x = res["o"]
    y = res["u"]
    # xy_flat = np.concatenate((x_flat, y_flat), axis=1)

    # Add targets to the surface and arrow plot
    surface = {}
    surface["q_var"] = res["q_var"]
    surface["q_mean"] = res["q_mean"]

    fig.clf()
    ax = fig.add_subplot(211, projection="3d")
    ax.plot_surface(x, y, surface["q_var"])
    # ax.set_zlim(0, 0.01)
    ax.set_xlabel("observation")
    ax.set_ylabel("action")
    ax.set_zlabel("q_var")
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        item.set_fontsize(6)

    ax = fig.add_subplot(212, projection="3d")
    ax.plot_surface(x, y, surface["q_mean"])
    ax.set_xlabel("observation")
    ax.set_ylabel("action")
    ax.set_zlabel("q_mean")
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        item.set_fontsize(6)

def plot_animation(load_dir, save, **kwargs):
    # Load data
    files = os.listdir(load_dir)
    files.sort()
    data = []
    for f in files:
        data.append({**np.load(os.path.join(load_dir, f))})

    # Plot animation
    fig = plt.figure(0)
    ani = animation.FuncAnimation(fig, animate, len(files), fargs=(fig, data), repeat=False, interval=300)

    if save:
        ani_store_dir = os.path.join(load_dir, "../uncertainty.mp4")
        print("Storing animation result to {}".format(ani_store_dir))
        ani.save(ani_store_dir, dpi=200)
    else:
        plt.show()  # close the figure and then continue

if __name__ == "__main__":
    import sys
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()

    ap.parser.add_argument("--load_dir", help="where to load the data", type=str, default=None)
    ap.parser.add_argument("--save", help="Whether or not to save the video.", type=int, default=0)
    ap.parse(sys.argv)

    plot_animation(**ap.get_dict())

