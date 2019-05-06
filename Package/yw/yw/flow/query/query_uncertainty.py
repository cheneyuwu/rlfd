import click
import os
import numpy as np
import pickle

import glob2
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
    x = res["x"]
    y = res["y"]
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    xy_flat = np.concatenate((x_flat, y_flat), axis=1)

    # Add targets to the surface and arrow plot
    surface = {}
    surface["rl_q"] = res["rl_q"].reshape(-1, *x.shape)
    surface["demo_q_upper"] = res["demo_q_mean"].reshape(-1, *x.shape) + np.sqrt(res["demo_q_var"].reshape(-1, *x.shape))
    surface["demo_q_lower"] = res["demo_q_mean"].reshape(-1, *x.shape) - np.sqrt(res["demo_q_var"].reshape(-1, *x.shape))
    n_sample = surface["rl_q"].shape[0]

    n_fig_row = int(np.sqrt(n_sample))
    n_fig_col = int(np.ceil(n_sample / n_fig_row))
    fig.clf()
    for i in range(n_sample):
        ax = fig.add_subplot(n_fig_row, n_fig_col, i + 1, projection="3d")
        col = cm.jet(np.linspace(0, 1, len(surface.keys())))
        col_patch = []
        for c, k in enumerate(surface.keys()):
            ax.plot_surface(x, y, surface[k][i], color=col[c])
            col_patch.append(mpatches.Patch(color=col[c], label=k))
        ax.legend(handles=col_patch, prop={'size': 3})
        ax.set_title("Sample at ({:.1f}, {:.1f})".format(res["o"][i][0][0], res["o"][i][0][1]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("q")
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            item.set_fontsize(6)

def uncertainty(fig, res):
    # Get the x axis and y axis
    x = res["x"]
    y = res["y"]
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    xy_flat = np.concatenate((x_flat, y_flat), axis=1)

    # Add targets to the surface and arrow plot
    surface = {}
    surface["q_var"] = np.sqrt(res["demo_q_var"].reshape(-1, *x.shape))
    n_sample = surface["q_var"].shape[0]

    n_fig_row = int(np.sqrt(n_sample))
    n_fig_col = int(np.ceil(n_sample / n_fig_row))
    fig.clf()
    for i in range(n_sample):
        ax = fig.add_subplot(n_fig_row, n_fig_col, i + 1, projection="3d")
        col = cm.jet(np.linspace(0, 1, len(surface.keys())))
        col_patch = []
        for c, k in enumerate(surface.keys()):
            ax.plot_surface(x, y, surface[k][i], color=col[c])
            col_patch.append(mpatches.Patch(color=col[c], label=k))
        ax.legend(handles=col_patch, prop={'size': 6})
        ax.set_title("Sample at ({:.1f}, {:.1f})".format(res["o"][i][0][0], res["o"][i][0][1]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("q")
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            item.set_fontsize(6)

def plot_animation(load_dir, save):
    # Load data
    files = os.listdir(load_dir)
    files.sort()
    data = []
    for f in files:
        data.append({**np.load(os.path.join(load_dir, f))})

    # Plot animation
    fig = plt.figure(0)
    ani = animation.FuncAnimation(fig, animate, len(files), fargs=(fig, data), repeat=False, interval=300)

    # Plot uncertainty of demonstration nn over s a space
    fig2 = plt.figure(1)
    res = data[-1]
    uncertainty(fig2, res)

    if save:
        ani_store_dir = os.path.join(load_dir, "../uncertainty.mp4")
        print("Storing animation result to {}".format(ani_store_dir))
        ani.save(ani_store_dir, dpi=200)

        fig_store_dir = os.path.join(load_dir, "../uncertainty.jpg")
        print("Storing figure result to {}".format(fig_store_dir))
        fig2.savefig(fig_store_dir)
    else:
        plt.show()  # close the figure and then continue


@click.command()
@click.option("--load_dir", type=str, default=None, help="Directory for loading.")
@click.option("--save", type=int, default=0, help="Whether or not to save the video.")
def main(**kwargs):
    plot_animation(**kwargs)


if __name__ == "__main__":
    main()

