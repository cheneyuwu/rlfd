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

def plot(frame, fig, data):
    res = data[frame]
    # Get the x axis and y axis
    x = res["x"]
    y = res["y"]
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    xy_flat = np.concatenate((x_flat, y_flat), axis=1)

    # Add targets to the surface and arrow plot
    surface = {}
    arrow = {}
    for k in res.keys():
        if k == "x" or k == "y":
            continue
        res[k] = res[k].reshape((*x.shape, -1))
        if res[k].shape[-1] == 1:
            surface[k] = res[k].reshape(*x.shape)
        elif res[k].shape[-1] == 2:
            arrow[k] = res[k].reshape((-1, 2)) * 0.1 # scale down a little bit

    fig.clf()
    ax = fig.add_subplot(121, projection="3d")
    col = cm.jet(np.linspace(0, 1, len(surface.keys())))
    col_patch = []
    for c, k in enumerate(surface.keys()):
        ax.plot_surface(x, y, surface[k], color=col[c])
        col_patch.append(mpatches.Patch(color=col[c], label=k))
    ax.legend(handles=col_patch)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Q")

    ax = fig.add_subplot(122)
    col = cm.jet(np.linspace(0, 1, len(arrow.keys())))
    col_patch = []
    for c, k in enumerate(arrow.keys()):
        for i in range(arrow[k].shape[0]):
            ax.arrow(*xy_flat[i], *arrow[k][i], head_width=0.02, color=col[c])
        col_patch.append(mpatches.Patch(color=col[c], label=k))
    ax.axis([-1.2, 1.2, -1.2, 1.2])
    ax.legend(handles=col_patch)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_animation(load_dir, save):
    # Load data
    files = os.listdir(load_dir)
    files.sort()
    data = []
    for f in files:
        data.append({**np.load(os.path.join(load_dir, f))})
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, plot, len(files), fargs=(fig, data), repeat=False, interval=300)
    if save:
        store_dir = os.path.join(load_dir, "../query.mp4")
        print("Storing result to {}".format(store_dir))
        ani.save(store_dir)
    else:
        plt.show() # close the figure and then continue


@click.command()
@click.option("--load_dir", type=str, default=None, help="Directory for loading.")
@click.option("--save", type=int, default=0, help="Whether or not to save the video.")
def main(**kwargs):
    plot_animation(**kwargs)


if __name__ == "__main__":
    main()

