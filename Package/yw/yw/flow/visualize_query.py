import os
import sys
import numpy as np
import pickle

import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.animation as animation


from yw.util.cmd_util import ArgParser


def visualize_potential_surface(ax, res):

    ax.plot_surface(*res["o"], res["potential"])
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_zlabel("potential")
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
        + ax.get_zticklabels()
    ):
        item.set_fontsize(6)


def visualize_action(ax, res, plot_opts={}):
    for i in range(res["o"].shape[0]):
        ax.arrow(*res["o"][i], *(res["u"][i] * 0.05), head_width=0.01, **plot_opts)
    ax.axis([-1, 0, -1, 0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def create_animate(frame, data):

    # use the default figure
    pl.figure(0)
    gs = gridspec.GridSpec(1, 1)

    # create a new axes for action
    ax = pl.subplot(gs[0, 0])
    ax.clear()
    # get the result
    res = data[frame]
    # plot
    visualize_action(ax, res)


def create_plot(load_dirs):

    # Load data
    queries = {"query_action": None, "query_potential_surface": None}
    for q_k in queries.keys():
        data = {}
        for load_dir in load_dirs:
            exp_name = load_dir.split("/")[-1]
            if q_k in os.listdir(load_dir):
                q = os.path.join(load_dir, q_k)
                result_files = os.listdir(q)
                result_files.sort()
                if result_files != []:
                    data[exp_name] = {**np.load(os.path.join(q, result_files[-1]))}
        queries[q_k] = data

    # Plot query actions
    if queries["query_action"] != None:
        # get data
        data = queries["query_action"]
        # create a new figure
        plt.figure(0)
        
        # merge plots
        # gs = gridspec.GridSpec(1, 1)
        # ax = plt.subplot(gs[0, 0])
        # ax.clear()
        # col = cm.jet(np.linspace(0, 1, len(data.keys())))
        # col_patch = []
        # for c, k in enumerate(data.keys()):
        #     visualize_action(ax, data[k], plot_opts={"color": col[c]})
        #     col_patch.append(mpatches.Patch(color=col[c], label=k))
        # ax.legend(handles=col_patch, loc="lower left")
        # ax.set_title("action")
        
        # do not merge plots
        gs = gridspec.GridSpec(1, len(data.keys()))
        col = cm.jet(np.linspace(0, 1, len(data.keys())))
        for c, k in enumerate(data.keys()):
            ax = plt.subplot(gs[0, c])
            ax.clear()
            visualize_action(ax, data[k], plot_opts={"color": col[c]})
            ax.legend(handles=[mpatches.Patch(color=col[c], label=k)], loc="lower left")

    if queries["query_potential_surface"] != None:
        # get data
        data = queries["query_potential_surface"]
        # create a new figure
        plt.figure(1)
        gs = gridspec.GridSpec(1, len(data.keys()))
        # create a new axes for action
        for c, k in enumerate(data.keys()):
            ax = plt.subplot(gs[0, c], projection="3d")
            ax.clear()
            # plot
            visualize_potential_surface(ax, data[k])
            ax.set_title(k)


def main(mode, load_dirs, save, **kwargs):

    if mode == "plot":

        create_plot(load_dirs)

        if save:
            res_store_dir = os.path.join(load_dir, "../uncertainty.mp4")
            print("Storing animation result to {}".format(res_store_dir))
            res.save(res_store_dir, dpi=200)
        else:
            plt.show()  # close the figure and then continue

    else:
        # Plot animation
        fig = plt.figure()
        res = animation.FuncAnimation(fig, create_animate, len(files), fargs=(data,), repeat=False, interval=300)

        if save:
            res_store_dir = os.path.join(load_dir, "../uncertainty.mp4")
            print("Storing animation result to {}".format(res_store_dir))
            res.save(res_store_dir, dpi=200)
        else:
            plt.show()  # close the figure and then continue


ap = ArgParser()
ap.parser.add_argument("--mode", help="plot the last or create an animation", type=str, default="plot")
ap.parser.add_argument(
    "--load_dir", help="which script to run", type=str, action="append", default=None, dest="load_dirs"
)
ap.parser.add_argument("--save", help="save gifure", type=int, default=0)

if __name__ == "__main__":
    ap.parse(sys.argv)
    main(**ap.get_dict())
