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

from yw.tool import logger
from yw.flow.plot import load_results


def visualize_potential_surface(ax, res):

    ax.plot_trisurf(res["o"][:, 0], res["o"][:, 1], res["surf"][:, 0])
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
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")


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


def create_plot(frame, fig, all_results, query_ls):

    # Plot frame
    fig.text(0.02, 0.02, "frame: {}".format(frame), ha="left", va="bottom", fontsize=14, color="r")

    # Load data
    data = {}
    for result in all_results:
        exp_name = result["params"]["config"]
        load_dir = result["dirname"]
        queries = {}
        for q_k in query_ls:
            if q_k in os.listdir(load_dir):
                q = os.path.join(load_dir, q_k)
                result_files = os.listdir(q)
                result_files.sort()
                if result_files != []:
                    queries[q_k] = {
                        **np.load(os.path.join(q, result_files[frame if frame < len(result_files) else -1]))
                    }
        if queries != {}:
            data[exp_name] = queries

    # Initialization
    # fig = plt.figure(0) # use the same figure
    num_rows = len(data.keys())
    num_cols = len(query_ls)
    # fig.set_size_inches(4 * num_cols, 4 * num_rows)
    gs = gridspec.GridSpec(num_rows, num_cols)
    col = cm.jet(np.linspace(0, 1.0, num_rows))

    for i, exp in enumerate(data.keys()):
        for j, query in enumerate(query_ls):

            if query not in data[exp].keys():
                continue

            if query in [
                "query_policy",
                "query_optimized_p_only",
                "query_optimized_q_only",
                "query_optimized_p_plus_q",
            ]:
                ax = plt.subplot(gs[i, j])
                ax.clear()
                visualize_action(ax, data[exp][query], plot_opts={"color": col[i]})
                ax.legend(handles=[mpatches.Patch(color=col[i], label=exp)], loc="lower left")

            if query in ["query_surface_p_only", "query_surface_q_only", "query_surface_p_plus_q"]:
                ax = plt.subplot(gs[i, j], projection="3d")
                ax.clear()
                visualize_potential_surface(ax, data[exp][query])
                ax.set_title(exp)

            fig.text(
                0.05,
                0.8 - 0.85 * i / num_rows,
                exp,
                ha="center",
                va="center",
                fontsize=14,
                color="r",
                rotation="vertical",
            )


def main(directories, save, mode="plot", **kwargs):

    logger.configure()
    assert logger.get_dir() is not None

    # Allow load dir to be *
    all_results = load_results(directories)
    # Data pre parser
    query_ls = [
        # "query_optimized_p_only",
        # "query_optimized_q_only",
        # "query_optimized_p_plus_q",
        # "query_policy",
        "query_surface_p_only",
        "query_surface_q_only",
        "query_surface_p_plus_q",
    ]
    data = {}
    num_frames = 0
    for result in all_results:
        exp_name = result["params"]["config"]
        load_dir = result["dirname"]
        queries = {}
        for q_k in query_ls:
            if q_k in os.listdir(load_dir):
                q = os.path.join(load_dir, q_k)
                result_files = os.listdir(q)
                num_frames = max(num_frames, len(result_files))
                if result_files != []:
                    queries[q_k] = 1
        if queries != {}:
            data[exp_name] = queries

    # Initialization
    fig = plt.figure(0)  # use the same figure
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.25, hspace=0.25)
    num_rows = len(data.keys())
    num_cols = len(query_ls)
    fig.set_size_inches(4 * num_cols, 4 * num_rows)

    for i, k in enumerate(query_ls):
        x = 0.14 + 0.83 / num_cols * i
        plt.figtext(
            x, 0.95, k.replace("query_", ""), ha="center", va="center", fontsize=14, color="r", rotation="horizontal"
        )

    if mode == "plot":

        create_plot(frame=-1, fig=fig, all_results=all_results, query_ls=query_ls)

        if save:
            res_store_dir = os.path.join(directories[0], "queries.png")
            print("Storing query plot to {}".format(res_store_dir))
            plt.savefig(res_store_dir, dpi=200)

    elif mode == "mv":
        # Plot animation
        fig = plt.figure(0)  # create figure before hand
        res = animation.FuncAnimation(
            fig, create_plot, num_frames, fargs=(fig, all_results, query_ls), repeat=False, interval=300
        )

        if save:
            res_store_dir = os.path.join(directories[0], "queries.mp4")
            print("Storing query animation to {}".format(res_store_dir))
            res.save(res_store_dir, dpi=200)

    else:
        return

    plt.show()  # close the figure and then continue


if __name__ == "__main__":
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()
    ap.parser.add_argument("--mode", help="plot the last or create an animation", type=str, default="plot")
    ap.parser.add_argument(
        "--directory", help="which script to run", type=str, action="append", default=None, dest="directories"
    )
    ap.parser.add_argument("--save", help="save gifure", type=int, default=0)
    ap.parse(sys.argv)
    main(**ap.get_dict())
