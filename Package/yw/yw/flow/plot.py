import os
import sys
import os.path as osp

import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import json

from yw.util.cmd_util import ArgParser
from yw.util.reader_util import load_csv


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])
    padded_xs = []
    for x in xs:
        assert x.shape[0] <= maxlen
        if x.shape[0] == maxlen:
            padded_xs.append(x)
        else:
            padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
            x_padded = np.concatenate([x, padding], axis=0)
            assert x_padded.shape[1:] == x.shape[1:]
            assert x_padded.shape[0] == maxlen
            padded_xs.append(x_padded)
    return np.array(padded_xs)


def strip(xs):
    minlen = np.min([len(x) for x in xs])
    stripped_xs = []
    for x in xs:
        assert x.shape[0] >= minlen
        stripped_xs.append(x[:minlen])
    return np.array(stripped_xs)


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode="same") / np.convolve(
        np.ones_like(y), np.ones(2 * k + 1), mode="same"
    )
    return xsmoo, ysmoo


def load_results(root_dir_or_dirs):
    """
    Load summaries of runs from a list of directories (including subdirectories)
    Looking for directories with both params.json and progress.csv.

    Arguments:

    Returns:
        allresults - list of dicts that contains "progress" and "params".
    """
    if isinstance(root_dir_or_dirs, str):
        rootdirs = [osp.expanduser(root_dir_or_dirs)]
    else:
        rootdirs = [osp.expanduser(d) for d in root_dir_or_dirs]
    allresults = []
    for rootdir in rootdirs:
        assert osp.exists(rootdir), "%s doesn't exist" % rootdir
        for dirname, _, files in os.walk(rootdir):
            if all([file in files for file in ["params.json", "progress.csv"]]):
                result = {"dirname": dirname}
                progcsv = os.path.join(dirname, "progress.csv")
                result["progress"] = load_csv(progcsv)
                if result["progress"] is None:
                    continue
                paramsjson = os.path.join(dirname, "params.json")
                # paramsjson = os.path.join(dirname, "params_renamed.json") # put here for the rename in launch.py
                with open(paramsjson, "r") as f:
                    result["params"] = json.load(f)
                allresults.append(result)
    return allresults


def plot_results(allresults, xys, target_dir, smooth=0):

    # collect data
    data = {}
    for results in allresults:
        # get environment name and algorithm configuration summary (should always exist)
        env_id = results["params"]["env_name"].replace("Dense", "")
        config = results["params"]["config"]
        assert config != ""

        for xy in xys:
            x = results["progress"][xy.split(":")[0]]
            y = results["progress"][xy.split(":")[1]]

            # Process and smooth data.
            if smooth:
                x, y = smooth_reward_curve(x, y)
            assert x.shape == y.shape

            if env_id not in data:
                data[env_id] = {}
            if xy not in data[env_id]:
                data[env_id][xy] = {}
            if config not in data[env_id][xy]:
                data[env_id][xy][config] = []
            data[env_id][xy][config].append((x, y))

    # each environment goes to one image
    fig = plt.figure()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.25, hspace=0.25)
    for env_id in sorted(data.keys()):
        print("Creating plots for environment: {}".format(env_id))

        fig.clf()
        for i, xy in enumerate(data[env_id].keys(), 1):
            # colors = ["r", "g", "b", "c", "m", "y", "k"]
            colors = cm.jet(np.linspace(0, 1.0, len(data[env_id][xy].keys())))
            ax = fig.add_subplot(1, len(xys), i)
            x_label = xy.split(":")[0]
            y_label = xy.split(":")[1]
            for j, config in enumerate(sorted(data[env_id][xy].keys())):
                xs, ys = zip(*data[env_id][xy][config])
                # either pad with nan or strip to the minimum length
                # xs, ys = pad(xs), pad(ys)
                xs, ys = strip(xs), strip(ys)
                assert xs.shape == ys.shape

                # from openai spinning up
                # ax.plot(xs[0], np.nanmedian(ys, axis=0), label=config)
                # ax.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
                # ours
                mean_y = np.nanmean(ys, axis=0)
                var_y = np.nanstd(ys, axis=0)
                ax.plot(xs[0], mean_y, label=config, color=colors[j % len(colors)])
                # ax.fill_between(xs[0], mean_y - var_y, mean_y + var_y, alpha=0.5, color=colors[j % len(colors)])
                # ax.fill_between(
                #     xs[0], mean_y - 3 * var_y, mean_y + 3 * var_y, alpha=0.25, color=colors[j % len(colors)]
                # )

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend(fontsize=5)
        fig.set_size_inches(4 * len(xys), 4)
        fig.suptitle(env_id)
        save_path = os.path.join(target_dir, "fig_{}.png".format(env_id))
        print("Saving image to " + save_path)
        plt.savefig(save_path, dpi=200)
    plt.show()


def main(dirs, xys, save_path=None, smooth=0, **kwargs):
    results = load_results(dirs)
    # get directory to save results
    target_dir = save_path if save_path else dirs[0]
    plot_results(results, xys, target_dir, smooth)


ap = ArgParser()
ap.parser.add_argument("--dir", help="result directory", type=str, action="append", dest="dirs")
ap.parser.add_argument("--save_path", help="plot saving directory", type=str, default=None)
ap.parser.add_argument(
    "--xy",
    help="value on x and y axis, splitted by :",
    type=str,
    # default=["epoch:test/success_rate"],
    action="append",
    dest="xys",
)
ap.parser.add_argument("--smooth", help="smooth the curve", type=int, default=0)

if __name__ == "__main__":
    ap.parse(sys.argv)
    main(**ap.get_dict())
