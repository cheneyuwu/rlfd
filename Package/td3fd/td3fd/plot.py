import json
import os
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from td3fd.util.cmd_util import ArgParser
from td3fd.util.reader_util import load_csv

matplotlib.use("Agg")  # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


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


def strip(xs, length=0):
    minlen = length if length != 0 else np.min([len(x) for x in xs])
    stripped_xs = []
    for x in xs:
        assert x.shape[0] >= minlen
        stripped_xs.append(x[:minlen])
    return np.array(stripped_xs)


def smooth_reward_curve(x, y, size=50):
    k = int(np.ceil(len(x) / size))  # half width
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
                paramsjson = os.path.join(dirname, "params_renamed.json")  # search for the renamed file first
                if not os.path.exists(paramsjson):
                    # continue # Note: this is only for plotting old figures
                    paramsjson = os.path.join(dirname, "params.json")
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
                x, y = smooth_reward_curve(x, y, max(3, len(x) / 10))
            assert x.shape == y.shape, (x.shape, y.shape)

            if env_id not in data:
                data[env_id] = {}
            if xy not in data[env_id]:
                data[env_id][xy] = {}
            if config not in data[env_id][xy]:
                data[env_id][xy][config] = []
            data[env_id][xy][config].append((x, y))

    fig = plt.figure()
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.35, top=0.9, wspace=0.25, hspace=0.25)
    # set sizes
    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig.clf()
    for env_n, env_id in enumerate(sorted(data.keys()), 1):
        print("Creating plots for environment: {}".format(env_id))
        for i, xy in enumerate(data[env_id].keys(), 1):
            # colors = ["r", "g", "b", "m", "c", "k", "y"]
            colors = cm.jet(np.linspace(0, 1.0, len(data[env_id][xy].keys())))
            markers = ["o", "v", "s", "d", "p", "h"]
            ax = fig.add_subplot(len(xys), len(data.keys()), env_n + i * (len(xys) - 1))
            x_label = xy.split(":")[0]
            y_label = xy.split(":")[1]
            for j, config in enumerate(sorted(data[env_id][xy].keys())):
                xs, ys = zip(*data[env_id][xy][config])
                if config == "default":
                    continue

                # CHANGE! either pad with nan or strip to the minimum length
                required_length = 0
                # xs, ys = pad(xs), pad(ys)
                xs, ys = strip(xs, required_length), strip(ys, required_length)
                assert xs.shape == ys.shape

                x = xs[0]
                # plot percentile
                # ax.plot(x, np.nanmedian(ys, axis=0), label=config)
                # ax.fill_between(x, np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
                # or plot mean var
                mean_y = np.nanmean(ys, axis=0)
                stddev_y = np.nanstd(ys, axis=0)
                ax.plot(
                    x,
                    mean_y,
                    label=config,
                    color=colors[j % len(colors)],
                    lw=2,
                    marker=markers[j % len(markers)],
                    ms=8,
                    markevery=max(1, int(x.shape[0] / 20)),
                )
                ax.fill_between(
                    x, mean_y - 1.0 * stddev_y, mean_y + 1.0 * stddev_y, alpha=0.2, color=colors[j % len(colors)]
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.tick_params(axis="x", pad=5, length=5, width=1)
                ax.tick_params(axis="y", pad=5, length=5, width=1)
                ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")
                ax.set_title(env_id)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(True)
                ax.spines["left"].set_visible(True)
                # use ax level legend
                # ax.legend(fontsize=5)
            num_lines = len(data[env_id][xy].keys())
    # use fig level legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(2, int(num_lines)), frameon=False)
    fig.set_size_inches(5 * len(data.keys()), 1 + 5 * len(xys))
    save_path = os.path.join(target_dir, "Result_{}.png".format("plot_name"))
    print("Saving image to " + save_path)
    plt.savefig(save_path, dpi=200)
    plt.show()


def main(dirs, xys, save_path=None, smooth=0, **kwargs):
    results = load_results(dirs)
    # get directory to save results
    target_dir = save_path if save_path else dirs[0]
    plot_results(results, xys, target_dir, smooth)


ap = ArgParser()
ap.parser.add_argument("--dirs", help="target or list of dirs", type=str, nargs="+", default=[os.getcwd()])
ap.parser.add_argument("--save_path", help="plot saving directory", type=str, default=os.getcwd())
ap.parser.add_argument(
    "--xy",
    help="value on x and y axis, splitted by :",
    type=str,
    default=[
        "epoch:test/success_rate",
        "epoch:test/total_shaping_reward",
        "epoch:test/total_reward",
        "epoch:test/mean_Q",
        "epoch:test/mean_Q_plus_P",
    ],
    action="append",
    dest="xys",
)
ap.parser.add_argument("--smooth", help="smooth the curve", type=int, default=0)

if __name__ == "__main__":
    ap.parse(sys.argv)
    main(**ap.get_dict())
