import os
import sys
import os.path as osp

import matplotlib

matplotlib.use("TkAgg")  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt

import numpy as np
import json

from yw.util.reader_util import load_csv


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


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
        for dirname, dirs, files in os.walk(rootdir):
            if all([file in files for file in ["params.json", "progress.csv"]]):
                result = {"dirname": dirname}
                progcsv = os.path.join(dirname, "progress.csv")
                result["progress"] = load_csv(progcsv)
                paramsjson = os.path.join(dirname, "params.json")
                with open(paramsjson, "r") as f:
                    result["params"] = json.load(f)
                allresults.append(result)
    return allresults


def plot_results(allresults, xys, target_dir, smooth=False):
    
    # collect data
    data = {}
    for results in allresults:
        # get environment name and algorithm configuration summary (should always exist)
        env_id = results["params"]["env_name"]
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
    for env_id in sorted(data.keys()):
        print("Creating plots for environment: {}".format(env_id))

        fig.clf()
        for i, xy in enumerate(data[env_id].keys(), 1):
            ax = fig.add_subplot(len(xys), 1, i)
            x_label = xy.split(":")[0]
            y_label = xy.split(":")[1]
            for config in sorted(data[env_id][xy].keys()):
                xs, ys = zip(*data[env_id][xy][config])
                xs, ys = pad(xs), pad(ys)
                assert xs.shape == ys.shape
                ax.plot(xs[0], np.nanmedian(ys, axis=0), label=config)
                ax.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()
                # fig.ylim(0, 1)
        fig.suptitle(env_id)
        save_path = os.path.join(target_dir, "fig_{}.png".format(env_id))
        print("Saving image to " + save_path)
        plt.savefig(save_path)


def plot(dirs, xys, save_path, smooth, **kwargs):
    results = load_results(dirs)
    # get directory to save results
    target_dir = save_path if save_path else dirs[0]
    plot_results(results, xys, target_dir, smooth)


if __name__ == "__main__":
    from yw.util.cmd_util import ArgParser

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
    ap.parse(sys.argv)
    plot(**ap.get_dict())
