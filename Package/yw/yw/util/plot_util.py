import os
import os.path as osp

import glob2
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
    load summaries of runs from a list of directories (including subdirectories)

    Arguments:

    Returns:
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

def plot_results(allresults, dir, smooth=False):
    data = {}
    for results in allresults:
        config = results["params"]["config"]
        # we do not need to plot if the result has demonstration training only
        if config == "":
            continue
        epoch = results["progress"]["epoch"]
        env_id = results["params"]["env_name"]
        env_id = env_id.replace("Dense", "")
        # currently we only plot the success_rate
        success_rate = results["progress"]["test/success_rate"]

        # Process and smooth data.
        assert success_rate.shape == epoch.shape
        x = epoch
        y = success_rate
        if smooth:
            x, y = smooth_reward_curve(epoch, success_rate)
        assert x.shape == y.shape

        if env_id not in data:
            data[env_id] = {}
        if config not in data[env_id]:
            data[env_id][config] = []
        data[env_id][config].append((x, y))

    # Plot data.
    for env_id in sorted(data.keys()):
        print("exporting {}".format(env_id))
        plt.clf()

        for config in sorted(data[env_id].keys()):
            xs, ys = zip(*data[env_id][config])
            xs, ys = pad(xs), pad(ys)
            assert xs.shape == ys.shape
            plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config)
            plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
        plt.title(env_id)
        plt.xlabel("Epoch")
        plt.ylabel("Median Success Rate")
        plt.legend()
        plt.ylim(0,1)
        save_path = os.path.join(dir, "fig_{}.png".format(env_id))
        plt.savefig(save_path)
        print("Save image to "+save_path)