import glob
import json
import os
import os.path as osp
import sys
from datetime import datetime
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm
from tensorboard.backend.event_processing import event_accumulator

from rlfd.utils.cmd_util import ArgParser
from rlfd.utils.reader_util import load_csv

matplotlib.use("Agg")  # Can change to 'Agg' for non-interactive mode
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# set sizes
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 16
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize


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
      np.ones_like(y), np.ones(2 * k + 1), mode="same")
  return xsmoo, ysmoo


def convert_tensorboard_data_to_csv(path,
                                    source_dir="summaries",
                                    target_dir="csv_summaries"):
  summary_iterator = event_accumulator.EventAccumulator(
      osp.join(path, source_dir),
      size_guidance={
          "tensors": 0
      },
  ).Reload()
  tags = summary_iterator.Tags()
  for tag in tags["tensors"]:
    # This is hardcoded in the tensorboard output
    try:
      y_label, x_label = tag.split(" vs ")
    except ValueError:
      y_label = tag
      x_label = "Step"
    xy_data = np.array([(event.step, tf.make_ndarray(event.tensor_proto))
                        for event in summary_iterator.Tensors(tag)])
    df = pd.DataFrame({y_label: xy_data[:, 1], x_label: xy_data[:, 0]})
    tag = tag.replace("/", "_")
    os.makedirs(osp.join(path, target_dir), exist_ok=True)
    df.to_csv(osp.join(path, target_dir, tag))


def load_results(root_dir_or_dirs):
  """
  Load summaries of runs from a list of directories (including subdirectories)
  Looking for directories with both params.json and progress.csv.

  Arguments:

  Returns:
      allresults - list of dicts that contains "summaries" and "params".
  """
  if isinstance(root_dir_or_dirs, str):
    rootdirs = [osp.expanduser(root_dir_or_dirs)]
  else:
    rootdirs = [osp.expanduser(d) for d in root_dir_or_dirs]
  allresults = []
  for rootdir in rootdirs:
    assert osp.exists(rootdir), "%s doesn't exist" % rootdir
    for dirname, subdirs, files in os.walk(rootdir):
      if (all([file in files for file in ["params.json"]]) and
          "summaries" in subdirs):
        result = {"dirname": dirname}
        result["progress"] = dict()
        source_dir = "summaries"
        target_dir = "csv_summaries"
        if source_dir in subdirs:
          convert_tensorboard_data_to_csv(dirname, source_dir, target_dir)
          for csv in os.listdir(osp.join(dirname, target_dir)):
            progress = load_csv(osp.join(dirname, target_dir, csv))
            result["progress"][csv] = progress
        # load parameters
        paramsjson = osp.join(
            dirname, "params_renamed.json")  # search for the renamed file first
        if not osp.exists(paramsjson):
          paramsjson = osp.join(dirname, "params.json")
        with open(paramsjson, "r") as f:
          result["params"] = json.load(f)

        allresults.append(result)

  return allresults


def plot_results(allresults, xys, save_dir, save_name, smooth=False):

  # collect data: [environment][legend][configuration] -> data
  data = {}
  for results in allresults:
    # get environment name and algorithm configuration summary (should always exist)
    env_id = results["params"]["env_name"].replace("Dense", "")
    config = results["params"]["config"]

    for xy in xys:
      csv_name = xy.replace("/", "_")
      try:
        y_label, x_label = xy.split(" vs ")
      except ValueError:
        y_label = xy
        x_label = "Step"
      if csv_name not in results["progress"].keys():
        continue
      x = results["progress"][csv_name][x_label]
      y = results["progress"][csv_name][y_label]

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
  size = 5
  fig.set_size_inches(size * len(data.keys()), size * len(xys))
  fig.subplots_adjust(left=0.15,
                      right=0.9,
                      bottom=0.25,
                      top=0.95,
                      wspace=0.25,
                      hspace=0.25)
  fig.clf()

  for env_n, env_id in enumerate(sorted(data.keys())):
    print("Creating plots for environment: {}".format(env_id))
    for i, xy in enumerate(data[env_id].keys()):

      # RL  BC  RLBCINIT
      # colors = ["y", "c", "k"]
      colors = ["m", "b", "g", "r"]
      # colors = ["b", "r", "g", "k", "c", "m", "y"]
      markers = ["o", "v", "s", "d", "p", "h"]

      ax = fig.add_subplot(len(xys), len(data.keys()),
                           env_n * len(data.keys()) + i + 1)
      try:
        y_label, x_label = xy.split(" vs ")
      except ValueError:
        y_label = xy
        x_label = "Step"
      for j, config in enumerate(sorted(data[env_id][xy].keys())):
        xs, ys = zip(*data[env_id][xy][config])
        if config == "default":
          continue

        # CHANGE! either pad with nan or strip to the minimum length
        required_length = 0
        # xs, ys = pad(xs), pad(ys)
        xs, ys = strip(xs, required_length), strip(ys, required_length)
        assert xs.shape == ys.shape

        # in case you want to cut
        last_idx = None
        # last_idx = 400  # for gym environments
        # last_idx = 2500
        xs = xs[..., :last_idx]
        ys = ys[..., :last_idx]

        # select the first x since they ar all same
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
        ax.fill_between(x,
                        mean_y - 1.0 * stddev_y,
                        mean_y + 1.0 * stddev_y,
                        alpha=0.2,
                        color=colors[j % len(colors)])
        #
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
        # ax.legend(loc="upper left", frameon=False)

  # use fig level legend
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles,
             labels,
             loc="lower center",
             ncol=2,
             frameon=False,
             fontsize=6)

  date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
  save_name = "plot_{}_{}.jpg".format(save_name, date_time)
  save_path = osp.join(save_dir, save_name)
  print("Saving image to " + save_path)
  plt.savefig(save_path, dpi=200)
  plt.show()


def main(dirs, xys, save_dir=None, save_name="", smooth=False, **kwargs):
  results = load_results(reduce(lambda a, b: a + glob.glob(b), dirs, []))
  # get directory to save results
  save_dir = save_dir if save_dir else dirs[0]
  plot_results(results, xys, save_dir, save_name, smooth)


if __name__ == "__main__":

  from rlfd.utils.cmd_util import ArgParser

  exp_parser = ArgParser(allow_unknown_args=False)
  exp_parser.parser.add_argument(
      "-d",
      "--dirs",
      help="top level directory to store experiment results",
      nargs="*",
      type=str,
      default=[os.getcwd()])
  exp_parser.parser.add_argument(
      "-p",
      "--xys",
      help="top level directory to store experiment results",
      nargs="*",
      type=str,
      default=[
          "OnlineTesting/AverageReturn vs EnvironmentSteps",
          # "OfflineTesting/AverageReturn",
      ])
  exp_parser.parser.add_argument("-s",
                                 "--save_dir",
                                 help="top level directory to store plots",
                                 type=str,
                                 default=os.getcwd())
  exp_parser.parser.add_argument("-n",
                                 "--save_name",
                                 help="name to store plots",
                                 type=str,
                                 default="")
  exp_parser.parser.add_argument("--smooth",
                                 help="smooth the plot",
                                 type=bool,
                                 default=False)
  exp_parser.parse(sys.argv)
  main(**exp_parser.get_dict())
