import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from rlfd import config, logger
from rlfd.td3 import train as td3_train
from rlfd.sac import train as sac_train
from rlfd.mage import train as mage_train


def main(config):

  root_dir = os.path.join(
      config["root_dir"], "config_" + config["config"],
      *[x + "_" + str(config[x]) for x in config["search_params_list"]])

  logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
  assert logger.get_dir() is not None

  # Limit gpu memory growth for tensorflow
  physical_gpus = tf.config.experimental.list_physical_devices("GPU")
  try:
    for gpu in physical_gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print("Found", len(physical_gpus), "physical GPUs", len(logical_gpus),
          "logical GPUs")
  except RuntimeError as e:
    print(e)  # Memory growth must be set before GPUs have been initialized

  # Get default params from config and update params.
  param_file = os.path.join(root_dir, "params.json")
  assert os.path.isfile(param_file), param_file
  with open(param_file, "r") as f:
    params = json.load(f)

  # Launch the training script
  if params["algo"] == "td3":
    td3_train.train(root_dir=root_dir, params=params)
  elif params["algo"] == "sac":
    sac_train.train(root_dir=root_dir, params=params)
  elif params["algo"] == "mage":
    mage_train.train(root_dir=root_dir, params=params)
  else:
    raise ValueError("Unknown algorithm: {}".format(params["algo"]))