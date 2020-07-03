import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from rlfd import config, logger
from rlfd.td3 import train as td3_train
from rlfd.sac import train as sac_train
from rlfd.sac_vf import train as sac_vf_train
from rlfd.mage import train as mage_train

ALGORITHMS = {
    "td3": td3_train.train,
    "sac": sac_train.train,
    "sac-vf": sac_vf_train.train,
    "mage": mage_train.train
}


def main(config):
  # Setup Paths
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
  # Load parameters.
  param_file = os.path.join(root_dir, "params.json")
  assert os.path.isfile(param_file), param_file
  with open(param_file, "r") as f:
    params = json.load(f)
  # Launch the corresponding training script.
  ALGORITHMS[params["algo"]](root_dir=root_dir, params=params)