import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from rlfd import config, logger
from rlfd.td3 import train as td3_train
from rlfd.mage import train as mage_train

try:
  from mpi4py import MPI
except ImportError:
  MPI = None


def main(root_dir, **kwargs):
  # allow calling this script using MPI to launch multiple training processes,
  # in which case only 1 process should print to stdout
  if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
    logger.configure(dir=root_dir,
                     format_strs=["stdout", "log", "csv"],
                     log_suffix="")
  else:
    logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
  assert logger.get_dir() is not None

  # Get default params from config and update params.
  param_file = os.path.join(root_dir, "params.json")
  assert os.path.isfile(param_file), param_file
  with open(param_file, "r") as f:
    params = json.load(f)

  # Launch the training script
  if params["alg"] == "td3":
    td3_train.train(root_dir=root_dir, params=params)
  elif params["alg"] == "mage":
    mage_train.train(root_dir=root_dir, params=params)
  else:
    assert False, "unknown algorithm"


if __name__ == "__main__":

  ap = ArgParser()
  # logging and saving path
  ap.parser.add_argument("--root_dir",
                         help="directory to launching process",
                         type=str)

  ap.parse(sys.argv)
  main(**ap.get_dict())
