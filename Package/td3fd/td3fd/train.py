import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from td3fd import config, logger

# pytorch
from td3fd.td3.train import train as ddpg_torch_train
from td3fd.bc.train import train as bc_torch_train
from td3fd.rlkit_sac.train import main as rlkit_sac_torch_train
from td3fd.rlkit_td3.train import main as rlkit_td3_torch_train

# tensorflow
from td3fd.ddpg.train import train as ddpg_tf_train

# from td3fd.gail.train import train as gail_train

from td3fd.util.cmd_util import ArgParser
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def main(root_dir, **kwargs):

    assert root_dir is not None, "provide root directory for saving training data"
    # allow calling this script using MPI to launch multiple training processes, in which case only 1 process should
    # print to stdout
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=root_dir, format_strs=["stdout", "log", "csv"], log_suffix="")
    else:
        logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
    assert logger.get_dir() is not None

    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "params.json")
    assert os.path.isfile(param_file), param_file
    with open(param_file, "r") as f:
        params = json.load(f)

    # seed everything.
    set_global_seeds(params["seed"])
    # get a new default session for the current default graph
    tf.compat.v1.InteractiveSession()

    # Launch the training script
    if params["alg"] == "ddpg-torch":
        ddpg_torch_train(root_dir=root_dir, params=params)
    elif params["alg"] == "bc-torch":
        bc_torch_train(root_dir=root_dir, params=params)
    elif params["alg"] == "ddpg-tf":
        ddpg_tf_train(root_dir=root_dir, params=params)
    elif params["alg"] == "rlkit-sac":
        rlkit_sac_torch_train(root_dir=root_dir, params=params)
    elif params["alg"] == "rlkit-td3":
        rlkit_td3_torch_train(root_dir=root_dir, params=params)
    # elif "gail" in params.keys():
    #     gail_train(root_dir=root_dir, params=params)
    else:
        assert False

    tf.compat.v1.get_default_session().close()


if __name__ == "__main__":

    ap = ArgParser()
    # logging and saving path
    ap.parser.add_argument("--root_dir", help="directory to launching process", type=str, default=None)

    ap.parse(sys.argv)
    main(**ap.get_dict())
