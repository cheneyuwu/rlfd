import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from td3fd import config, logger
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


DEFAULT_PARAMS = {
    "seed": 0,
    "num_eps": 40,
    "fix_T": True,
    "max_concurrency": 10,
    "demo": {"random_eps": 0.0, "noise_eps": 0.1, "render": False},
    "filename": "demo_data.npz",
}


def main(policy, root_dir, **kwargs):
    """Generate demo from policy file
    """
    assert root_dir is not None, "must provide the directory to store into"
    assert policy is not None, "must provide the policy"

    # Setup
    logger.configure()
    assert logger.get_dir() is not None
    rank = MPI.COMM_WORLD.Get_rank() if MPI != None else 0

    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "demo_config.json")
    if os.path.isfile(param_file):
        with open(param_file, "r") as f:
            params = json.load(f)
    else:
        logger.warn("WARNING: demo_config.json not found! using the default parameters.")
        params = DEFAULT_PARAMS.copy()
        if rank == 0:
            param_file = os.path.join(root_dir, "demo_config.json")
            with open(param_file, "w") as f:
                json.dump(params, f)

    # Set random seed for the current graph
    set_global_seeds(params["seed"])
    # get a default session for the current graph
    tf.InteractiveSession()

    # Load policy.
    with open(policy, "rb") as f:
        policy = pickle.load(f)

    # Extract environment construction information
    params["env_name"] = policy.info["env_name"].replace("Dense", "")  # the reward should be sparse
    params["r_scale"] = policy.info["r_scale"]
    params["r_shift"] = policy.info["r_shift"]
    params["eps_length"] = policy.info["eps_length"] if policy.info["eps_length"] != 0 else policy.T
    params["env_args"] = policy.info["env_args"]
    params["gamma"] = policy.info["gamma"]
    if params["fix_T"]:
        params["demo"]["num_episodes"] = np.minimum(params["num_eps"], params["max_concurrency"])
    else:
        params["demo"]["num_episodes"] = params["num_eps"]
    params = config.add_env_params(params=params)
    demo = config.config_demo(params=params, policy=policy)

    # Generate demonstration data
    if params["fix_T"]:
        episode = None
        num_eps_togo = params["num_eps"]
        demo.clear_history()
        while num_eps_togo > 0:
            eps = demo.generate_rollouts()
            num_eps_to_store = np.minimum(num_eps_togo, params["max_concurrency"])
            if episode == None:
                episode = {k: eps[k][:num_eps_to_store, ...] for k in eps.keys()}
            else:
                episode = {k: np.concatenate((episode[k], eps[k][:num_eps_to_store, ...]), axis=0) for k in eps.keys()}
            num_eps_togo -= num_eps_to_store

        assert all([episode[k].shape[0] == params["num_eps"] for k in episode.keys()])
    else:
        episode = demo.generate_rollouts()
    # record logs
    for key, val in demo.logs("test"):
        logger.record_tabular(key, np.mean(val))
    if rank == 0:
        logger.dump_tabular()

    # Store demonstration data (only the main thread)
    if rank == 0:
        os.makedirs(root_dir, exist_ok=True)
        file_name = os.path.join(root_dir, params["filename"])
        # array(batch_size x (T or T+1) x dim_key), we only need the first one!
        np.savez_compressed(file_name, **episode)  # save the file
        logger.info("Demo file has been stored into {}.".format(file_name))

    tf.compat.v1.get_default_session().close()


if __name__ == "__main__":

    from td3fd.util.cmd_util import ArgParser

    ap = ArgParser()
    ap.parser.add_argument("--root_dir", help="policy store directory", type=str, default=None)
    ap.parser.add_argument("--policy", help="input policy file for training", type=str, default=None)
    ap.parse(sys.argv)

    main(**ap.get_dict())
