import sys
import os
import pickle

import numpy as np
import tensorflow as tf

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# DDPG Package import
from yw.tool import logger
from yw.ddpg_main import config
from yw.util.util import set_global_seeds


def main(policy_file, store_dir, seed, num_eps, **kwargs):
    """Generate demo from policy file
    """
    assert store_dir is not None, "must provide the directory to store into"
    assert policy_file is not None, "must provide the policy_file"

    # Setup
    logger.configure()
    assert logger.get_dir() is not None
    rank = MPI.COMM_WORLD.Get_rank() if MPI != None else 0

    # reset default graph every time this function is called.
    tf.reset_default_graph()
    # Set random seed for the current graph
    set_global_seeds(seed)
    # get a default session for the current graph
    tf.InteractiveSession()

    # Load policy.
    with open(policy_file, "rb") as f:
        policy = pickle.load(f)

    # Extract environment construction information
    env_name = policy.info["env_name"]
    T = policy.info["eps_length"] if policy.info["eps_length"] != 0 else policy.T
    max_concurrency = 10  # avoid too many envs

    # Prepare params.
    params = {}
    params["env_name"] = env_name
    params["r_scale"] = policy.info["r_scale"]
    params["r_shift"] = policy.info["r_shift"]
    params["eps_length"] = T
    params["env_args"] = policy.info["env_args"]
    params["rollout_batch_size"] = np.minimum(num_eps, max_concurrency)
    params["seed"] = seed
    params["render"] = 0
    params = config.add_env_params(params=params)
    demo = config.config_demo(params=params, policy=policy)

    # Run evaluation.
    episode = None
    num_eps_togo = num_eps
    demo.clear_history()
    while num_eps_togo > 0:
        eps = demo.generate_rollouts()
        num_eps_to_store = np.minimum(num_eps_togo, max_concurrency)
        if episode == None:
            episode = {k: eps[k][:num_eps_to_store, ...] for k in eps.keys()}
        else:
            episode = {k: np.concatenate((episode[k], eps[k][:num_eps_to_store, ...]), axis=0) for k in eps.keys()}
        num_eps_togo -= num_eps_to_store

    assert all([episode[k].shape[0] == num_eps for k in episode.keys()])

    # Add expected Q value
    # exp_q = np.empty(episode["r"].shape)
    # exp_q[:, -1, :] = episode["r"][:, -1, :] / (1 - policy.gamma)
    # for i in range(params["eps_length"] - 1):
    #     exp_q[:, -2 - i, :] = policy.gamma * exp_q[:, -1 - i, :] + episode["r"][:, -2 - i, :]
    # episode["q"] = exp_q

    # record logs
    for key, val in demo.logs("test"):
        logger.record_tabular(key, np.mean(val))
    if rank == 0:
        logger.dump_tabular()

    # store demonstration data (only the main thread)
    if rank == 0:
        os.makedirs(store_dir, exist_ok=True)
        file_name = os.path.join(store_dir, "demo_data.npz")
        # array(batch_size x (T or T+1) x dim_key), we only need the first one!
        np.savez_compressed(file_name, **episode)  # save the file
        logger.info("Demo file has been stored into {}.".format(file_name))

    # Close the default session to prevent memory leaking
    tf.get_default_session().close()


from yw.util.cmd_util import ArgParser

ap = ArgParser()
ap.parser.add_argument("--policy_file", help="input policy for training", type=str, default=None)
ap.parser.add_argument("--store_dir", help="policy store directory", type=str, default=None)
ap.parser.add_argument("--seed", help="RNG seed", type=int, default=0)
ap.parser.add_argument("--num_eps", help="number of iterations or episodes", type=int, default=1000)

if __name__ == "__main__":
    ap.parse(sys.argv)

    main(**ap.get_dict())
