import sys
import os
import numpy as np
import pickle

from yw.ddpg_main import config


from yw.util.mpi_util import set_global_seeds
from yw.util.lit_util import toy_regression_data

# DDPG Package import
from yw.tool import logger


def generate_demo_data(policy_file, store_dir, seed, num_itr, shuffle, render, entire_eps, **kwargs):
    """Generate demo from policy file
    """
    assert policy_file is not None, "Must provide the policy_file!"
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, "rb") as f:
        policy = pickle.load(f)

    # Extract environment construction information
    env_name = policy.info["env_name"]
    r_scale = policy.info["r_scale"]
    r_shift = policy.info["r_shift"]
    eps_length = policy.info["eps_length"]
    T = policy.T

    max_concurrency = 10
    num_eps = int(np.ceil(num_itr/T)) if shuffle and entire_eps else num_itr

    # Prepare params.
    params = {}
    params["env_name"] = env_name
    params["r_scale"] = r_scale
    params["r_shift"] = r_shift
    params["eps_length"] = eps_length
    params["rank_seed"] = seed
    params["render"] = render
    params["rollout_batch_size"] = np.minimum(num_eps, max_concurrency)
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
            episode = {k: eps[k][:num_eps_to_store,...] for k in eps.keys()}
        else:
            episode = {k:np.concatenate((episode[k], eps[k][:num_eps_to_store,...]), axis=0) for k in eps.keys()}
        num_eps_togo -= num_eps_to_store

    assert all([episode[k].shape[0] == num_eps for k in episode.keys()])

    # Add expected Q value
    exp_q = np.empty(episode["r"].shape)
    exp_q[:, -1, :] = episode["r"][:, -1, :] / (1 - policy.gamma)
    for i in range(policy.T - 1):
        exp_q[:, -2 - i, :] = policy.gamma * exp_q[:, -1 - i, :] + episode["r"][:, -2 - i, :]
    episode["q"] = exp_q

    # value debug print out the q value
    # logger.info("The expected q values are: {}".format(episode["q"]))

    # record logs
    for key, val in demo.logs("test"):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()

    # array(batch_size x (T or T+1) x dim_key), we only need the first one!
    result = {}

    if not shuffle:
        result = episode
    else:
        if entire_eps:
            result["o"] = episode["o"][:, :-1, ...] # observation is T+1
            result["g"] = episode["g"][:, :, ...]
            result["u"] = episode["u"][:, :, ...]
            result["q"] = episode["q"][:, :, ...]
        else:
            result["o"] = episode["o"][:, :1, ...]
            result["g"] = episode["g"][:, :1, ...]
            result["u"] = episode["u"][:, :1, ...]
            result["q"] = episode["q"][:, :1, ...]

        assert result["o"].shape[0] * result["o"].shape[1] >= num_itr, "No enough data!"
        result = {key: result[key].reshape((-1,) + result[key].shape[2:])[:num_itr] for key in result.keys()}

    # store demonstration data
    os.makedirs(store_dir, exist_ok=True)
    file_name = store_dir + "/" + env_name
    file_name += ".npz"

    logger.debug(
        "generate_demo.generate_demo_data -> data shape is: {}".format({k: result[k].shape for k in result.keys()})
    )

    np.savez_compressed(file_name, **result)  # save the file
    logger.info("Demo file has been stored into {}.".format(file_name))


def main(loglevel, **kwargs):
    logger.set_level(loglevel)
    generate_demo_data(**kwargs)


if __name__ == "__main__":
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()
    ap.parser.add_argument("--policy_file", help="input policy for training", type=str, default=None)
    ap.parser.add_argument("--loglevel", help="log level", type=int, default=2)
    ap.parser.add_argument("--store_dir", help="policy store directory", type=str, default=os.getenv("PROJECT") + "/Temp/generate_demo/fake_data.npz")
    ap.parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    ap.parser.add_argument("--num_itr", help="number of iterations or episodes", type=int, default=1000)
    ap.parser.add_argument("--shuffle", help="whether or not to shuffle and reshape the data", type=int, default=1)
    ap.parser.add_argument("--entire_eps", help="store entire episodes", type=int, default=1)
    ap.parser.add_argument("--render", help="render", type=int, default=0)
    ap.parse(sys.argv)

    print("Launching the training process.")
    main(**ap.get_dict())

