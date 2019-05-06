import click
import os
import numpy as np
import pickle

from yw.ddpg_main import config


from yw.util.mpi_util import set_global_seeds
from yw.util.lit_util import toy_regression_data

# DDPG Package import
from yw.tool import logger


def generate_demo_data(policy_file, store_dir, seed, num_itr, render, entire_eps):
    """Generate demo from policy file
    """
    assert policy_file is not None, "Must provide the policy_file!"
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, "rb") as f:
        policy = pickle.load(f)

    # extract environment construction information
    env_name = policy.info["env_name"]
    r_scale = policy.info["r_scale"]
    r_shift = policy.info["r_shift"]
    eps_length = policy.info["eps_length"]
    T = policy.T

    # Prepare params.
    params = {}
    params["env_name"] = env_name
    params["r_scale"] = r_scale
    params["r_shift"] = r_shift
    params["eps_length"] = eps_length
    params["rank_seed"] = seed
    params["render"] = render
    params["rollout_batch_size"] = int(np.ceil(num_itr/T)) if entire_eps else num_itr
    params = config.add_env_params(params=params)
    demo = config.config_demo(params=params, policy=policy)

    # Run evaluation.
    demo.clear_history()
    episode = demo.generate_rollouts()

    # add expected Q value
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


@click.command()
@click.option("--policy_file", type=str, default=None, help="Input policy for training.")
@click.option("--loglevel", type=int, default=1, help="Set to 1 for debugging.")
@click.option("--store_dir", type=str, default=os.getenv("PROJECT") + "/Temp/Multigoal/demo/fake_data.npz", help="Log directory.")
@click.option("--seed", type=int, default=1)
@click.option("--num_itr", type=int, default=1000)
@click.option("--entire_eps", type=int, default=1, help="Whether or not to store the entire episode.")
@click.option("--render", type=int, default=0)
def main(loglevel, **kwargs):
    logger.set_level(loglevel)
    generate_demo_data(**kwargs)


if __name__ == "__main__":
    main()

