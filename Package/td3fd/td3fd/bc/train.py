import json
import os
import pickle
import sys

import numpy as np

from td3fd import config, logger
from td3fd.memory import iterbatches
from td3fd.bc import config as bc_config
from td3fd.util.cmd_util import ArgParser
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def train(root_dir, params):

    # Check parameters
    config.check_params(params, bc_config.default_params)

    # Training parameter
    save_interval = 10
    num_epochs = params["bc"]["num_epochs"]
    batch_size = params["bc"]["batch_size"]

    # Seed everything.
    set_global_seeds(params["seed"])

    # Setup paths
    policy_save_path = os.path.join(root_dir, "policies")
    os.makedirs(policy_save_path, exist_ok=True)
    latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
    periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")

    # Construct ...
    config.add_env_params(params=params)
    policy = bc_config.configure_bc(params=params)
    evaluator = config.config_evaluator(params=params, policy=policy)
    # adding demonstration data to the demonstration buffer
    demo_file = os.path.join(root_dir, "demo_data.npz")
    assert os.path.isfile(demo_file), "demonstration training set does not exist"
    demo_memory = config.config_memory(params=params)
    demo_memory.load_from_file(demo_file)

    # Train the policy with behaviorial cloning
    demo_data = demo_memory.sample()
    policy.update_stats(demo_data) # Note: no need for image based envs
    for epoch in range(num_epochs):
        losses = np.empty(0)
        for (o, g, u) in iterbatches(
            (demo_data["o"], demo_data["g"], demo_data["u"]), batch_size=batch_size
        ):
            batch = {"o": o, "g": g, "u": u}
            loss = policy.train(batch)
            losses = np.append(losses, loss.cpu().data.numpy())
        if epoch % (num_epochs / 100) == (num_epochs / 100 - 1):
            logger.info("epoch: {} loss: {}".format(epoch, np.mean(losses)))

    # Train rl policy
    for epoch in range(num_epochs):

        # test
        evaluator.clear_history()
        episode = evaluator.generate_rollouts()

        # log
        logger.record_tabular("epoch", epoch)
        for key, val in evaluator.logs("test"):
            logger.record_tabular(key, val)
        # for key, val in rollout_worker.logs("train"):
        #     logger.record_tabular(key, val)
        for key, val in policy.logs():
            logger.record_tabular(key, val)
        logger.dump_tabular()

        # save the policy
        # success_rate = evaluator.current_success_rate()
        # logger.info("Current success rate: {}".format(success_rate))
        save_msg = ""
        if save_interval > 0 and epoch % save_interval == (save_interval - 1):
            policy_path = periodic_policy_path.format(epoch)
            policy.save(policy_path)
            save_msg += "periodic, "
        policy.save(latest_policy_path)
        save_msg += "latest"
        logger.info("Saving", save_msg, "policy.")


def main(root_dir, **kwargs):

    # allow calling this script using MPI to launch multiple training processes, in which case only 1 process should
    # print to stdout
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=root_dir, format_strs=["stdout", "log", "csv"], log_suffix="")
    else:
        logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
    assert logger.get_dir() is not None

    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "copied_params.json")
    if os.path.isfile(param_file):
        with open(param_file, "r") as f:
            params = json.load(f)
        config.check_params(params, ddpg_config.default_params)
    else:
        logger.warn("WARNING: params.json not found! using the default parameters.")
        params = ddpg_config.default_params.copy()
    comp_param_file = os.path.join(root_dir, "params.json")
    with open(comp_param_file, "w") as f:
        json.dump(params, f)

    # Launch the training script
    train(root_dir=root_dir, params=params)


if __name__ == "__main__":

    ap = ArgParser()
    # logging and saving path
    ap.parser.add_argument("--root_dir", help="directory to launching process", type=str, default=None)

    ap.parse(sys.argv)
    main(**ap.get_dict())
