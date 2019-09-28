import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from td3fd import logger
from td3fd.gail import config
from td3fd.util.cmd_util import ArgParser
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def train(root_dir, params):

    policy = config.configure_gail(params=params)
    rollout_worker = config.config_rollout(params=params, policy=policy)
    evaluator = config.config_evaluator(params=params, policy=policy)

    save_interval = 50

    # Setup paths
    # rl policies (cannot restart training)
    policy_save_path = os.path.join(root_dir, "gail")
    os.makedirs(policy_save_path, exist_ok=True)
    latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
    periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")

    # Adding demonstration data to the demonstration buffer
    demo_file = os.path.join(root_dir, "demo_data.npz")
    assert os.path.isfile(demo_file), "demonstration training set does not exist"
    policy.init_demo_buffer(demo_file)

    for epoch in range(policy.num_epochs):
        # Train
        # train generator and value function
        rollout_worker.clear_history()
        for _ in range(policy.policy_step):
            episode = rollout_worker.generate_rollouts()
            policy.clear_buffer()
            policy.store_episode(episode)
            policy.train_policy()
        # train discriminator
        policy.train_disc()
        # Evaluate
        evaluator.clear_history()
        evaluator.generate_rollouts()

        if epoch % 5 == 4:
            logger.record_tabular("epoch", int((epoch + 1) / 5))
            for key, val in evaluator.logs("test"):
                logger.record_tabular(key, val)
            for key, val in rollout_worker.logs("train"):
                logger.record_tabular(key, val)
            for key, val in policy.logs():
                logger.record_tabular(key, val)
            logger.dump_tabular()

        # save the policy
        save_msg = ""
        success_rate = evaluator.current_success_rate()
        logger.info("Current success rate: {}".format(success_rate))
        if save_interval > 0 and epoch % save_interval == (save_interval - 1):
            policy_path = periodic_policy_path.format(epoch)
            policy.save_policy(policy_path)
            save_msg += "periodic, "
        policy.save_policy(latest_policy_path)
        save_msg += "latest"
        logger.info("Saving", save_msg, "policy.")


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
    param_file = os.path.join(root_dir, "copied_params.json")
    if os.path.isfile(param_file):
        with open(param_file, "r") as f:
            params = json.load(f)
        config.check_params(params)
    else:
        logger.warn("WARNING: params.json not found! using the default parameters.")
        params = config.DEFAULT_PARAMS.copy()
    comp_param_file = os.path.join(root_dir, "params.json")
    with open(comp_param_file, "w") as f:
        json.dump(params, f)

    # reset default graph (must be called before setting seed)
    tf.reset_default_graph()
    # seed everything.
    set_global_seeds(params["seed"])
    # get a new default session for the current default graph
    tf.InteractiveSession()

    # Prepare parameters for training
    params = config.add_env_params(params=params)

    # Launch the training script
    train(root_dir=root_dir, params=params)

    tf.get_default_session().close()


if __name__ == "__main__":

    ap = ArgParser()
    # logging and saving path
    ap.parser.add_argument("--root_dir", help="directory to launching process", type=str, default=None)

    ap.parse(sys.argv)
    main(**ap.get_dict())
