import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from td3fd import config, logger
from td3fd.memory import iterbatches
from td3fd.td3 import config as ddpg_config
from td3fd.util.cmd_util import ArgParser
from td3fd.util.util import set_global_seeds

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def train(root_dir, params):

    # seed everything.
    set_global_seeds(params["seed"])

    config.add_env_params(params=params)

    policy = ddpg_config.configure_ddpg(params=params)
    memory = config.config_memory(params=params)
    rollout_worker = config.config_rollout(params=params, policy=policy)
    evaluator = config.config_evaluator(params=params, policy=policy)

    # Adding demonstration data to the demonstration buffer
    demo_memory = None
    if policy.demo_strategy != "none" or policy.sample_demo_buffer:
        demo_file = os.path.join(root_dir, "demo_data.npz")
        assert os.path.isfile(demo_file), "demonstration training set does not exist"
        demo_memory = config.config_memory(params=params)
        demo_memory.load_from_file(demo_file)

    # Training parameter
    save_interval = 4
    shaping_num_epochs = params["shaping"]["num_epochs"]
    shaping_batch_size = params["shaping"]["batch_size"]
    num_epochs = policy.num_epochs
    num_batches = policy.num_batches
    num_cycles = policy.num_cycles
    batch_size = policy.batch_size

    # Setup paths
    policy_save_path = os.path.join(root_dir, "policies")
    os.makedirs(policy_save_path, exist_ok=True)
    latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
    periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")

    # TODO: Incremental learning Option 1: dagger like method
    # if policy.demo_strategy in ["nf", "gan"]:
    #     with open("demo_policy.pkl", "rb") as f:
    #         demo_policy = pickle.load(f)

    # Train shaping potential
    shaping = None
    if policy.demo_strategy in ["nf", "gan"]:
        shaping = ddpg_config.configure_shaping(params)
        logger.info("Training the reward shaping potential.")
        for epoch in range(shaping_num_epochs):
            demo_data = demo_memory.sample()
            for (o, g, u) in iterbatches(
                (demo_data["o"], demo_data["g"], demo_data["u"]), batch_size=shaping_batch_size
            ):
                batch = {"o": o, "g": g, "u": u}
                d_loss, g_loss = shaping.train(batch)
                if epoch % (shaping_num_epochs / 100) == (shaping_num_epochs / 100 - 1):
                    logger.info("epoch: {} demo shaping loss: {}".format(epoch, d_loss))

    # Generate random experiences before training
    for _ in range(10000):
        episode = rollout_worker.generate_rollouts(random=True)
        memory.store_episode(episode)

    # Train the rl agent
    for epoch in range(num_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(num_cycles):
            episode = rollout_worker.generate_rollouts()
            memory.store_episode(episode)
            for _ in range(num_batches):
                batch = memory.sample(batch_size)
                policy.train(batch)
            policy.update_target_net()
        # test
        evaluator.clear_history()
        episode = evaluator.generate_rollouts()

        # TODO: Incremental learning
        # if policy.demo_strategy in ["nf", "gan"]:
        # # Option 1: adding more experience (with correction) to the demonstration buffer
        # o = episode["o"][:, :-1, ...].reshape(-1, *policy.dimo)
        # g = episode["g"].reshape(-1, *policy.dimg)
        # u = demo_policy.get_actions(o, g, compute_q=False)
        # u = u.reshape(episode["u"].shape)
        # episode["u"] = u
        # policy.add_to_demo_buffer(episode)
        # for epoch in range(shaping_num_epochs):
        #     loss = policy.train_shaping()
        #     if epoch % (shaping_num_epochs / 100) == (shaping_num_epochs / 100 - 1):
        #         logger.info("epoch: {} demo shaping loss: {}".format(epoch, loss))
        #         policy.evaluate_shaping()
        # # Option 2: train gan discriminator using fake data from generator
        # if epoch % 10 == 0:
        #     for _ in range(2):
        #         evaluator.clear_history()
        #         episode = evaluator.generate_rollouts()
        #         loss = policy.train_shaping_policy(episode)
        #         logger.info("epoch: {} demo shaping loss: {}".format(epoch, loss))
        #         policy.evaluate_shaping()

        # log
        logger.record_tabular("epoch", epoch)
        for key, val in evaluator.logs("test"):
            logger.record_tabular(key, val)
        for key, val in rollout_worker.logs("train"):
            logger.record_tabular(key, val)
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
