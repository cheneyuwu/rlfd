import os
import sys

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import json
import pickle
import numpy as np
import tensorflow as tf

from yw.tool import logger
from yw.ddpg_main import config
from yw.util.util import set_global_seeds
from yw.util.mpi_util import mpi_average

from itertools import combinations  # for dimension projection


def train(
    comm,
    root_dir,
    save_interval,
    policy,
    rollout_worker,
    evaluator,
    n_epochs,
    n_batches,
    n_cycles,
    shaping_policy,
    **kwargs
):
    logger.info(
        "Training the RL agent with n_epochs: {:d}, n_cycles: {:d}, n_batches: {:d}.".format(
            n_epochs, n_cycles, n_batches
        )
    )

    rank = comm.Get_rank() if comm != None else 0

    assert root_dir != None
    # rl
    policy_save_path = root_dir + "/rl/"
    os.makedirs(policy_save_path, exist_ok=True)
    latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
    best_policy_path = os.path.join(policy_save_path, "policy_best.pkl")
    periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")
    # shaping
    shaping_save_path = root_dir + "/shaping/"
    os.makedirs(shaping_save_path, exist_ok=True)
    latest_shaping_path = os.path.join(shaping_save_path, "shaping_latest.ckpt")
    # queries
    query_shaping_save_path = root_dir + "/query_shaping/"
    os.makedirs(query_shaping_save_path, exist_ok=True)
    query_policy_save_path = root_dir + "/query_policy/"
    os.makedirs(query_policy_save_path, exist_ok=True)

    # Adding demonstration data to the demonstration buffer
    if policy.demo_strategy != "none":
        demo_file = os.path.join(root_dir, "demo_data.npz")
        assert os.path.isfile(demo_file), "demonstration training set does not exist"
        update_stats = policy.demo_strategy == "bc" or policy.demo_strategy == "rbmaf" or policy.demo_strategy == "rb"
        policy.init_demo_buffer(demo_file, update_stats=update_stats)

    # Pre-Training a potential function
    if policy.demo_strategy == "maf" or policy.demo_strategy == "rbmaf":
        if not shaping_policy:
            logger.info("Training the policy for reward shaping.")
            num_epoch = 1000
            for epoch in range(num_epoch):
                loss = policy.train_shaping()

                if rank == 0 and epoch % (num_epoch / 10) == (num_epoch / 10 - 1):
                    logger.info("epoch: {} demo shaping loss: {}".format(epoch, loss))

                if rank == 0 and epoch % 100 == 0:
                    logger.info("Saving latest policy to {}.".format(latest_shaping_path))
                    policy.save_shaping_weights(latest_shaping_path)
            # uncomment to save trained potential functions
            # if rank == 0:
            #     logger.info("Saving latest policy to {}.".format(latest_shaping_path))
            #     policy.save_shaping_weights(latest_shaping_path)
        else:
            logger.info("Using the provided policy weights: {}".format(latest_shaping_path))
            policy.load_shaping_weights(latest_shaping_path)
            # query
            # dims = list(range(policy.dimo + policy.dimg))
            # for dim1, dim2 in combinations(dims, 2):
            #     policy.query_potential(
            #         dim1=dim1,
            #         dim2=dim2,
            #         filename=os.path.join(
            #             query_shaping_save_path, "dim_{}_{}_{:04d}.jpg".format(dim1, dim2, epoch)
            #         ),
            #     )

    best_success_rate = -1

    for epoch in range(n_epochs):
        logger.debug("train_ddpg_main.train_reinforce -> epoch: {}".format(epoch))

        # Store anything we need into a numpyz file.
        policy.query_policy(
            filename=os.path.join(query_policy_save_path, "query_{:03d}.npz".format(epoch)),  # comment to show plot
            fid=3,
        )

        # Train
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            logger.debug("train_ddpg_main.train_reinforce -> cycle: {}".format(cycle))
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for batch in range(n_batches):
                logger.debug("train_ddpg_main.train_reinforce -> batch: {}".format(batch))
                policy.train()
            policy.check_train()
            policy.update_target_net()

        # Test
        logger.debug("train_ddpg_main.train_reinforce -> Testing.")
        evaluator.clear_history()
        evaluator.generate_rollouts()

        # Log
        logger.record_tabular("epoch", epoch)
        for key, val in evaluator.logs("test"):
            logger.record_tabular(key, mpi_average(val, comm=comm))
        for key, val in rollout_worker.logs("train"):
            logger.record_tabular(key, mpi_average(val, comm=comm))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val, comm=comm))

        if rank == 0:
            logger.dump_tabular()

        # Save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate(), comm=comm)
        if rank == 0 and success_rate >= best_success_rate:
            best_success_rate = success_rate
            logger.info("New best success rate: {}.".format(best_success_rate))
            logger.info("Saving policy to {}.".format(best_policy_path))
            evaluator.save_policy(best_policy_path)

        if rank == 0 and save_interval > 0 and epoch % save_interval == 0:
            policy_path = periodic_policy_path.format(epoch)
            logger.info("Saving periodic policy to {}.".format(policy_path))
            evaluator.save_policy(policy_path)

        if rank == 0:
            logger.info("Saving latest policy to {}.".format(latest_policy_path))
            evaluator.save_policy(latest_policy_path)

        # Make sure that different threads have different seeds
        if comm != None:
            local_uniform = np.random.uniform(size=(1,))
            root_uniform = local_uniform.copy()
            comm.Bcast(root_uniform, root=0)
            if rank != 0:
                assert local_uniform[0] != root_uniform[0]


def main(root_dir, comm=None, **kwargs):

    assert root_dir is not None, "provide root directory for saving training data"

    # Consider rank as pid.
    if comm is None:
        comm = MPI.COMM_WORLD if MPI is not None else None
    num_cpu = comm.Get_size() if comm is not None else 1
    rank = comm.Get_rank() if comm is not None else 0

    if MPI.COMM_WORLD.Get_rank() == 0 and rank == 0:
        logger.configure(dir=root_dir, format_strs=["stdout", "log", "csv"], log_suffix="")
    elif rank == 0:
        logger.configure(dir=root_dir, format_strs=["log", "csv"], log_suffix="")
    else:
        logger.configure(format_strs=["log"])
    assert logger.get_dir() is not None

    log_level = 2  # 1 for debugging, 2 for info
    logger.set_level(log_level)
    logger.info("Launching the training process with {} cpu core(s).".format(num_cpu))
    logger.info("Setting log level to {}.".format(log_level))

    # Get default params from config and update params.
    param_file = os.path.join(root_dir, "copied_params.json")
    if os.path.isfile(param_file):
        with open(param_file, "r") as f:
            params = json.load(f)
    else:
        logger.warn("WARNING: params.json not found! using the default parameters.")
        params = config.DEFAULT_PARAMS.copy()
    if rank == 0:
        comp_param_file = os.path.join(root_dir, "params.json")
        # if params["config"] == "default": # Modify the config name here!
        #     params["config"] = "RL+Demo:" + params["ddpg"]["demo_strategy"]
        with open(comp_param_file, "w") as f:
            json.dump(params, f)

    # Reset default graph (must be called before setting seed)
    tf.reset_default_graph()
    # seed everything.
    set_global_seeds(params["seed"])
    # get a new default session for the current default graph
    tf.InteractiveSession()

    # Prepare parameters for training
    params = config.add_env_params(params=params)

    # Configure and train rl agent
    policy = config.configure_ddpg(params=params, comm=comm)
    rollout_worker = config.config_rollout(params=params, policy=policy)
    evaluator = config.config_evaluator(params=params, policy=policy)

    # Launch the training script
    train(
        comm=comm,
        root_dir=root_dir,
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        **params["train"]
    )

    # Close the default session to prevent memory leaking
    tf.get_default_session().close()


if __name__ == "__main__":

    from yw.util.cmd_util import ArgParser

    ap = ArgParser()
    # logging and saving path
    ap.parser.add_argument("--root_dir", help="directory to launching process", type=str, default=None)

    ap.parse(sys.argv)
    main(**ap.get_dict())
