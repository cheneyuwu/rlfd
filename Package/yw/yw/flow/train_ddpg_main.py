"""Active learning project

    This python script trains an agent from demonstration data and aims to out-perform expert demonstration though
    reinforcement learning and active learning.

    Steps:
        1. Train a Q value estimator through supervised learning from demonstration.
            Should allow user to use existing policy file.
            train_demo.py contains this part only.
        2. Train an agent based on RL (DDPG) to output an action for each input that contains a state and a goal.

"""

import os
import sys

from mpi4py import MPI

import json
import pickle
import numpy as np

from yw.tool import logger
from yw.ddpg_main import config
from yw.util.mpi_util import mpi_fork, mpi_average, set_global_seeds

def train_gp_q_estimator(save_path, save_interval, policy, n_epochs, demo_file, demo_test_file, **kwargs):

    # Consider rank as pid
    rank = MPI.COMM_WORLD.Get_rank()

    # Make directions for saving policies
    if save_path:
        policy_save_path = save_path + "/rl_demo_critic/"
        os.makedirs(policy_save_path, exist_ok=True)
        latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
        periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")

    # Fill in demonstration data
    policy.update_dataset(demo_file, demo_test_file)

    check_interval = n_epochs // 20 if n_epochs > 20 else 1
    for epoch in range(n_epochs):

        # train - This will go through the entire training set once.
        policy.train()

        # test
        if rank == 0 and epoch % check_interval == 0:
            loss = policy.check()
            logger.record_tabular("epoch", epoch)
            logger.record_tabular("loss", loss)
            logger.dump_tabular()

        # save periodical policy
        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info("Saving periodic policy to {} ...".format(policy_path))
            policy.save(policy_path)
            logger.info("Saving policy to {}".format(latest_policy_path))
            policy.save(latest_policy_path)

        # Sanity check - Make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    # Save latest policy
    if rank == 0 and save_path:
        logger.info("Saving policy to {}".format(latest_policy_path))
        policy.save(latest_policy_path)

def train_nn_q_estimator(save_path, save_interval, policy, n_epochs, demo_file, demo_test_file, **kwargs):

    # Consider rank as pid
    rank = MPI.COMM_WORLD.Get_rank()

    # Make directions for saving policies
    if save_path:
        policy_save_path = save_path + "/rl_demo_critic/"
        os.makedirs(policy_save_path, exist_ok=True)
        latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
        periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")
        # queries
        output = save_path + "/output/"
        os.makedirs(output, exist_ok=True)

    # Fill in demonstration data
    policy.update_dataset(demo_file, demo_test_file)

    check_interval = n_epochs // 100 if n_epochs > 100 else 1
    for epoch in range(n_epochs):

        # train - This will go through the entire training set once.
        policy.train()

        # test
        if rank == 0 and epoch % check_interval == 0:
            loss = policy.check()
            logger.record_tabular("epoch", epoch)
            logger.record_tabular("loss", loss)
            logger.dump_tabular()

        # save periodical policy
        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info("Saving periodic policy to {} ...".format(policy_path))
            policy.save(policy_path)

        # Save latest policy
        if rank == 0 and save_path:
            logger.info("Saving policy to {}".format(latest_policy_path))
            policy.save(latest_policy_path)

        # Store anything we need into a numpyz file.
        # policy.query_output(os.path.join(output, "query_latest.npz"))

        # Sanity check - Make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def train_action_imitator(
    save_path, save_interval, policy, n_epochs, demo_file, demo_test_file, evaluator, n_test_rollouts, **kwargs
):

    # Consider rank as pid
    rank = MPI.COMM_WORLD.Get_rank()

    # Make directions for saving policies
    if save_path:
        policy_save_path = save_path + "/rl_demo_actor/"
        os.makedirs(policy_save_path, exist_ok=True)
        latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
        periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")

    # Fill in demonstration data
    policy.init_demo_buffer(demo_file, demo_test_file)

    check_interval = n_epochs // 100 if n_epochs > 100 else 1
    for epoch in range(n_epochs):
        # train - This will go through the entire training set once.
        policy.train()

        # test
        logger.debug("train_ddpg_main.train_reinforce -> Testing.")
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        loss = policy.check()
        logger.record_tabular("epoch", epoch)
        logger.record_tabular("loss", loss)
        for key, val in evaluator.logs("test"):
            logger.record_tabular(key, mpi_average(val))
        if rank == 0 and epoch % check_interval == 0:
            logger.dump_tabular()

        # save periodical policy
        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info("Saving periodic policy to {} ...".format(policy_path))
            policy.save(policy_path)

        # Save latest policy
        if rank == 0 and save_path:
            logger.info("Saving policy to {}".format(latest_policy_path))
            policy.save(latest_policy_path)

    # Sanity check - Make sure that different threads have different seeds
    local_uniform = np.random.uniform(size=(1,))
    root_uniform = local_uniform.copy()
    MPI.COMM_WORLD.Bcast(root_uniform, root=0)
    if rank != 0:
        assert local_uniform[0] != root_uniform[0]


def train_reinforce(
    save_path,
    save_interval,
    policy,
    rollout_worker,
    evaluator,
    n_epochs,
    n_cycles,
    n_batches,
    n_test_rollouts,
    demo_file,
    **kwargs,
):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        policy_save_path = save_path + "/rl/"
        os.makedirs(policy_save_path, exist_ok=True)
        latest_policy_path = os.path.join(policy_save_path, "policy_latest.pkl")
        best_policy_path = os.path.join(policy_save_path, "policy_best.pkl")
        periodic_policy_path = os.path.join(policy_save_path, "policy_{}.pkl")
        # queries
        ac_output_save_path = save_path + "/ac_output/"
        critic_q_save_path = save_path + "/critic_q/"
        uncertainty_save_path = save_path + "/uncertainty/"
        os.makedirs(ac_output_save_path, exist_ok=True)
        os.makedirs(critic_q_save_path, exist_ok=True)
        os.makedirs(uncertainty_save_path, exist_ok=True)

    if policy.demo_actor is not "none":
        policy.init_demo_buffer(demo_file)

    best_success_rate = -1

    for epoch in range(n_epochs):
        logger.debug("train_ddpg_main.train_reinforce -> epoch: {}".format(epoch))
        # train
        rollout_worker.clear_history()
        policy.update_global_step()
        for cycle in range(n_cycles):
            logger.debug("train_ddpg_main.train_reinforce -> cycle: {}".format(cycle))
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for batch in range(n_batches):
                logger.debug("train_ddpg_main.train_reinforce -> batch: {}".format(batch))
                policy.train()
            # policy.check_train()
            policy.update_target_net()

        # Test
        logger.debug("train_ddpg_main.train_reinforce -> Testing.")
        evaluator.clear_history()
        evaluator.generate_rollouts()

        # record logs
        logger.record_tabular("epoch", epoch)
        for key, val in evaluator.logs("test"):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs("train"):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # Store anything we need into a numpyz file.
        # policy.query_ac_output(os.path.join(ac_output_save_path, "query_{:03d}.npz".format(epoch)))
        # policy.query_critic_q(os.path.join(critic_q_save_path, "query_latest.npz"))
        # policy.query_uncertainty(os.path.join(uncertainty_save_path, "query_{:03d}.npz".format(epoch)))

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_path:
            best_success_rate = success_rate
            logger.info("New best success rate: {}.".format(best_success_rate))
            logger.info("Saving policy to {}.".format(best_policy_path))
            evaluator.save_policy(best_policy_path)

        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info("Saving periodic policy to {}.".format(policy_path))
            evaluator.save_policy(policy_path)

        if rank == 0 and save_path:
            logger.info("Saving latest policy to {}.".format(latest_policy_path))
            evaluator.save_policy(latest_policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def train(
    logdir,
    loglevel,
    save_path,
    save_interval,
    env,
    r_scale,
    r_shift,
    eps_length,
    seed,
    train_rl,
    train_rl_epochs,
    rl_num_sample,
    rl_ca_ratio,
    exploit,
    replay_strategy,
    demo_critic,
    demo_actor,
    train_demo_epochs,
    demo_num_sample,
    demo_net_type,
    demo_file,
    demo_test_file,
    demo_policy_file,
    override_params,
    debug_params,
    **kwargs,
):

    # Consider rank as pid.
    rank = MPI.COMM_WORLD.Get_rank()
    num_cpu = MPI.COMM_WORLD.Get_size()

    # Configure logging.
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None  # it is either the default log dir or the specified directory

    logger.info("Launching the training process with {} cpu core(s).".format(num_cpu))
    logger.info("Setting log level to {}.".format(loglevel))
    logger.set_level(loglevel)
    logger.debug("train_ddpg_main.launch -> Using debug mode. Avoid training with too many epochs.")

    # Seed everything.
    rank_seed = seed + 1_000_000 * rank
    set_global_seeds(rank_seed)

    # Get default params from config and update params.
    params = config.DEFAULT_PARAMS
    params["rank_seed"] = rank_seed
    params["env_name"] = env
    params["r_scale"] = r_scale
    params["r_shift"] = r_shift
    params["eps_length"] = eps_length
    params["exploit"] = exploit
    params["train_rl_epochs"] = train_rl_epochs
    params["train_demo_epochs"] = train_demo_epochs
    params["rl_demo_critic"] = demo_critic
    params["rl_demo_actor"] = demo_actor
    params["rl_ca_ratio"] = rl_ca_ratio
    params["rl_num_sample"] = rl_num_sample
    params["demo_num_sample"] = demo_num_sample
    params["demo_net_type"] = "yw.ddpg_main.demo_policy:" + demo_net_type
    params["replay_strategy"] = replay_strategy  # For HER: future or none
    params["config"] = (
        "-".join(
            [
                "ddpg",
                demo_critic,
                "r_sample:" + str(rl_num_sample),
                "d_sample:" + str(-1 if demo_critic is "none" else demo_num_sample),
            ]
        )
        if train_rl
        else ""
    )
    if override_params:
        # make it possible to override any parameter.
        for key, val in override_params.items():
            assert key in params.keys(), "Wrong override parameter."
            params.update({key: type(params[key])(val)})
    if debug_params:
        # put any parameter in the debug_param global dictionary.
        params.update(**config.DEBUG_PARAMS)

    # Record params in a json file for later use.
    # This should be done before preparing_params because that function will add function variables that cannot be
    # logged. Also at this moment there should be no function variables.
    if rank == 0:
        with open(os.path.join(logger.get_dir(), "params.json"), "w") as f:
            json.dump(params, f)

    # Prepare parameters for training
    params = config.add_env_params(params=params)

    # Configure and Train demonstration
    demo_policy = None
    if demo_actor == "nn" and demo_file is not None:
        logger.info("Training demonstration neural net to produce expected action.")
        demo_policy = config.configure_action_imitator(params=params)
        evaluator = config.config_rollout(params=params, policy=demo_policy)
        train_action_imitator(
            save_path=save_path,
            save_interval=save_interval,
            policy=demo_policy,
            n_epochs=params["train_demo_epochs"],
            demo_file=demo_file,
            demo_test_file=demo_test_file,
            evaluator=evaluator,
            n_test_rollouts=params["n_test_rollouts"],
        )
    elif demo_critic == "nn" and demo_file is not None:
        logger.info("Training demonstration neural net to produce expected Q value.")
        demo_policy = config.configure_nn_q_estimator(params=params)
        train_nn_q_estimator(
            save_path=save_path,
            save_interval=save_interval,
            policy=demo_policy,
            n_epochs=params["train_demo_epochs"],
            demo_file=demo_file,
            demo_test_file=demo_test_file,
        )
    elif demo_critic == "gp" and demo_file is not None:
        logger.info("Training gaussian process to produce expected Q value.")
        demo_policy = config.configure_gp_q_estimator(params=params)
        train_gp_q_estimator(
            save_path=save_path,
            save_interval=save_interval,
            policy=demo_policy,
            n_epochs=params["train_demo_epochs"],
            demo_file=demo_file,
            demo_test_file=demo_test_file,
        )
    else:
        logger.info("Skip demonstration training.")
        if demo_policy_file:
            logger.info("Will use provided policy file from {}.".format(demo_policy_file))
            with open(demo_policy_file, "rb") as f:
                demo_policy = pickle.load(f)
        else:
            logger.info("Demo policy file does not exit, cannot use demo policy.")

    # Configure and train rl agent
    if train_rl:
        policy = config.configure_ddpg(params=params, demo_policy=demo_policy)
        rollout_worker = config.config_rollout(params=params, policy=policy)
        evaluator = config.config_evaluator(params=params, policy=policy)
        logger.info("Training the RL agent.")
        train_reinforce(
            save_path=save_path,
            save_interval=save_interval,
            policy=policy,
            rollout_worker=rollout_worker,
            evaluator=evaluator,
            n_epochs=params["train_rl_epochs"],
            n_cycles=params["n_cycles"],
            n_batches=params["n_batches"],
            n_test_rollouts=params["n_test_rollouts"],
            demo_file=demo_file,
        )
    else:
        logger.info("Skip RL agent training.")

if __name__ == "__main__":
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()
    # logging
    ap.parser.add_argument("--logdir", help="log directory", type=str, default=os.getenv("TEMPDIR") + "/Train/log")
    ap.parser.add_argument("--loglevel", help="log level", type=int, default=2)
    # save results - this will be for both demo and rl
    ap.parser.add_argument("--save_path", help="policy path", type=str, default=os.getenv("TEMPDIR") + "/Train/policy")
    ap.parser.add_argument("--save_interval", help="the interval which policy pickles are saved", type=int, default=0)
    # program - TODO: test if this works for the demonstration training part.
    ap.parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    # environment setup
    ap.parser.add_argument("--env", help="name of the environment", type=str, default="Reach2DFirstOrder")
    ap.parser.add_argument("--r_scale", help="down scale the reward", type=float, default=1.0)
    ap.parser.add_argument("--r_shift", help="shift the reward (before shifting)", type=float, default=0.0)
    ap.parser.add_argument("--eps_length", help="change the maximum episode length", type=int, default=0)
    # training
    ap.parser.add_argument("--train_rl", help="whether or not to train the ddpg", type=int, default=0)
    ap.parser.add_argument(
        "--train_rl_epochs", help="the number of training epochs to run for RL", type=int, default=1
    )
    ap.parser.add_argument(
        "--train_demo_epochs", help="the number of training epochs to run for demo", type=int, default=1
    )
    # DDPG configuration
    ap.parser.add_argument("--rl_num_sample", help="number of ddpg heads", type=int, default=1)
    ap.parser.add_argument(
        "--rl_ca_ratio", help="use 2 for td3 or 1 for normal ddpg", type=int, choices=[1, 2], default=1
    )  # do not use this flag for now
    ap.parser.add_argument("--exploit", help="whether or not to use e-greedy exploration", type=int, default=0)
    ap.parser.add_argument(
        "--replay_strategy",
        help="the HER replay strategy to be used. 'future' uses HER, 'none' disables HER",
        type=str,
        choices=["none", "future"],
        default="future",
    )
    # demo configuration
    ap.parser.add_argument(
        "--demo_critic",
        help="use a neural network as critic or a gaussian process. Need to provide or train a demo policy if not set to none",
        type=str,
        choices=["nn", "gp", "none"],
        default="none",
    )
    ap.parser.add_argument(
        "--demo_actor",
        help="use a neural network as actor. Need to provide or train a demo policy if not set to none",
        type=str,
        choices=["nn", "none"],
        default="none",
    )
    ap.parser.add_argument(
        "--demo_policy_file",
        help="demonstration policy file (provide this if you want to train RL agent only)",
        type=str,
        default=None,
    )
    ap.parser.add_argument("--demo_num_sample", help="number of demonstration heads", type=int, default=1)
    ap.parser.add_argument("--demo_file", help="demonstration training dataset", type=str, default=None)
    ap.parser.add_argument("--demo_test_file", help="demonstration test dataset", type=str, default=None)
    ap.parser.add_argument(
        "--demo_net_type",
        help="demonstration neural network type, ensemble, baysian",
        type=str,
        choices=["EnsembleNN", "BaysianNN"],
        default="EnsembleNN",
    )
    # others
    ap.parser.add_argument(
        "--override_params", help="whether or not to overwrite default parameters", type=int, default=0
    )
    ap.parser.add_argument(
        "--debug_params", help="override some parameters for internal regression tests", type=int, default=0
    )
    ap.parse(sys.argv)

    print("Launching the training process.")
    train(**ap.get_dict())
