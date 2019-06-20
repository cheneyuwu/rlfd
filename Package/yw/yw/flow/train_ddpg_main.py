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

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import json
import pickle
import numpy as np

from yw.tool import logger
from yw.ddpg_main import config
from yw.util.util import set_global_seeds
from yw.util.mpi_util import mpi_average


def train_reinforce(
    save_path, save_interval, policy, rollout_worker, evaluator, n_epochs, n_batches, n_cycles, demo_file, **kwargs
):
    rank = MPI.COMM_WORLD.Get_rank() if MPI != None else 0

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

    if policy.demo_actor != "none" or policy.demo_critic == "shaping":
        policy.init_demo_buffer(demo_file, update_stats=policy.demo_actor != "none")

    best_success_rate = -1

    for epoch in range(n_epochs):
        logger.debug("train_ddpg_main.train_reinforce -> epoch: {}".format(epoch))

        # Store anything we need into a numpyz file.
        # policy.query_ac_output(os.path.join(ac_output_save_path, "query_{:03d}.npz".format(epoch)))
        # policy.query_critic_q(os.path.join(critic_q_save_path, "query_latest.npz"))
        policy.query_uncertainty(os.path.join(uncertainty_save_path, "query_{:03d}.npz".format(epoch)))

        # Train
        rollout_worker.clear_history()
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

        # Log
        logger.record_tabular("epoch", epoch)
        for key, val in evaluator.logs("test"):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs("train"):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # Save the policy if it's better than the previous ones
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

        # Make sure that different threads have different seeds
        if MPI != None:
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
    env_args,
    seed,
    train_rl_epochs,
    rl_num_sample,
    rl_ca_ratio,
    exploit,
    rl_replay_strategy,
    num_demo,
    demo_critic,
    demo_actor,
    demo_file,
    debug_params,
    unknown_params,
    **kwargs,
):

    # Consider rank as pid.
    rank = MPI.COMM_WORLD.Get_rank() if MPI != None else 0
    num_cpu = MPI.COMM_WORLD.Get_size() if MPI != None else 1

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
    params["env_args"] = dict(env_args) if env_args else {}
    params["train_rl_epochs"] = train_rl_epochs
    params["rl_ca_ratio"] = rl_ca_ratio
    params["rl_num_sample"] = rl_num_sample
    params["exploit"] = exploit
    params["rl_replay_strategy"] = rl_replay_strategy  # For HER: future or none
    params["rl_num_demo"] = num_demo
    params["rl_demo_critic"] = demo_critic
    params["rl_demo_actor"] = demo_actor
    params["config"] = "-".join(
        [
            "ddpg:" + demo_critic,
            "num_demo:" + str(num_demo),
            "r_sample:" + str(rl_num_sample),
            "replay:" + rl_replay_strategy,
        ]
    )
    # make it possible to override any parameter.
    for key, val in unknown_params.items():
        assert key in params.keys(), "Wrong override parameter: {}.".format(key)
        params.update({key: type(params[key])(val)})
    # for debugging only
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

    # Configure and train rl agent
    policy = config.configure_ddpg(params=params)
    rollout_worker = config.config_rollout(params=params, policy=policy)
    evaluator = config.config_evaluator(params=params, policy=policy)

    logger.info(
        "Training the RL agent with n_epochs: {}, n_cycles: {}, n_batches: {}.".format(
            params["train_rl_epochs"], params["n_cycles"], params["n_batches"]
        )
    )
    train_reinforce(
        save_path=save_path,
        save_interval=save_interval,
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        n_epochs=params["train_rl_epochs"],
        n_cycles=params["n_cycles"],
        n_batches=params["n_batches"],
        demo_file=demo_file,
    )


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
    ap.parser.add_argument(
        "--env_arg",
        action="append",
        type=lambda kv: [kv.split(":")[0], eval(str(kv.split(":")[1] + '("' + kv.split(":")[2] + '")'))],
        dest="env_args",
    )
    # training
    ap.parser.add_argument(
        "--train_rl_epochs", help="the number of training epochs to run for RL", type=int, default=1
    )
    # DDPG configuration
    ap.parser.add_argument("--rl_num_sample", help="number of ddpg heads", type=int, default=1)
    ap.parser.add_argument(
        "--rl_ca_ratio", help="use 2 for td3 or 1 for normal ddpg", type=int, choices=[1, 2], default=1
    )  # do not use this flag for now
    ap.parser.add_argument("--exploit", help="whether or not to use e-greedy exploration", type=int, default=0)
    ap.parser.add_argument(
        "--rl_replay_strategy",
        help="the replay strategy to be used. 'future' uses HER, 'none' disables HER",
        type=str,
        choices=["none", "her"],
        default="none",
    )
    # demo configuration
    ap.parser.add_argument("--num_demo", help="Number of demonstrations, measured in episodes.", type=int, default=0)
    ap.parser.add_argument(
        "--demo_critic",
        help="use a neural network as critic or a gaussian process. Need to provide or train a demo policy if not set to none",
        type=str,
        choices=["shaping", "none"],
        default="none",
    )
    ap.parser.add_argument(
        "--demo_actor",
        help="use a neural network as actor. Need to provide or train a demo policy if not set to none",
        type=str,
        choices=["shaping", "none"],
        default="none",
    )
    ap.parser.add_argument("--demo_file", help="demonstration training dataset", type=str, default=None)
    # others
    ap.parser.add_argument(
        "--debug_params", help="override some parameters for internal regression tests", type=int, default=0
    )
    ap.parse(sys.argv)

    train(**ap.get_dict())
