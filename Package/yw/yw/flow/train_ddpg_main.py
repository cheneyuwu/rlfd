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
from subprocess import CalledProcessError

import click
import numpy as np
import json
import pickle
from mpi4py import MPI

from yw.ddpg_main import config
from yw.tool import logger
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


def launch(
    logdir,
    loglevel,
    save_path,
    save_interval,
    num_cpu,
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
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ["--bind-to", "core"])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == "parent":
            sys.exit(0)
        import yw.util.tf_util as U

        U.single_threaded_session().__enter__()

    # Consider rank as pid.
    rank = MPI.COMM_WORLD.Get_rank()

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


# Top Level User API
# =====================================
@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))

# Log
@click.option("--logdir", type=str, default=os.getenv("PROJECT") + "/Temp/Temp/log", help="Log directory.")
@click.option("--loglevel", type=int, default=1, help="Logger level.")

# Save results - this will be for both demo and rl
@click.option("--save_path", type=str, default=os.getenv("PROJECT") + "/Temp/Temp/policy", help="Policy path.")
@click.option("--save_interval", type=int, default=0, help="The interval with which policy pickles are saved.")

# Program - TODO: test if this works for the demonstration training part.
@click.option("--num_cpu", type=int, default=1, help="The number of CPU cores to use (using MPI).")
@click.option("--seed", type=int, default=0, help="Seed both the environment and the training code.")

# Environment setup
@click.option("--env", type=str, default="Reach2D", help="Name of the environment.")
@click.option("--r_scale", type=float, default=1, help="Down scale the reward.")
@click.option("--r_shift", type=float, default=0, help="Shift the reward (before shifting).")
@click.option("--eps_length", type=int, default=0, help="Change the maximum episode length.")

# Training
@click.option("--train_rl", type=int, default=0, help="Whether or not to train the ddpg.")
@click.option("--train_rl_epochs", type=int, default=1, help="The number of training epochs to run for RL.")
@click.option("--train_demo_epochs", type=int, default=1, help="The number of training epochs to run for demo.")

# DDPG Configuration
@click.option("--rl_num_sample", type=int, default=1, help="Number of ddpg heads.")
@click.option(
    "--rl_ca_ratio", type=click.Choice([1, 2]), default=1, help="Use 2 for td3 or 1 for normal ddpg."
)  # this will only affect number of critic, do not use it for now
@click.option("--exploit", type=int, default=0, help="Whether or not to use e-greedy exploration.")
@click.option(
    "--replay_strategy",
    type=click.Choice(["future", "none"]),
    default="future",
    help='The HER replay strategy to be used. "future" uses HER, "none" disables HER.',
)

# Demo Configuration
@click.option(
    "--demo_critic",
    type=click.Choice(["nn", "gp", "none"]),
    default="none",
    help="Use a neural network as critic or a gaussian process. Need to provide or train a demo policy if not set to none",
)
@click.option(
    "--demo_actor",
    type=click.Choice(["nn", "none"]),
    default="none",
    help="Use a neural network as actor. Need to provide or train a demo policy if not set to none",
)
@click.option(
    "--demo_policy_file",
    type=str,
    default=None,
    help="Demonstration policy file. Provide this if you want to train RL agent only.",
)
@click.option("--demo_num_sample", type=int, default=1, help="Number of demonstration heads.")
@click.option("--demo_file", type=str, default=None, help="Demonstration training dataset.")
@click.option("--demo_test_file", type=str, default=None, help="Demonstration test dataset.")
@click.option(
    "--demo_net_type",
    type=click.Choice(["EnsembleNN", "BaysianNN"]),
    default="EnsembleNN",
    help="Demonstration neural network type, ensemble, baysian or ensemble of actor",
)

# Others
@click.option("--override_params", type=int, default=0, help="Whether or not to overwrite default parameters.")
@click.option("--debug_params", type=int, default=0, help="Override some parameters for internal regression tests.")
@click.pass_context
def main(ctx, **kwargs):
    print("Provided Arguments:")
    print("{:<30}{:<30}".format("Option", "Value"))
    for key, value in kwargs.items():
        print("{:<30}{:<30}".format(str(key), str(value)))

    if kwargs["override_params"] == 1:
        unparsed_options = []
        for i in range(0, len(ctx.args)):
            unparsed_options += list(filter(bool, ctx.args[i].split("=")))
        assert len(unparsed_options) % 2 == 0, "Wrong number of arguments!"
        override = {unparsed_options[i].strip("-"): unparsed_options[i + 1] for i in range(0, len(ctx.args), 2)}
        print("Overrided Parameters:")
        print("{:<30}{:<30}".format("Option", "Value"))
        for key, value in override.items():
            print("{:<30}{:<30}".format(str(key), str(value)))
        kwargs["override_params"] = override

    print("Launching the training process.")
    launch(**kwargs)


if __name__ == "__main__":
    main()
