# =============================================================================
# Import
# =============================================================================

# System import
import os
import sys
from subprocess import CalledProcessError

# Infra import
import click
import numpy as np
import json
from mpi4py import MPI

# For configuring all the parameters
from yw.ddpg_no_goal import config

# DDPG Package import
from yw import logger
# from yw.ddpg.rollout import RolloutWorker
from yw.util.mpi_util import mpi_fork, mpi_average, set_global_seeds


# =============================================================================
# Functions
# =============================================================================


def train(
    save_path,
    policy_save_interval,
    policy,
    rollout_worker,
    evaluator,
    demo,
    n_epochs,
    n_cycles,
    n_batches,
    n_test_rollouts,
    **kwargs
):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    logger.info("\n*** Training ***")
    best_mean_Q = -999

    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            # Yuchen debug only
            # for key in episode.keys():
            #     logger.info(key)
            #     logger.info(episode[key].shape)
            #     logger.info(episode[key])
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
                policy.check_train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular("epoch", epoch)
        for key, val in evaluator.logs("test"):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs("train"):
            logger.record_tabular(key, mpi_average(val))
        # for key, val in policy.logs():
        #     logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        mean_Q = mpi_average(evaluator.current_mean_Q())
        if rank == 0 and mean_Q >= best_mean_Q and save_path:
            best_mean_Q = mean_Q
            logger.info('New best mean Q: {}. Saving policy to {} ...'.format(best_mean_Q, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]
    demo.clear_history()
    demo.generate_rollouts()


def launch(
    env,
    logdir,
    loglevel,
    save_path,
    num_cpu,
    seed,
    n_epochs,
    clip_return,
    policy_save_interval,
    # override_params={}
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
    # Consider rank as pid
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None  # it is either the default log dir or the specified one
    logger.set_level(loglevel)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params["env_name"] = env
    params["rank_seed"] = rank_seed
    params["clip_return"] = clip_return
    # params.update(**override_params)  # makes it possible to override any parameter

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    params = config.prepare_params(params=params)

    # Configure everything
    policy = config.configure_ddpg(params=params)
    rollout_worker = config.config_rollout(params=params, policy=policy)
    evaluator = config.config_evaluator(params=params, policy=policy)
    demo = config.config_demo(params=params, policy=policy)
    logger.info("\n*** params ***")
    config.log_params(params=params)

    train(
        save_path=save_path,
        policy=policy,
        rollout_worker=rollout_worker,
        evaluator=evaluator,
        demo=demo,
        n_epochs=n_epochs,
        n_cycles=params["n_cycles"],
        n_batches=params["n_batches"],
        n_test_rollouts=params["n_test_rollouts"],
        policy_save_interval=policy_save_interval,
    )


# =============================================================================
# Top Level User API
# =============================================================================
@click.command()
@click.option(
    "--logdir", type=str, default="/home/yuchen/Desktop/FlorianResearch/RLProject/temp2/log", help="Log directory."
)
@click.option(
    "--loglevel", type=str, default=2, help="Logger level."
)
@click.option(
    "--save_path", type=str, default="/home/yuchen/Desktop/FlorianResearch/RLProject/temp2/policy", help="Policy directory."
)
@click.option(
    '--policy_save_interval', type=int, default=5,
    help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option("--num_cpu", type=int, default=6, help="the number of CPU cores to use (using MPI)")
@click.option(
    "--seed", type=int, default=0, help="The random seed used to seed both the environment and the training code"
)
@click.option("--env", type=str, default="Pendulum-v0", help="Name of the environment.")
@click.option("--n_epochs", type=int, default=51, help="The number of training epochs to run")
@click.option("--clip_return", type=int, default=1, help="whether or not returns should be clipped")
def main(**kwargs):
    launch(**kwargs)


if __name__ == "__main__":
    main()