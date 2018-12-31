import click
import numpy as np
import pickle

# For configuring all the parameters
from yw.ddpg_no_goal import config
# from yw.ddpg.rollout import RolloutWorker
from yw.util.mpi_util import mpi_fork, mpi_average, set_global_seeds
# DDPG Package import
from yw import logger


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params["env_name"] = env_name
    params["rank_seed"] = seed
    # params.update(**override_params)  # makes it possible to override any parameter

    params = config.prepare_params(params=params)

    demo = config.config_demo(params=params, policy=policy)

    # Run evaluation.
    demo.clear_history()
    for _ in range(n_test_rollouts):
        demo.generate_rollouts()

    # record logs
    for key, val in demo.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
