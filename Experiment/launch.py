import argparse
import os
import sys
import json
import copy
import importlib
from shutil import copyfile

# must include gym before loading mpi, for compute canada cluster
import gym

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from yw.tool import logger
from yw.util.mpi_util import mpi_input, mpi_exit
from yw.flow.train_ddpg_main import main as train_entry
from yw.flow.generate_demo import main as demo_entry
from yw.flow.plot import main as plot_entry
from yw.flow.run_agent import main as display_entry
from yw.flow.query.generate_query import main as generate_query_entry
from yw.flow.query.visualize_query import main as visualize_query_entry


def import_param_config(load_dir):
    """Assume that there is a gv called params_config that contains all the params
    """
    module = importlib.import_module(load_dir.replace("/", ".").replace(".py", ""))
    params_config = getattr(module, "params_config")
    return params_config


def generate_params(root_dir, param_config):
    res = {root_dir: {}}  # the result is a directory name with a dict
    for param_name, param_val in param_config.items():
        if type(param_val) == tuple:
            new_res = {}
            for direc, params in res.items():
                for val in param_val:
                    new_dir = os.path.join(direc, param_name + "_" + str(val))
                    new_params = copy.deepcopy(params)
                    new_params[param_name] = val
                    new_res[new_dir] = new_params
            res = new_res
        elif type(param_val) == dict:
            sub_res = generate_params("", param_val)
            new_res = {}
            for direc, params in res.items():
                for sub_direc, sub_params in sub_res.items():
                    new_dir = os.path.join(direc, sub_direc)
                    new_params = copy.deepcopy(params)
                    new_params[param_name] = copy.deepcopy(sub_params)
                    new_res[new_dir] = new_params
            res = new_res
        else:
            for params in res.values():
                params[param_name] = param_val
    return res


def main(targets, exp_dir, policy_dir, **kwargs):

    # Consider rank as pid.
    comm = MPI.COMM_WORLD if MPI is not None else None
    total_num_cpu = comm.Get_size() if comm is not None else 1
    rank = comm.Get_rank() if comm is not None else 0

    # Use the default logger setting
    logger.configure()

    for target in targets:

        if "train:" in target:
            logger.info("\n\n=================================================")
            logger.info("Launching the training experiment!")
            logger.info("=================================================")
            # adding checking
            config_file = target.replace("train:", "")
            params_config = import_param_config(config_file)
            dir_param_dict = generate_params(exp_dir, params_config)

            # store json config file to the target directory
            if rank == 0:
                for k, v in dir_param_dict.items():
                    if os.path.exists(k):
                        msg = "Directory {} already exists! overwrite the directory? (!enter to cancel): ".format(k)
                        if not input(msg) == "":
                            logger.info("Canceled!")
                            mpi_exit(1)
                    os.makedirs(k, exist_ok=True)
                    # copy params.json file
                    with open(os.path.join(k, "params.json"), "w") as f:
                        json.dump(v, f)
                    # copy demo_sata file if exist
                    demo_file = os.path.join(exp_dir, "demo_data.npz")
                    demo_dest = os.path.join(k, "demo_data.npz")
                    if os.path.isfile(demo_file):
                        copyfile(demo_file, demo_dest)

            # sync the process
            comm.Barrier()

            if policy_dir == None:
                policy_dir = list(dir_param_dict.keys())[0]
                logger.info("Setting policy_dir to {}".format(policy_dir))

            # run experiments
            parallel = 1  # change this number to allow launching in serial
            num_cpu = 2  # change this number to allow multiple processes
            # total num exps
            num_exp = len(dir_param_dict.keys())

            if comm is None:
                logger.info("No MPI provided. Launching scripts in series.")
                for k in dir_param_dict.keys():
                    train_entry(root_dir=k)

            elif not parallel:
                logger.info("{} experiments in series. ({} cpus for each experiment)".format(num_exp, num_cpu))
                assert num_cpu <= total_num_cpu, "no enough cpu! need {} cpus".format(num_cpu)
                comm_for_exps = comm.Split(color=int(rank >= num_cpu))
                if rank < num_cpu:
                    assert comm_for_exps.Get_size() == num_cpu
                    for k in dir_param_dict.keys():
                        train_entry(root_dir=k, comm=comm_for_exps)

            else:
                logger.info("{} experiments in parallel. ({} cpus for each experiment)".format(num_exp, num_cpu))
                cpus_for_training = num_exp * num_cpu
                assert cpus_for_training <= total_num_cpu, "no enough cpu! need {} cpus".format(cpus_for_training)
                # select processes to run the experiment, which is [0, num_exp]
                comm_for_exps = comm.Split(color=int(rank >= cpus_for_training))
                if rank < cpus_for_training:
                    assert comm_for_exps.Get_size() == cpus_for_training
                    color_for_each_exp = comm_for_exps.Get_rank() % num_exp
                    comm_for_each_exp = comm_for_exps.Split(color=color_for_each_exp)
                    assert comm_for_each_exp.Get_size() == num_cpu
                    k = list(dir_param_dict.keys())
                    train_entry(root_dir=k[color_for_each_exp], comm=comm_for_each_exp)

            comm.Barrier()

        elif target == "demo":
            assert policy_dir != None
            policy_file = os.path.join(policy_dir, "rl/policy_latest.pkl")
            logger.info("\n\n=================================================")
            logger.info("Using policy file from {} to generate demo data.".format(policy_file))
            logger.info("=================================================")
            demo_entry(num_eps=100, seed=0, policy_file=policy_file, store_dir=exp_dir)

        elif target == "display":
            assert policy_dir != None
            logger.info("\n\n=================================================")
            logger.info("Displaying.")
            logger.info("=================================================")
            display_entry(policy_file=os.path.join(policy_dir, "rl/policy_latest.pkl"), seed=0, num_itr=10)

        elif target == "plot" and rank == 0:
            logger.info("\n\n=================================================")
            logger.info("Plotting.")
            logger.info("=================================================")
            plot_entry(
                dirs=[exp_dir],
                xys=[
                    "epoch:test/success_rate",
                    "epoch:test/total_shaping_reward",
                    "epoch:test/total_reward",
                    "epoch:test/mean_Q",
                    "epoch:test/mean_Q_plus_P",
                ],
            )

        elif target == "gen_query":
            # currently assume only 1 level of subdir
            query_dir = os.path.join(exp_dir, "*")
            logger.info("\n\n=================================================")
            logger.info("Generating queries at: {}".format(query_dir))
            logger.info("=================================================")
            generate_query_entry(directories=[query_dir], save=1)

        elif target == "vis_query":
            logger.info("\n\n=================================================")
            logger.info("Visualizing queries.")
            logger.info("=================================================")
            # currently assume only 1 level of subdir
            query_dir = os.path.join(exp_dir, "*")
            visualize_query_entry(load_dirs=[query_dir], save=1)

        else:
            assert 0, "unknown target: {}".format(target)

        logger.reset()

    return


if __name__ == "__main__":

    from yw.util.cmd_util import ArgParser

    """
    Example flow:
    1. (mpirun -np 1) python launch.py --targets train:rldense
       Train a rl agent with config defined in rldense.py located in the same directory as launch.py. The output is in
       TempResult/Temp by default (you can overwrite this default using --exp_dir <overwrite directory>).
    2. (mpirun -np 1) python launch.py --targets train:rldense demo
       In addition to 1, also generate demo_data using the trained rl agent and store the demo file as demo_data.py in
       TempResult/Temp by default.
    3. (mpirun -np 1) python launch.py --targets train:rldense demo train:rlnorm
       In addition to 2, also use the generated demo file to train an rl agent with config defined in rlnorm.py located
       in the same directory as launch.py. The output is TempResult/Temp
    4. (mpirun -np 1) python launch.py --targets train:rldense plot
       In addition to 1, also collects result in TempResult/Temp and generates plots
    
    Note:
    1. In the params config file (e.g. rldense.py), if the val of any key is a list, this script will create a
       sub-folder named "key-val" and put the exp result there.

    Run this script with different chain of targets and see what happens! I hope the flow makes sense:)
    """
    exp_parser = ArgParser()
    exp_parser.parser.add_argument("--targets", help="target or list of target", type=str, nargs="+", default=None)
    exp_parser.parser.add_argument(
        "--exp_dir",
        help="top level directory to store experiment results",
        type=str,
        default=os.path.join(os.getenv("EXPERIMENT"), "TempResult/Temp"),
    )
    exp_parser.parser.add_argument(
        "--policy_dir", help="top level directory to store experiment results", type=str, default=None
    )
    exp_parser.parse(sys.argv)
    main(**exp_parser.get_dict())
