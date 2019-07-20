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
from yw.util.mpi_util import mpi_input
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
    rank = MPI.COMM_WORLD.Get_rank() if MPI is not None else 0

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
            for k, v in dir_param_dict.items():
                if os.path.exists(k):
                    if (
                        not mpi_input(
                            "Directory {} already exists! overwrite the directory? (!enter to cancel): ".format(k)
                        )
                        == ""
                    ):
                        logger.info("Canceled!")
                        exit(1)
                os.makedirs(k, exist_ok=True)
                # copy params.json file
                with open(os.path.join(k, "params.json"), "w") as f:
                    json.dump(v, f)
                # copy demo_sata file if exist
                demo_file = os.path.join(exp_dir, "demo_data.npz")
                demo_dest = os.path.join(k, "demo_data.npz")
                if os.path.isfile(demo_file):
                    copyfile(demo_file, demo_dest)

            # run experiments
            for k in dir_param_dict.keys():
                train_entry(root_dir=k)
                if policy_dir == None:
                    policy_dir = k

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
