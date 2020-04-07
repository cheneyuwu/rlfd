import argparse
import copy
import importlib
import json
import os
import shutil
import sys

# must include gym before loading mpi, for compute canada cluster
try:
    import mujoco_py
except:
    mujoco_py = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from td3fd import logger
from td3fd.demo_util.generate_demo import main as demo_entry
from td3fd.demo_util.generate_demo_rlkit import main as demo_rlkit_entry
from td3fd.evaluate import main as evaluate_entry
from td3fd.evaluate_rlkit import main as evaluate_rlkit_entry
from td3fd.plot import main as plot_entry
from td3fd.train import main as train_entry
from td3fd.util.mpi_util import mpi_exit, mpi_input

# tf debug
# from td3fd.ddpg.debug.generate_query import main as generate_query_entry
# from td3fd.ddpg.debug.visualize_query import main as visualize_query_entry
from td3fd.ddpg2.debug.check_potential import main as check_potential_entry
# # torch debug
# from td3fd.td3.debug.generate_query import main as generate_query_entry
# from td3fd.td3.debug.visualize_query import main as visualize_query_entry

def import_param_config(load_dir):
    """Assume that there is a gv called params_config that contains all the params
    """
    # get the full path of the load directory
    abs_load_dir = os.path.abspath(os.path.expanduser(load_dir))
    spec = importlib.util.spec_from_file_location("module.name", abs_load_dir)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params_configs = [
        getattr(module, params_config) for params_config in dir(module) if params_config.startswith("params_config")
    ]
    return params_configs


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

def transform_config_name(config_name):
    """ Transfer the legend names"""
    print(config_name)
    for i in range(len(config_name)):
        if config_name[i].startswith("config"):
            if config_name[i] == "config_TD3_NF_Shaping":
                return ["TD3+Shaping (NF)"]
            elif config_name[i] == "config_TD3_GAN_Shaping":
                return ["TD3+Shaping (GAN)"]
            elif config_name[i] == "config_BC":
                return ["BC"]
            elif config_name[i] == "config_TD3":
                return ["TD3"]
            elif config_name[i] == "config_TD3_BC":
                if "prm_loss_weight_0.0001" in config_name:
                    return ["TD3+BC, $\lambda$=0.0001"]
                elif "prm_loss_weight_0.001" in config_name:
                    return ["TD3+BC, $\lambda$=0.001"]
                elif "prm_loss_weight_0.01" in config_name:
                    return ["TD3+BC, $\lambda$=0.01"]
                elif "prm_loss_weight_0.1" in config_name:
                    return ["TD3+BC, $\lambda$=0.1"]
                else:
                    return ["TD3+BC"]
            elif config_name[i] == "config_TD3_BC_QFilter":
                if "prm_loss_weight_0.0001" in config_name:
                    return ["TD3+BC+QFilter, $\lambda$=0.0001"]
                elif "prm_loss_weight_0.001" in config_name:
                    return ["TD3+BC+QFilter, $\lambda$=0.001"]
                elif "prm_loss_weight_0.01" in config_name:
                    return ["TD3+BC+QFilter, $\lambda$=0.01"]
                elif "prm_loss_weight_0.1" in config_name:
                    return ["TD3+BC+QFilter, $\lambda$=0.1"]
                else:
                    return ["TD3+BC+QFilter"]
            elif config_name[i] == "config_TD3_BC_Init":
                return ["TD3+BC Init."]
    return config_name


def main(targets, exp_dir, policy, save_dir, **kwargs):

    # Consider rank as pid.
    comm = MPI.COMM_WORLD if MPI is not None else None
    total_num_cpu = comm.Get_size() if comm is not None else 1
    rank = comm.Get_rank() if comm is not None else 0

    # Use the default logger setting
    logger.configure()

    # Setup
    assert targets is not None, "require --targets"
    # get the abs path of the exp dir
    assert exp_dir is not None, "must provide the experiment root directory --exp_dir"
    exp_dir = os.path.abspath(os.path.expanduser(exp_dir))
    if policy is not None:
        policy = os.path.abspath(os.path.expanduser(policy))

    for target in targets:
        if "rename:" in target:
            logger.info("\n\n=================================================")
            logger.info("Renaming the config to params_renamed.json!")
            logger.info("=================================================")
            # excluded params
            exc_params = []
            exc_params = ["seed"]  # CHANGE this!
            # adding checking
            config_file = target.replace("rename:", "")
            params_configs = import_param_config(config_file)
            dir_param_dict = {}
            for params_config in params_configs:
                dir_param_dict.update(generate_params(exp_dir, params_config))
            if rank == 0:
                for k, v in dir_param_dict.items():
                    assert os.path.exists(k), k
                    # copy params.json file, rename the config entry
                    varied_params = k[len(exp_dir) + 1 :].split("/")
                    config_name = [x for x in varied_params if not any([x.startswith(y) for y in exc_params])]
                    config_name = transform_config_name(config_name)
                    v["config"] = "-".join(config_name)
                    with open(os.path.join(k, "params_renamed.json"), "w") as f:
                        json.dump(v, f)

        elif "train:" in target:
            logger.info("\n\n=================================================")
            logger.info("Launching the training experiment!")
            logger.info("=================================================")
            # adding checking
            config_file = target.replace("train:", "")
            params_configs = import_param_config(config_file)
            dir_param_dict = {}
            for params_config in params_configs:
                dir_param_dict.update(generate_params(exp_dir, params_config))

            # store json config file to the target directory
            if rank == 0:
                for k, v in dir_param_dict.items():
                    if os.path.exists(k):  # COMMENT out this check for restarting
                        logger.info("Directory {} already exists!".format(k))
                        mpi_exit(1, comm=comm)
                    os.makedirs(k, exist_ok=True)
                    # copy params.json file
                    with open(os.path.join(k, "params.json"), "w") as f:
                        json.dump(v, f)
                    # copy demo_sata file if exist
                    demo_file = os.path.join(exp_dir, "demo_data.npz")
                    demo_dest = os.path.join(k, "demo_data.npz")
                    if os.path.isfile(demo_file):
                        shutil.copyfile(demo_file, demo_dest)

            # sync the process
            if comm is not None:
                comm.Barrier()

            if policy == None:
                policy = os.path.join(list(dir_param_dict.keys())[0], "rl/policy_latest.pkl")
                logger.info("Setting policy to {}".format(policy))

            # run experiments
            parallel = 1  # CHANGE this number to allow launching in serial
            # total num exps
            num_exp = len(dir_param_dict.keys())

            if comm is None:
                logger.info("No MPI provided. Launching scripts in series.")
                for k in dir_param_dict.keys():
                    train_entry(root_dir=k)

            elif not parallel:
                logger.info("{} experiments in series.".format(num_exp))
                comm_for_exps = comm.Split(color=int(rank > 0))
                if rank == 0:
                    assert comm_for_exps.Get_size() == 1
                    for k in dir_param_dict.keys():
                        train_entry(root_dir=k, comm=comm_for_exps)
            else:
                logger.info("{} experiments in parallel.".format(num_exp))
                assert num_exp <= total_num_cpu, "no enough cpu! need {} cpus".format(num_exp)
                # select processes to run the experiment, which is [0, num_exp]
                comm_for_exps = comm.Split(color=int(rank >= num_exp))
                if rank < num_exp:
                    assert comm_for_exps.Get_size() == num_exp
                    comm_for_each_exp = comm_for_exps.Split(color=comm_for_exps.Get_rank())
                    assert comm_for_each_exp.Get_size() == 1
                    k = list(dir_param_dict.keys())
                    train_entry(root_dir=k[comm_for_exps.Get_rank()], comm=comm_for_each_exp)

            if comm is not None:
                comm.Barrier()

        elif target == "demo":
            assert policy != None
            logger.info("\n\n=================================================")
            logger.info("Using policy file from {} to generate demo data.".format(policy))
            logger.info("=================================================")
            if rank == 0:
                demo_entry(policy=policy, root_dir=exp_dir)

        elif "demo_rlkit:" in target:
            assert policy != None
            logger.info("\n\n=================================================")
            logger.info("Using policy file from {} to generate demo data.".format(policy))
            logger.info("=================================================")
            config_file = target.replace("demo_rlkit:", "")
            params_configs = import_param_config(config_file)
            env_name = params_configs[0]["env_name"]
            if rank == 0:
                demo_rlkit_entry(policy=policy, root_dir=exp_dir, env_name=env_name)

        elif target == "evaluate":
            assert policy != None
            logger.info("\n\n=================================================")
            logger.info("Evaluating using policy file from {}.".format(policy))
            logger.info("=================================================")
            if rank == 0:
                evaluate_entry(policy=policy)

        elif "evaluate_rlkit:" in target:
            assert policy != None
            logger.info("\n\n=================================================")
            logger.info("Evaluating using policy file from {}.".format(policy))
            logger.info("=================================================")
            config_file = target.replace("evaluate_rlkit:", "")
            params_configs = import_param_config(config_file)
            env_name = params_configs[0]["env_name"]
            if rank == 0:
                evaluate_rlkit_entry(policy=policy, env_name=env_name)

        elif target == "plot":
            logger.info("\n\n=================================================")
            logger.info("Plotting.")
            logger.info("=================================================")
            if rank == 0:  # plot does not need to be run on all threads
                plot_entry(
                    dirs=[exp_dir],
                    save_dir=save_dir,
                    xys=[
                        # "epoch:test/success_rate",
                        # "epoch:test/:reward_per_eps",
                        # "train/steps:test/success_rate",
                        "train/steps:test/reward_per_eps",
                        # for rlkit
                        # "exploration/num steps total:evaluation/Average Returns"
                        # "exploration/num steps total:trainer/Policy Loss"
                        # "train-steps:evaluation/episode-reward-mean",  # for soft learning
                    ],
                    smooth=True,
                )

        # elif target == "gen_query":
        #     logger.info("\n\n=================================================")
        #     logger.info("Generating queries at: {}".format(exp_dir))
        #     logger.info("=================================================")
        #     if rank == 0:
        #         generate_query_entry(exp_dir=exp_dir, save=True)

        # elif target == "vis_query":
        #     logger.info("\n\n=================================================")
        #     logger.info("Visualizing queries.")
        #     logger.info("=================================================")
        #     if rank == 0:
        #         visualize_query_entry(exp_dir=exp_dir, save=True)

        elif target == "check":
            logger.info("\n\n=================================================")
            logger.info("Check potential.")
            logger.info("=================================================")
            if rank == 0:
                check_potential_entry(exp_dir=exp_dir)

        elif target == "copy_result":
            expdata_dir = os.path.abspath(os.path.expanduser(os.environ["EXPDATA"]))
            print("Experiment directory:", expdata_dir)
            assert os.path.exists(expdata_dir)
            exprun_dir = os.path.abspath(os.path.expanduser(os.environ["EXPRUN"]))
            print("Experiment running direcory:", exprun_dir)
            assert os.path.exists(exprun_dir)
            rel_exp_dir = os.path.relpath(exp_dir, exprun_dir)
            assert not rel_exp_dir.startswith("..")
            # copy files
            for dirname, _, files in os.walk(exp_dir):
                if not "log.txt" in files:
                    continue
                # copy files
                rel_dir_name = os.path.relpath(dirname, exprun_dir)
                target_dir_name = os.path.join(expdata_dir, rel_dir_name)
                os.makedirs(target_dir_name, exist_ok=True)
                print("Found results:", rel_dir_name)

                for file in ["log.txt", "params.json", "params_renamed.json", "progress.csv", "rl/policy_latest.pkl"]:
                    src = os.path.join(dirname, file)
                    if not os.path.exists(src):
                        continue
                    shutil.copy2(src, target_dir_name)
        else:
            assert 0, "unknown target: {}".format(target)

        if comm is not None:
            comm.Barrier()

        logger.reset()

    return


if __name__ == "__main__":

    from td3fd.util.cmd_util import ArgParser

    exp_parser = ArgParser(allow_unknown_args=False)
    exp_parser.parser.add_argument(
        "--exp_dir", help="top level directory to store experiment results", type=str, default=os.getcwd()
    )
    exp_parser.parser.add_argument(
        "--save_dir", help="top level directory to store plots", type=str, default=os.getcwd()
    )
    exp_parser.parser.add_argument(
        "--targets",
        help="target or list of targets in [demo_data, train:<parameter file>.py, plot, evaluate]",
        type=str,
        nargs="+",
    )
    exp_parser.parser.add_argument(
        "--policy",
        help="when target is evaluate or demodata, specify the policy file to be used, <policy name>.pkl",
        type=str,
        default=None,
    )
    exp_parser.parse(sys.argv)
    main(**exp_parser.get_dict())
