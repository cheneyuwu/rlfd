import argparse
import collections
import copy
import importlib
import json
import os
import shutil
import sys

import mujoco_py  # include at the beginning (bug on compute canada cluster)
import tensorflow as tf
import ray
from ray import tune

from rlfd import logger, plot, train, train_tune
from rlfd.utils import mpi_util
from rlfd.demo_utils import generate_demo
from rlfd import evaluate

# TODO: remove include from the old repo
# tf debug
# from td3fd.ddpg.debug.generate_query import main as generate_query_entry
# from td3fd.ddpg.debug.visualize_query import main as visualize_query_entry
# from td3fd.ddpg2.debug.check_potential import main as check_potential_entry

try:
  from mpi4py import MPI
except ImportError:
  MPI = None

# Tensorflow environment setup
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')


def import_param_config(load_dir):
  """Assume that there is a gv called params_config that contains all the params
    """
  # get the full path of the load directory
  abs_load_dir = os.path.abspath(os.path.expanduser(load_dir))
  spec = importlib.util.spec_from_file_location("module.name", abs_load_dir)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  params_configs = [
      getattr(module, params_config)
      for params_config in dir(module)
      if params_config.startswith("params_config")
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


def get_search_params(param_config):
  """Returns a list of parameter keys we want to grid-search for and a dict of 
  grid-search values.
  """

  def update_dictionary(dictionary, key, value):  # without overwriting keys
    while key in dictionary.keys():
      key += "."
    dictionary[key] = value

  search_params_list = []
  search_params_dict = collections.OrderedDict()
  for param_name, param_val in param_config.items():
    if type(param_val) == tuple:
      search_params_list += [param_name]
      update_dictionary(search_params_dict, param_name,
                        tune.grid_search(list(param_val)))
    elif type(param_val) == dict:
      search_params_list += get_search_params(param_val)[0]
      for k, v in get_search_params(param_val)[1].items():
        update_dictionary(search_params_dict, k, v)
  return search_params_list, search_params_dict


def transform_config_name(config_name):
  """ Transfer the legend names"""
  for i in range(len(config_name)):
    if config_name[i].startswith("config"):
      if config_name[i] == "config_TD3_NF_Shaping":
        for name in config_name:
          if name.startswith("potential_weight_"):
            weight = name.replace("potential_weight_", "")
            return ["TD3+Shaping(NF), $k^{NF}$=" + weight]
        return ["TD3+Shaping(NF)"]
      elif config_name[i] == "config_TD3_NF_Shaping_Decay":
        for name in config_name:
          if name.startswith("potential_weight_"):
            weight = name.replace("potential_weight_", "")
            return ["TD3+Shaping(NF) w/ Decay, $k^{NF}$=" + weight]
        return ["TD3+Shaping(NF) w/ Decay"]
      elif config_name[i] == "config_TD3_GAN_Shaping":
        for name in config_name:
          if name.startswith("potential_weight_"):
            weight = name.replace("potential_weight_", "")
            return ["TD3+Shaping(GAN), $k^{GAN}$=" + weight]
        return ["TD3+Shaping(GAN)"]
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


def main(targets, exp_dir, policy, save_dir, num_cpus, num_gpus, ip_head,
         redis_password, **kwargs):

  # Consider rank as pid.
  comm = MPI.COMM_WORLD if MPI is not None else None
  total_num_cpu = comm.Get_size() if comm is not None else 1
  rank = comm.Get_rank() if comm is not None else 0

  # Use the default logger setting
  logger.configure()

  # Setup
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
          varied_params = k[len(exp_dir) + 1:].split("/")
          config_name = [
              x for x in varied_params
              if not any([x.startswith(y) for y in exc_params])
          ]
          print(config_name)
          config_name = transform_config_name(config_name)
          print("||||---> ", config_name)
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
            mpi_util.mpi_exit(1, comm=comm)
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

      # run experiments
      parallel = 1  # CHANGE this number to allow launching in serial
      # total num exps
      num_exp = len(dir_param_dict.keys())

      if comm is None:
        logger.info("No MPI provided. Launching scripts in series.")
        for k in dir_param_dict.keys():
          train.main(root_dir=k)

      elif not parallel:
        logger.info("{} experiments in series.".format(num_exp))
        comm_for_exps = comm.Split(color=int(rank > 0))
        if rank == 0:
          assert comm_for_exps.Get_size() == 1
          for k in dir_param_dict.keys():
            train.main(root_dir=k, comm=comm_for_exps)
      else:
        logger.info("{} experiments in parallel.".format(num_exp))
        assert num_exp <= total_num_cpu, "no enough cpu! need {} cpus".format(
            num_exp)
        # select processes to run the experiment, which is [0, num_exp]
        comm_for_exps = comm.Split(color=int(rank >= num_exp))
        if rank < num_exp:
          assert comm_for_exps.Get_size() == num_exp
          comm_for_each_exp = comm_for_exps.Split(
              color=comm_for_exps.Get_rank())
          assert comm_for_each_exp.Get_size() == 1
          k = list(dir_param_dict.keys())
          train.main(root_dir=k[comm_for_exps.Get_rank()],
                     comm=comm_for_each_exp)

      if comm is not None:
        comm.Barrier()

    elif "tune:" in target:
      logger.info("\n\n=================================================")
      logger.info("Launching the training experiment (with ray.tune)!")
      logger.info("=================================================")
      # adding checking
      config_file = target.replace("tune:", "")
      params_configs = import_param_config(config_file)
      assert len(params_configs) == 1, "Only support 1 config now."
      params_config = params_configs[0]
      dir_param_dict = generate_params(exp_dir, params_config)
      config_name = params_config.pop("config")
      config_name = config_name[0] if type(
          config_name) == tuple else config_name
      search_params_list, search_params_dict = get_search_params(params_config)
      # store json config file to the target directory
      if rank == 0:
        for k, v in dir_param_dict.items():
          if os.path.exists(k):  # COMMENT out this check for restarting
            logger.info(
                "Directory {} already exists! Remove it? [Y/n]".format(k))
            remove = input()
            if remove != "y":
              mpi_util.mpi_exit(1, comm=comm)
            shutil.rmtree(k)
          os.makedirs(k, exist_ok=True)
          # copy params.json file
          with open(os.path.join(k, "params.json"), "w") as f:
            json.dump(v, f)
          # copy demo_sata file if exist
          demo_file = os.path.join(exp_dir, "demo_data.npz")
          demo_dest = os.path.join(k, "demo_data.npz")
          if os.path.isfile(demo_file):
            shutil.copyfile(demo_file, demo_dest)
        ray.init(num_cpus=num_cpus if not ip_head else None,
                 num_gpus=num_gpus if not ip_head else None,
                 address=ip_head,
                 redis_password=redis_password)
        tune.run(train_tune.main,
                 verbose=1,
                 local_dir=os.path.join(exp_dir, "config_" + config_name),
                 resources_per_trial={
                     "cpu": 1,
                     "gpu": num_gpus / num_cpus,
                 },
                 config=dict(root_dir=exp_dir,
                             config=config_name,
                             search_params_list=search_params_list,
                             **search_params_dict))

    elif target == "demo":
      assert policy != None
      logger.info("\n\n=================================================")
      logger.info(
          "Using policy file from {} to generate demo data.".format(policy))
      logger.info("=================================================")
      if rank == 0:
        generate_demo.main(policy=policy, root_dir=exp_dir)

    elif target == "evaluate":
      assert policy != None
      logger.info("\n\n=================================================")
      logger.info("Evaluating using policy file from {}.".format(policy))
      logger.info("=================================================")
      if rank == 0:
        evaluate.main(policy=policy)

    elif target == "plot":
      logger.info("\n\n=================================================")
      logger.info("Plotting.")
      logger.info("=================================================")
      if rank == 0:  # plot does not need to be run on all threads
        plot.main(
            dirs=[exp_dir],
            save_dir=save_dir,
            xys=[
                # "train/steps:test/reward_per_eps",
                "Testing/AverageReturn vs EnvironmentSteps",
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

    # elif target == "check":
    #   logger.info("\n\n=================================================")
    #   logger.info("Check potential.")
    #   logger.info("=================================================")
    #   if rank == 0:
    #     check_potential_entry(exp_dir=exp_dir)

    else:
      assert False, "Unknown target: {}".format(target)

    if comm is not None:
      comm.Barrier()

    logger.reset()


if __name__ == "__main__":

  from rlfd.utils.cmd_util import ArgParser

  exp_parser = ArgParser(allow_unknown_args=False)
  exp_parser.parser.add_argument(
      "--exp_dir",
      help="top level directory to store experiment results",
      type=str,
      default=os.getcwd())
  exp_parser.parser.add_argument("--save_dir",
                                 help="top level directory to store plots",
                                 type=str,
                                 default=os.getcwd())
  exp_parser.parser.add_argument(
      "--targets",
      help="list of targets: [demo, train:<parameter file>.py, plot, evaluate]",
      type=str,
      nargs="+",
  )
  exp_parser.parser.add_argument(
      "--policy",
      help="The policy file to be used for evaluate or demo, <policy name>.pkl",
      type=str,
      default=None,
  )
  exp_parser.parser.add_argument(
      "--num_cpus",
      help="ray.init num_cpus",
      type=int,
      default=1,
  )
  exp_parser.parser.add_argument(
      "--num_gpus",
      help="ray.init num_gpus",
      type=int,
      default=0,
  )
  # Options below are cluster specific
  exp_parser.parser.add_argument(
      "--ip_head",
      help="ray.init address",
      type=str,
      default=None,
  )
  exp_parser.parser.add_argument(
      "--redis_password",
      help="ray.init redis_password",
      type=str,
      default=None,
  )
  exp_parser.parse(sys.argv)
  main(**exp_parser.get_dict())
