import argparse
import collections
import copy
import importlib
import json
import os
import shutil
import sys
osp = os.path

import mujoco_py  # include at the beginning (bug on compute canada cluster)
import tensorflow as tf
import ray
from ray import tune

# Tensorflow environment setup
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

from rlfd import logger, plot, train, evaluate
from rlfd.demo_utils import generate_demo


def import_param_config(load_dir):
  """Assume that there is a gv called params_config that contains all the params
    """
  # get the full path of the load directory
  abs_load_dir = os.path.abspath(os.path.expanduser(load_dir))
  spec = importlib.util.spec_from_file_location("module.name", abs_load_dir)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
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
  """Change the legend names to whatever you want it to be."""
  return config_name


def main(targets, exp_dir, policy, save_dir, num_cpus, num_gpus, ip_head,
         redis_password, **kwargs):
  # Setup
  exp_dir = os.path.abspath(os.path.expanduser(exp_dir))
  if policy is not None:
    policy = os.path.abspath(os.path.expanduser(policy))

  for target in targets:
    if "rename:" in target:
      print("\n\n=================================================")
      print("Renaming the config to params_renamed.json!")
      print("=================================================")
      # excluded params
      exc_params = []
      exc_params = ["seed"]  # CHANGE this!
      # adding checking
      config_file = target.replace("rename:", "")
      params_config = import_param_config(config_file)
      dir_param_dict = generate_params(exp_dir, params_config)
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
      print("\n\n=================================================")
      print("Launching the training experiment!")
      print("=================================================")
      # adding checking
      config_file = target.replace("train:", "")
      params_config = import_param_config(config_file)
      dir_param_dict = generate_params(exp_dir, params_config)
      config_name = params_config.pop("config")
      config_name = config_name[0] if type(
          config_name) == tuple else config_name
      search_params_list, search_params_dict = get_search_params(params_config)
      # store json config file to the target directory
      overwrite_all = remove_all = False
      for k, v in dir_param_dict.items():
        if os.path.exists(k) and (not overwrite_all and not remove_all):
          resp = input("Directory {} exists! (R)emove/(O)verwrite?".format(k))
          if resp.lower() == "r":
            remove_all = True
          elif resp.lower() == "o":
            overwrite_all = True
          else:
            exit(1)
        if remove_all:
          shutil.rmtree(k)
        os.makedirs(k, exist_ok=True)
        # copy params.json file
        with open(os.path.join(k, "params.json"), "w") as f:
          json.dump(v, f)
        # copy demo_sata file if exist
        for f in ["demo_data.npz", "demo_policy.pkl", "shaping.pkl"]:
          source = os.path.join(exp_dir, f)
          destination = os.path.join(k, f)
          if os.path.isfile(source):
            shutil.copyfile(source, destination)
      ray.init(num_cpus=num_cpus if not ip_head else None,
               num_gpus=num_gpus if not ip_head else None,
               address=ip_head,
               redis_password=redis_password,
               temp_dir=osp.join(osp.expanduser("~"), ".ray"))
      tune.run(train.main,
               verbose=1,
               local_dir=os.path.join(exp_dir, "config_" + config_name),
               resources_per_trial={
                   "cpu": 1,
                   "gpu": num_gpus / num_cpus,
               },
               config=dict(root_dir=exp_dir,
                           config=config_name,
                           search_params_list=search_params_list,
                           **search_params_dict),
               progress_reporter=tune.CLIReporter(
                   ["mode", "epoch", "time_total_s"]))

    elif target == "demo":
      assert policy != None
      print("\n\n=================================================")
      print("Using policy file from {} to generate demo data.".format(policy))
      print("=================================================")
      generate_demo.main(policy=policy, root_dir=exp_dir)

    elif target == "evaluate":
      assert policy != None
      print("\n\n=================================================")
      print("Evaluating using policy file from {}.".format(policy))
      print("=================================================")
      evaluate.main(policy=policy)

    elif target == "plot":
      print("\n\n=================================================")
      print("Plotting.")
      print("=================================================")
      plot.main(
          dirs=[exp_dir],
          save_dir=save_dir,
          xys=[
              "OnlineTesting/AverageReturn vs EnvironmentSteps",
              # "OfflineTesting/AverageReturn",
          ],
          smooth=True,
      )

    else:
      raise ValueError("Unknown target: {}".format(target))


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
