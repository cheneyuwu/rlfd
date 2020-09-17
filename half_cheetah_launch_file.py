from copy import deepcopy
from rlfd.params.mbpo  import gym_mujoco_params  # import default parameters for gym_mujoco environments
# the launch file defines a global dict called params_config
params_config = deepcopy(gym_mujoco_params)  # get default parameters of an algorithm.
params_config["config"] = ("MBPO", )  # provide your exp config name, will be used for plotting.
params_config["env_name"] = "halfcheetah-medium-v0"  # which environment
# Make whatever changes to the default parameters.
params_config["seed"] = tuple(range(1))  # a tuple means grid search.
