from copy import deepcopy
from td3fd.rlkit_td3.params.reach2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3",)
params_config["demo_strategy"] = "none"
params_config["seed"] = (0, 1)
# params_config["env_name"] = "Reach2DFDense"
