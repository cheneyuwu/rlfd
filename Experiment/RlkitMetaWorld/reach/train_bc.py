from copy import deepcopy
from td3fd.bc.params.metaworld import params_config as base_params

# TD3
params_config = deepcopy(base_params)
params_config["config"] = "BC"
params_config["env_name"] = ("reach-v1",)
params_config["seed"] = 1
