from copy import deepcopy
from td3fd.td3.params.metaworld import params_config as base_params

# TD3
params_config = deepcopy(base_params)
params_config["config"] = "TD3"
params_config["demo_strategy"] = ("none",)
params_config["env_name"] = ("pick-place-v1", )
params_config["seed"] = 0
