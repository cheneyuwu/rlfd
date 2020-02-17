from copy import deepcopy
from td3fd.rlkit_sac.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC",)
params_config["env_name"] = "hammer-v1"
params_config["demo_strategy"] = "none"
params_config["seed"] = tuple(range(5))