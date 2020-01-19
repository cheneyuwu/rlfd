from copy import deepcopy
from td3fd.rlkit_td3.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC",)
params_config["env_name"] = "HalfCheetah-v3"
params_config["trainer_kwargs"]["q_filter"] = False
params_config["demo_strategy"] = "bc"
params_config["seed"] = tuple(range(5))
