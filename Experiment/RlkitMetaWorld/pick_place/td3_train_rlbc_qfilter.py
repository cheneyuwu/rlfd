from copy import deepcopy
from td3fd.rlkit_td3.params.default import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC_QFilter",)
params_config["env_name"] = "pick-place-v1"
params_config["trainer_kwargs"]["q_filter"] = True
params_config["demo_strategy"] = "bc"
params_config["seed"] = tuple(range(5))