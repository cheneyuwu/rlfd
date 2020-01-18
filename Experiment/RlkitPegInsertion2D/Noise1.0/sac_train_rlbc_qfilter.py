from copy import deepcopy
from td3fd.rlkit_sac.params.peginhole2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("SAC_BC_QFilter",)
params_config["demo_strategy"] = "bc"
params_config["trainer_kwargs"]["q_filter"] = True
params_config["seed"] = tuple(range(2))