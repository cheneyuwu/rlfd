from copy import deepcopy
from td3fd.td3.params.ddpg_fetch_peg_in_hole_2d import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = "TD3"
params_config["demo_strategy"] = ("none",)
params_config["env_name"] = ("YWFetchPegInHole2D-v0",)
params_config["seed"] = 0
