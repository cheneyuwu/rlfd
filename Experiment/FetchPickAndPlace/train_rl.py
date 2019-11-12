from copy import deepcopy
from td3fd.ddpg.param.ddpg_fetch_pick_and_place_rand_init import params_config as base_params

# rl
params_config = deepcopy(base_params)
params_config["config"] = ("TD3",)
params_config["ddpg"]["demo_strategy"] = "none"
params_config["seed"] = 0