from copy import deepcopy
from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_2d import params_config as base_params

# rl
params_config_rl = deepcopy(base_params)
params_config_rl["ddpg"]["demo_strategy"] = ("none",)
params_config_rl["seed"] = tuple(range(20))