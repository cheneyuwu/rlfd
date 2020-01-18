from copy import deepcopy
from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_2d import params_config as base_params

# pure bc
params_config_bc = deepcopy(base_params)
params_config_bc["ddpg"]["demo_strategy"] = ("pure_bc",)
params_config_bc["ddpg"]["sample_demo_buffer"] = 1
params_config_bc["seed"] = tuple(range(20))

# 80 cpus