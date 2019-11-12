from copy import deepcopy
from td3fd.ddpg.param.ddpg_reacher import params_config as base_params

# td3
params_config = deepcopy(base_params)
params_config["config"] = ("TD3",)
params_config["ddpg"]["demo_strategy"] = "none"
params_config["seed"] = 0
