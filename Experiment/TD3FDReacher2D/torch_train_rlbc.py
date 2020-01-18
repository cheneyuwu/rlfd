from copy import deepcopy
from td3fd.td3.params.ddpg_reacher import params_config as base_params

# td3fd via direct behavior cloning
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC",)
params_config["demo_strategy"] = "bc"
params_config["seed"] = 0
