from copy import deepcopy
from td3fd.ddpg.param.ddpg_reacher import params_config as base_params

# td3fd via direct behavior cloning
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC",)
params_config["ddpg"]["demo_strategy"] = "bc"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["seed"] = 0
