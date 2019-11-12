from copy import deepcopy
from td3fd.ddpg.param.ddpg_reacher import params_config as base_params

# td3fd via normalizing flow (MAF) shaping
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_MAF_Shaping",)
params_config["ddpg"]["demo_strategy"] = "nf"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["seed"] = tuple(range(1))
