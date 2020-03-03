from copy import deepcopy
from td3fd.ddpg.params.peginsertionrandinit import params_config as base_params

# td3
params_config = deepcopy(base_params)
params_config["config"] = ("TD3",)
params_config["ddpg"]["demo_strategy"] = "none"
params_config["ddpg"]["sample_demo_buffer"] = False

#
params_config["seed"] = tuple(range(4))
