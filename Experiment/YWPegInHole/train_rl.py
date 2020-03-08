from copy import deepcopy
from td3fd.ddpg2.params.peginsertionrandinit import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3",)
params_config["ddpg"]["demo_strategy"] = "none"
params_config["ddpg"]["sample_demo_buffer"] = False

# not initialize the policy using behavior cloning
params_config["ddpg"]["initialize_with_bc"] = False
params_config["ddpg"]["initialize_num_epochs"] = 2000

params_config["seed"] = tuple(range(4))
