from copy import deepcopy
from td3fd.ddpg2.params.reach2d import params_config as base_params

params_config = deepcopy(base_params)

params_config["config"] = ("BC",)

params_config["ddpg"]["demo_strategy"] = "none"
params_config["ddpg"]["sample_demo_buffer"] = True

params_config["ddpg"]["use_n_step_return"] = False

# initialize the policy using behavior cloning
params_config["ddpg"]["initialize_with_bc"] = True
params_config["ddpg"]["initialize_num_epochs"] = 2000

params_config["ddpg"]["num_batches"] = 0 # do not further train with RL

# BC
params_config["seed"] = tuple(range(4))
