from copy import deepcopy
from td3fd.ddpg2.params.pickandplacerandinit import params_config as base_params

params_config = deepcopy(base_params)

params_config["config"] = ("TD3_GAN_Shaping",)

params_config["ddpg"]["demo_strategy"] = "gan"
params_config["ddpg"]["sample_demo_buffer"] = True

params_config["ddpg"]["use_n_step_return"] = False

# initialize the policy using behavior cloning
params_config["ddpg"]["initialize_with_bc"] = True
params_config["ddpg"]["initialize_num_epochs"] = 2000

# TD3_GAN_Shaping
params_config["ddpg"]["shaping_params"]["gan"]["gp_lambda"] = (1.0,)
params_config["ddpg"]["shaping_params"]["gan"]["potential_weight"] = (1e-1,)
params_config["ddpg"]["shaping_params"]["num_epochs"] = (int(1e4),)
params_config["seed"] = tuple(range(4))
