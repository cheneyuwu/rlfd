from copy import deepcopy
from td3fd.ddpg.params.metaworld import params_config as base_params

params_config = deepcopy(base_params)

params_config["config"] = ("TD3_NF_Shaping",)

params_config["env_name"] = ("pick-place-v1",)
params_config["env_args"] = {"random_init": True}

params_config["ddpg"]["demo_strategy"] = "nf"
params_config["ddpg"]["sample_demo_buffer"] = True
params_config["ddpg"]["num_demo"] = 30

params_config["ddpg"]["use_n_step_return"] = True

# initialize the policy using behavior cloning
# params_config["ddpg"]["initialize_with_bc"] = True
# params_config["ddpg"]["initialize_num_epochs"] = 2000

params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (400.0,)
params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (10.0,)
params_config["ddpg"]["shaping_params"]["num_epochs"] = (int(1e4),)

params_config["seed"] = tuple(range(1))
