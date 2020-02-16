from copy import deepcopy
from td3fd.ddpg.params.metaworld import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("TD3_MAF_Shaping",)
params_config["env_name"] = ("pick-place-v1",)

# use maf shaping
params_config["ddpg"]["demo_strategy"] = "nf"
params_config["ddpg"]["sample_demo_buffer"] =  True

# use multi step return
params_config["ddpg"]["use_n_step_return"] = True

# initialize the policy using behavior cloning
# params_config["ddpg"]["initialize_with_bc"] = True
# params_config["ddpg"]["initialize_num_epochs"] = 2000

# maf shaping methods
params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (400.0,)
params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (10.0,)
params_config["train"]["shaping_n_epochs"] = (int(1e4),)
params_config["seed"] = tuple(range(5))
