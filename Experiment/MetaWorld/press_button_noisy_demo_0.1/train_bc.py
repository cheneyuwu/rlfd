from copy import deepcopy
from td3fd.ddpg.params.metaworld import params_config as base_params

params_config = deepcopy(base_params)

params_config["config"] = ("TD3_BC_1e-2",)

params_config["env_name"] = ("button-press_topdown-v1",)

params_config["ddpg"]["demo_strategy"] = "bc"
params_config["ddpg"]["sample_demo_buffer"] = True

params_config["ddpg"]["use_n_step_return"] = True

# initialize the policy using behavior cloning
params_config["ddpg"]["initialize_with_bc"] = True
params_config["ddpg"]["initialize_num_epochs"] = 2000

params_config["ddpg"]["bc_params"]["q_filter"] = False
params_config["ddpg"]["bc_params"]["prm_loss_weight"] = 1e-2
params_config["seed"] = tuple(range(1))
