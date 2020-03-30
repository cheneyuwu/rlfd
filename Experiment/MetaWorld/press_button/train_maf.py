from copy import deepcopy
from td3fd.ddpg2.params.metaworld import params_config as base_params

params_config = deepcopy(base_params)

params_config["config"] = ("TD3_NF_Shaping",)

params_config["env_name"] = ("button-press_topdown-v1",)

params_config["ddpg"]["demo_strategy"] = "nf"
params_config["ddpg"]["sample_demo_buffer"] = True

params_config["ddpg"]["use_n_step_return"] = True

# initialize the policy using behavior cloning
params_config["ddpg"]["initialize_with_bc"] = True
params_config["ddpg"]["initialize_num_epochs"] = 2000

# TD3_NF_Shaping
# params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (200.0, 400.0)
# params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (1.0, 5.0, 10.0)
# params_config["seed"] = tuple(range(4))
# TD3_NF_Shaping_2
# params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (2e3, 1e3, 5e2)
# params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (5.0, 10.0, 20.0)
# params_config["seed"] = tuple(range(4))
# TD3_NF_Shaping_3 3 seeds
# params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (1e3,)
# params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (1.0, 3.0)
# params_config["ddpg"]["shaping_params"]["num_epochs"] = (int(1e4), int(6e3))
# params_config["seed"] = tuple(range(4))
# TD3_NF_Shaping_4 4 seeds
# params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (1e3,)
# params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (5.0,)
# params_config["ddpg"]["shaping_params"]["num_epochs"] = (int(1e4), int(6e3))
# params_config["seed"] = tuple(range(4))

# TD3_NF_Shaping_2_Decay
# params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (2e3, 1e3, 5e2)
# params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (5.0, 10.0, 20.0)
# params_config["ddpg"]["shaping_params"]["num_epochs"] = (int(1e4),)
# params_config["seed"] = tuple(range(2))

# TD3_NF_Shaping_2_Decay Local
# params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (1e3,)
# params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (5.0, 10.0)
# params_config["ddpg"]["shaping_params"]["num_epochs"] = (int(1e4),)
# params_config["seed"] = tuple(range(2))


params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (1e3,)
params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (5.0,)
params_config["ddpg"]["shaping_params"]["num_epochs"] = (int(1e1),)
params_config["ddpg"]["shaping_params"]["potential_decay_scale"] = 0.5
params_config["ddpg"]["shaping_params"]["potential_decay_epoch"] = 3
params_config["seed"] = tuple(range(1))
