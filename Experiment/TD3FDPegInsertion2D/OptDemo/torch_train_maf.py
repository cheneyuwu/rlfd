from copy import deepcopy
from td3fd.ddpg.param.ddpg_fetch_peg_in_hole_2d import params_config as base_params

# maf training
# params_config_maf = deepcopy(base_params)
# params_config_maf["ddpg"]["demo_strategy"] = ("nf",)
# params_config_maf["ddpg"]["sample_demo_buffer"] = 1
# params_config_maf["train"]["shaping_n_epochs"] = int(1e4)
# params_config_maf["ddpg"]["shaping_params"]["nf"]["num_bijectors"] = (3,)
# params_config_maf["ddpg"]["shaping_params"]["nf"]["layer_sizes"] = ([64, 64],)
# params_config_maf["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (200.0,)
# params_config_maf["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (3.0,)
# params_config_maf["seed"] = tuple(range(8))

# maf plotting
params_config_maf = deepcopy(base_params)
params_config_maf["ddpg"]["demo_strategy"] = ("nf",)
params_config_maf["ddpg"]["sample_demo_buffer"] = 1
params_config_maf["train"]["shaping_n_epochs"] = int(1e4)
params_config_maf["ddpg"]["shaping_params"]["nf"]["num_bijectors"] = (3,)
params_config_maf["ddpg"]["shaping_params"]["nf"]["layer_sizes"] = ([64, 64],)
params_config_maf["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (200.0,)
params_config_maf["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (3.0,)
params_config_maf["seed"] = tuple(range(4))