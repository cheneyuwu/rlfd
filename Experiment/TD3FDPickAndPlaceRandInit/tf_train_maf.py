from copy import deepcopy
from td3fd.ddpg.params.pickandplacerandinit import params_config as base_params

params_config = deepcopy(base_params)
params_config["config"] = ("test_tf_MAF_2",)
params_config["ddpg"]["demo_strategy"] = ("nf",)
params_config["ddpg"]["sample_demo_buffer"] = True
# params_config["ddpg"]["shaping_params"]["nf"]["num_bijectors"] = (3,)
# params_config["ddpg"]["shaping_params"]["nf"]["layer_sizes"] = ([64, 64],)
# params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = (200.0,)
# params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = (3.0,)
params_config["seed"] = tuple(range(5))
