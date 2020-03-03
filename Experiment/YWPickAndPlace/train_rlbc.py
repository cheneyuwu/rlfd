from copy import deepcopy
from td3fd.ddpg.params.pickandplacerandinit import params_config as base_params

# td3fd via direct behavior cloning
params_config = deepcopy(base_params)
params_config["config"] = ("TD3_BC",)
params_config["ddpg"]["demo_strategy"] = "bc"
params_config["ddpg"]["sample_demo_buffer"] = True

# bc
params_config["ddpg"]["bc_params"]["q_filter"] = (False,)
params_config["ddpg"]["bc_params"]["prm_loss_weight"] = (1e-4, 1e-2)
params_config["seed"] = tuple(range(4))
