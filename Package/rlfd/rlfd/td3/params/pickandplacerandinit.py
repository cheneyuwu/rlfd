from copy import deepcopy
from .default_params import parameters

params_config = deepcopy(parameters)
params_config["env_name"] = "YWFetchPickAndPlaceRandInit-v0"
params_config["gamma"] = None
params_config["fix_T"] = True
params_config["ddpg"]["num_epochs"] = int(6e2)
params_config["ddpg"]["num_cycles"] = 10
params_config["ddpg"]["num_batches"] = 40
params_config["ddpg"]["batch_size"] = 256
params_config["ddpg"]["layer_sizes"] = [256, 256, 256]
params_config["ddpg"]["policy_noise"] = 0.1
params_config["ddpg"]["action_l2"] = 0.4
params_config["ddpg"]["polyak"] = 0.95
params_config["ddpg"]["shaping_params"]["num_epochs"] = int(1e4)
params_config["ddpg"]["shaping_params"]["nf"]["reg_loss_weight"] = 4e2
params_config["ddpg"]["shaping_params"]["nf"]["potential_weight"] = 1e1
params_config["ddpg"]["shaping_params"]["gan"]["latent_dim"] = 16
params_config["ddpg"]["shaping_params"]["gan"]["potential_weight"] = 1.0
params_config["rollout"]["num_episodes"] = 4
params_config["rollout"]["num_steps"] = None
params_config["rollout"]["noise_eps"] = 0.2
params_config["rollout"]["random_eps"] = 0.2