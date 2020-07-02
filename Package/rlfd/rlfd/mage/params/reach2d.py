from copy import deepcopy
from .default_params import parameters

params_config = deepcopy(default_params)
params_config["env_name"] = "Reach2DF"
params_config["gamma"] = None
params_config["fix_T"] = True
params_config["ddpg"]["num_epochs"] = int(1e2)
params_config["ddpg"]["num_cycles"] = 10
params_config["ddpg"]["num_batches"] = 40
params_config["ddpg"]["batch_size"] = 256
params_config["ddpg"]["expl_gaussian_noise"] = 0.2
params_config["ddpg"]["expl_random_prob"] = 0.2
params_config["ddpg"]["layer_sizes"] = [256, 256]
params_config["ddpg"]["action_l2"] = 0.4
params_config["ddpg"]["polyak"] = 0.95
params_config["ddpg"]["shaping_params"]["num_epochs"] = int(3e3)
params_config["ddpg"]["shaping_params"]["gan"]["layer_sizes"] = [256, 256]
params_config["ddpg"]["shaping_params"]["gan"]["latent_dim"] = 4
params_config["ddpg"]["shaping_params"]["gan"]["gp_lambda"] = 0.5
params_config["ddpg"]["shaping_params"]["gan"]["potential_weight"] = 1.0
params_config["rollout"]["num_episodes"] = 4
params_config["rollout"]["num_steps"] = None