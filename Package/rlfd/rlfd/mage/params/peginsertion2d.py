from copy import deepcopy
from .default_params import parameters

params_config = deepcopy(parameters)
params_config["env_name"] = "YWFetchPegInHole2D-v0"
params_config["fix_T"] = True
params_config["random_expl_num_cycles"] = 10
params_config["num_epochs"] = int(4e2)
params_config["num_cycles_per_epoch"] = 10
params_config["num_batches_per_cycle"] = 40
params_config["expl_num_episodes_per_cycle"] = 4
params_config["expl_num_steps_per_cycle"] = None
params_config["agent"]["gamma"] = 0.975
params_config["agent"]["online_batch_size"] = 256
params_config["agent"]["layer_sizes"] = [256, 256]
params_config["agent"]["polyak"] = 0.95
params_config["agent"]["target_update_freq"] = 40
params_config["agent"]["model_update_interval"] = 400
params_config["agent"]["shaping_params"]["num_epochs"] = int(1e4)
params_config["agent"]["shaping_params"]["nf"]["reg_loss_weight"] = 4e2
params_config["agent"]["shaping_params"]["nf"]["potential_weight"] = 1e1