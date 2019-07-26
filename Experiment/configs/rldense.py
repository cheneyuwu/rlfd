from .rl import params_config as base_params

params_config = base_params.copy()
params_config["config"] = ("RL_with_Dense_Rwd",)
params_config["env_name"] = "FetchPickAndPlaceDense-v1"
params_config["train"]["n_epochs"] = 500
params_config["seed"] = 0
