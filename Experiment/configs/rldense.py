from .rl import params_config as base_params

params_config = base_params.copy()
params_config["config"] = ("RLwithDenseRwd",)
params_config["r_shift"] = 0.0
params_config["env_name"] = "FetchPickAndPlaceDense-v1"
params_config["train"]["n_epochs"] = 500
