from .rl import params_config as base_params

params_config = base_params.copy()
params_config["config"] = ("RLwithDemoInRB",)
params_config["ddpg"]["demo_strategy"] = "rb"