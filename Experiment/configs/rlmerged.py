from .rl import params_config as base_params

params_config = base_params.copy()
params_config["config"] = "default"
params_config["seed"] = tuple(range(10))
params_config["ddpg"]["demo_strategy"] = ("none", "bc", "maf", "rb", "rbmaf")
