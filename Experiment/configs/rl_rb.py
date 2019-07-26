from .rl import params_config as base_params

params_config = base_params.copy()
params_config["config"] = ("RL_with_Demo_in_RB",)
params_config["ddpg"]["sample_demo_buffer"] = 1
