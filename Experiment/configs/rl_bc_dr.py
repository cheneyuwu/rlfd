from .rl import params_config as base_params

params_config = base_params.copy()
params_config["config"] = ("RL_with_BC_use_DemoReward",)
params_config["ddpg"]["demo_strategy"] = "bc"
params_config["ddpg"]["sample_demo_buffer"] = 1
params_config["ddpg"]["use_demo_reward"] = 1
