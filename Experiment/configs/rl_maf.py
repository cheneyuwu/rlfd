from .rl import params_config as base_params

params_config = base_params.copy()
params_config["config"] = ("RL_with_MAF",)
params_config["ddpg"]["demo_strategy"] = "maf"
params_config["ddpg"]["sample_demo_buffer"] = 1
# params_config["ddpg"]["shaping_params"]["reg_loss_weight"] = (800.0, 1000)
# params_config["ddpg"]["shaping_params"]["potential_weight"] = (4.0, 3.0)
