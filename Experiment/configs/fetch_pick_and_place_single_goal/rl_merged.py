from yw.ddpg_main.param.rl_fetch_pick_and_place_single_goal import params_config as base_params

params_config = base_params.copy()
params_config["config"] = "default"
params_config["seed"] = tuple(range(20))
params_config["ddpg"]["demo_strategy"] = ("none",)
# params_config["ddpg"]["demo_strategy"] = ("bc", "maf")
params_config["ddpg"]["sample_demo_buffer"] = 1
