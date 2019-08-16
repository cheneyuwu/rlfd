from yw.ddpg_main.param.rl_fetch_pick_and_place_single_goal import params_config as base_params

params_config = base_params.copy()
params_config["config"] = ("RL_with_Demo_in_RB_use_Demo_Rwd",)
params_config["ddpg"]["sample_demo_buffer"] = 1
params_config["ddpg"]["use_demo_reward"] = 1
