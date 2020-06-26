from copy import deepcopy
from rlfd.td3.params.gym_mujoco import params_config as base_params

params_config = deepcopy(base_params)

params_config["config"] = ("TD3",)

params_config["env_name"] = "HalfCheetah-v2"  # "InvertedPendulum-v2"

params_config["ddpg"]["demo_strategy"] = "none"
params_config["ddpg"]["sample_demo_buffer"] = False

params_config["ddpg"]["use_n_step_return"] = False

# do not initialize the policy using behavior cloning
params_config["ddpg"]["initialize_with_bc"] = False
params_config["ddpg"]["initialize_num_epochs"] = 2000

# TD3
params_config["seed"] = tuple(range(5))