# yapf: disable
from copy import deepcopy

# Default Parameters
default_params = {
    # config summary
    "algo": "GAN",
    "config": "default",
    # environment config
    "env_name": "InvertedPendulum-v2",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "fix_T": False,
    # learner
    "pretrained": None,
    "offline_num_epochs": int(1e3),
    "offline_num_batches_per_epoch": 1000,
    "random_expl_num_cycles": int(0),
    "num_epochs": int(0),
    "num_cycles_per_epoch": 1,
    "num_batches_per_cycle": 1000,
    "expl_num_episodes_per_cycle": None,
    "expl_num_steps_per_cycle": 1000,
    "eval_num_episodes_per_cycle": 0,
    "eval_num_steps_per_cycle": None,
    # agent config
    "agent": {
        "offline_batch_size": 256,
        # replay buffer setup
        "buffer_size": int(1e6),
        # normalize observation
        "norm_obs_offline": True,
        "norm_eps": 0.01,
        "norm_clip": 5,
        # actor critic networks
        "layer_sizes": [256, 256],
        "latent_dim": 10,
        "gp_lambda": 0.1,
        "critic_freq": 5,
    },
    "seed": 0,
}

# OpenAI Gym
gym_mujoco_params = deepcopy(default_params)