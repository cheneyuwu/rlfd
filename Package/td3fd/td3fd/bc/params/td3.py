import numpy as np

params_config = {
    # config summary
    "alg": "bc-torch",
    "config": "default",
    # environment config
    "env_name": "Reacher-v2",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "gamma": 0.99,
    "fix_T": False,
    # normalize observation
    "norm_obs": True,  # whethere or not to normalize observations
    "norm_eps": 0.01,  # epsilon used for observation normalization
    "norm_clip": 5,  # normalized observations are cropped to this value    
    # ddpg training
    "num_demo": 40,
    # DDPG config
    "bc": {
        "num_epochs": int(1e3),
        "batch_size": 256,
        # actor critic networks
        "layer_sizes": [256, 256],
        "pi_lr": 3e-4,
    },
    "memory": {
        # replay buffer setup
        "buffer_size": int(1e6),
    },
    "evaluator": {
        "num_episodes": 4,
        "num_steps": None,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": False,
        "history_len": 300,
    },
    "seed": 0,
}
