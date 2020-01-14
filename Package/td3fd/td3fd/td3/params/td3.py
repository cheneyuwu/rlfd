import numpy as np

params_config = {
    # config summary
    "alg": "ddpg-torch",
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
    "norm_obs": False,  # whethere or not to normalize observations
    "norm_eps": 0.01,  # epsilon used for observation normalization
    "norm_clip": 5,  # normalized observations are cropped to this value
    # ddpg training
    "num_demo": 20,
    "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
    # DDPG config
    "ddpg": {
        "num_epochs": int(1e3),
        "num_cycles": 500,
        "num_batches": 1,
        "batch_size": 256,
        "batch_size_demo": 128,
        # actor critic networks
        "layer_sizes": [256, 256],
        "twin_delayed": True,
        "policy_freq": 2,
        "policy_noise": 0.2,
        "policy_noise_clip": 0.5,
        "q_lr": 3e-4,
        "pi_lr": 3e-4,
        "action_l2": 0.0,
        # double q learning
        "polyak": 0.995,
        "bc_params": {"q_filter": False, "prm_loss_weight": 1.0, "aux_loss_weight": 1.0},
    },
    "shaping": {
        "num_epochs": int(30),
        "batch_size": 128,
        "nf": {
            "num_blocks": 4,
            "num_hidden": 100,
            "prm_loss_weight": 1.0,
            "reg_loss_weight": 200.0,
            "potential_weight": 500.0,
        },
        "gan": {
            "layer_sizes": [256, 256, 256],
            "potential_weight": 3.0,
        },
    },
    "memory": {
        # replay buffer setup
        "buffer_size": int(1e6),
    },
    # rollouts config
    "rollout": {
        "num_episodes": None,
        "num_steps": 1,
        "noise_eps": 0.1,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": False,
        "history_len": 300,
    },
    "evaluator": {
        "num_episodes": 4,
        "num_steps": None,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 300,
    },
    "seed": 0,
}
