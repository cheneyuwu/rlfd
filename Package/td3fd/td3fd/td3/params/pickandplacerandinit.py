import numpy as np

params_config = {
    # config summary
    "alg": "ddpg-torch",
    "config": "default",
    # environment config
    "env_name": "YWFetchPickAndPlaceRandInit-v0",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "gamma": 0.95,
    "fix_T": True,
    # normalize observation
    "norm_obs": False,  # whethere or not to normalize observations
    "norm_eps": 0.01,  # epsilon used for observation normalization
    "norm_clip": 5,  # normalized observations are cropped to this value
    # ddpg training
    "num_demo": 40,
    "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
    # DDPG config
    "ddpg": {
        "num_epochs": int(3e3),
        "num_cycles": 10,
        "num_batches": 40,
        "batch_size": 256,
        "batch_size_demo": 128,
        # actor critic networks
        "layer_sizes": [256, 256],
        "twin_delayed": True,
        "policy_freq": 2,
        "policy_noise": 0.2,
        "policy_noise_clip": 0.5,
        "q_lr": 1e-3,
        "pi_lr": 1e-3,
        "action_l2": 0.0,
        # double q learning
        "polyak": 0.95,
        "bc_params": {"q_filter": False, "prm_loss_weight": 1.0, "aux_loss_weight": 1.0},
    },
    "shaping": {
        "num_ensembles": 1,
        "num_epochs": int(1e4),
        "batch_size": 128,
        "nf": {
            "num_blocks": 4,
            "num_hidden": 100,
            "prm_loss_weight": 1.0,
            "reg_loss_weight": 500.0,
            "potential_weight": 3.0,
        },
        "gan": {
            "latent_dim": 16,
            "lambda_term": 0.1,
            "gp_target": 1.0,
            "sub_potential_mean": False,
            "layer_sizes": [256, 256, 256],
            "potential_weight": 3.0,
        },
    },
    "memory": {
        # replay buffer setup
        "buffer_size": int(1e4),
    },
    # rollouts config
    "rollout": {
        "num_episodes": 4,
        "num_steps": None,
        "noise_eps": 0.1,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": False,
        "history_len": 300,
    },
    "evaluator": {
        "num_episodes": 10,
        "num_steps": None,
        "noise_eps": 0.05,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 300,
    },
    "seed": 0,
}
