params_config = {
    # config summary
    "alg": "old-ddpg-tf",
    "config": "default",
    # environment config
    "env_name": "YWFetchPickAndPlaceRandInit-v0",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 40,
    "env_args": {},
    "gamma": None,
    "fix_T": True,
    # DDPG config
    "ddpg": {
        "num_epochs": int(4e3),
        "num_cycles": 10,
        "num_batches": 40,
        "batch_size": 256,
        # use demonstrations
        "batch_size_demo": 128,
        "sample_demo_buffer": False,
        "use_demo_reward": False,
        "num_demo": 50,
        "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
        # normalize observation
        "norm_eps": 0.01,
        "norm_clip": 5,
        # actor critic networks
        "scope": "ddpg",
        "layer_sizes": [256, 256, 256],
        "twin_delayed": True,
        "policy_freq": 2,
        "policy_noise": 0.1,
        "policy_noise_clip": 0.5,
        "q_lr": 0.001,
        "pi_lr": 0.001,
        "action_l2": 0.4,
        # double q learning
        "polyak": 0.95,
        "bc_params": {"q_filter": False, "prm_loss_weight": 1.0, "aux_loss_weight": 1.0},
        "shaping_params": {
            "num_epochs": int(1e4),
            "batch_size": 128,
            "num_ensembles": 2,
            "nf": {
                "num_masked": 4,
                "num_bijectors": 4,
                "layer_sizes": [256, 256],
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 400.0,
                "potential_weight": 10.0,
            },
            "gan": {
                "layer_sizes": [256, 256, 256],
                "latent_dim": 25,
                "gp_lambda": 0.1,
                "critic_iter": 5,
                "potential_weight": 0.5,
            },
        },
        # replay buffer setup
        "buffer_size": int(5e5),
    },
    # rollouts config
    "rollout": {
        "num_episodes": 4,
        "num_steps": None,
        "noise_eps": 0.2,
        "polyak_noise": 0.0,
        "random_eps": 0.2,
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
    "seed": tuple(range(2)),
}
