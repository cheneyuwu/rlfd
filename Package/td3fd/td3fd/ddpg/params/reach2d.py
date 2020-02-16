params_config = {
    # config summary
    "alg": "ddpg-tf",
    "config": "default",
    # environment config
    "env_name": "Reach2DF",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "gamma": None,
    "fix_T": False,
    # DDPG config
    "ddpg": {
        "num_epochs": int(1e2),
        "num_cycles": 10,
        "num_batches": 40,
        "batch_size": 256,
        # use demonstrations
        "batch_size_demo": 128,
        "sample_demo_buffer": False,
        "initialize_with_bc": False,
        "initialize_num_epochs": 0,
        "use_demo_reward": False,
        "num_demo": 40,
        "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
        # normalize observation
        "norm_eps": 0.01,
        "norm_clip": 5,
        # actor critic networks
        "scope": "ddpg",
        "layer_sizes": [256, 256, 256],
        "twin_delayed": True,
        "policy_freq": 2,
        "policy_noise": 0.2,
        "policy_noise_clip": 0.5,
        "q_lr": 1e-3,
        "pi_lr": 1e-3,
        "action_l2": 0.4,
        # double q learning
        "polyak": 0.95,
        "bc_params": {"q_filter": False, "prm_loss_weight": 0.1, "aux_loss_weight": 1.0},
        "shaping_params": {
            "num_epochs": int(3e3),
            "batch_size": 128,
            "num_ensembles": 2,
            "nf": {
                "num_masked": 2,
                "num_bijectors": 4,
                "layer_sizes": [256, 256],
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 200.0,
                "potential_weight": 3.0,
            },
            "gan": {
                "layer_sizes": [256, 256, 256],
                "latent_dim": 6,
                "gp_lambda": 0.1,
                "critic_iter": 5,
                "potential_weight": 3.0,
            },
        },
        # replay buffer setup
        "buffer_size": int(1e6),
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
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 300,
    },
    "seed": 0,
}
