# Best parameters found so far to be used for the Open AI fetch pick and place environment with a single goal.
params_config = {
    # config summary
    "config": "default",
    # environment config
    "env_name": "YWFetchPickAndPlaceRandInit-v0",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 40,
    "env_args": {},
    "gamma": 0.99,
    "fix_T": True,
    # DDPG config
    "ddpg": {
        # replay buffer setup
        "buffer_size": int(1e6),
        # actor critic networks
        "scope": "ddpg",
        "use_td3": True,
        "layer_sizes": [256, 256, 256],
        "initializer_type": "glorot",  # ["zero", "glorot"]
        "q_lr": 0.001,
        "pi_lr": 0.001,
        "action_l2": 0.4,
        "batch_size": 256,
        # double q learning
        "polyak": 0.95,
        # use demonstrations
        "sample_demo_buffer": False,
        "batch_size_demo": 128,
        "use_demo_reward": False,
        "num_demo": 50,
        "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
        "bc_params": {"q_filter": True, "prm_loss_weight": 1.0, "aux_loss_weight": 1.0},
        "shaping_params": {
            "batch_size": 128,
            "nf": {
                "num_ens": 2,
                "nf_type": "maf",  # ["maf", "realnvp"]
                "lr": 2e-4,
                "num_masked": 4,
                "num_bijectors": 4,
                "layer_sizes": [256, 256],
                "initializer_type": "glorot",  # ["zero", "glorot"]
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 1000.0,
                "potential_weight": 3.0,
            },
            "gan": {
                "num_ens": 4,
                "layer_sizes": [256, 256, 256],
                "initializer_type": "glorot",  # ["zero", "glorot"]
                "latent_dim": 25,
                "gp_lambda": 0.1,
                "critic_iter": 5,
                "potential_weight": 0.5,
            },
        },
        # normalize observation
        "norm_eps": 0.01,
        "norm_clip": 5,
    },
    # rollouts config
    "rollout": {
        "num_episodes": 4,
        "num_steps": None,
        "noise_eps": 0.2,
        "polyak_noise": 0.0,
        "random_eps": 0.2,
        "compute_q": False,
        "history_len": 10,
    },
    "evaluator": {
        "num_episodes": 10,
        "num_steps": None,
        "noise_eps": 0.1,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": True,
        "history_len": 1,
    },
    # training config
    "train": {
        "n_epochs": int(4e3),
        "n_cycles": 10,
        "n_batches": 40,
        "shaping_n_epochs": int(2e4),
        "pure_bc_n_epochs": int(1e3),
    },
    "seed": tuple(range(2)),
}