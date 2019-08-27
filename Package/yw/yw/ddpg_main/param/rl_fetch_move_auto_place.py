# Best parameters found so far to be used for the Open AI fetch pick and place environment with a single goal and two objects.
params_config = {
    # config summary
    "config": "default",
    # environment config
    "env_name": "FetchMoveAutoPlace-v1",  # Important: make sure that you have 2 objects
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 80,
    "env_args": {},
    "fix_T": True,
    # DDPG config
    "ddpg": {
        # replay buffer setup
        "buffer_size": int(1e6),
        "replay_strategy": "none",  # ["her", "none"] (her for hindsight exp replay)
        # actor critic networks
        "scope": "ddpg",
        "use_td3": 1,
        "layer_sizes": [256, 256, 256],
        "initializer_type": "glorot",  # ["zero", "glorot"]
        "Q_lr": 0.001,
        "pi_lr": 0.001,
        "action_l2": 0.4,
        "batch_size": 256,
        # double q learning
        "polyak": 0.95,
        # use demonstrations
        "sample_demo_buffer": 0,
        "batch_size_demo": 128,
        "use_demo_reward": 0,
        "num_demo": 50,
        "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
        "bc_params": {"pure_bc": False, "q_filter": 1, "prm_loss_weight": 1.0, "aux_loss_weight": 10.0},
        "shaping_params": {
            "batch_size": 128,
            "nf": {
                "num_ens": 1,
                "nf_type": "maf",  # ["maf", "realnvp"]
                "lr": 5e-4,
                "num_masked": 6,
                "num_bijectors": 6,
                "layer_sizes": [512, 512],
                "initializer_type": "glorot",  # ["zero", "glorot"]
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 2000.0,
                "potential_weight": 3.0,
            },
            "gan": {
                "num_ens": 1,
                "layer_sizes": [256, 256, 256],
                "initializer_type": "glorot",  # ["zero", "glorot"]
                "latent_dim": 40,  # use the same number of dims as obs
                "gp_lambda": 0.1,
                "critic_iter": 5,
                "potential_weight": 4.0,
            },
        },
        # normalize observation
        "norm_eps": 0.01,
        "norm_clip": 5,
        # i/o clippings
        "clip_obs": 200.0,
        "clip_pos_returns": False,
        "clip_return": False,
    },
    # HER config
    "her": {"k": 4},
    # rollouts config
    "rollout": {
        "rollout_batch_size": 4,
        "noise_eps": 0.2,
        "polyak_noise": 0.0,
        "random_eps": 0.1,
        "compute_Q": False,
        "history_len": 10,
    },
    "evaluator": {
        "rollout_batch_size": 20,
        "noise_eps": 0.05,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_Q": True,
        "history_len": 1,
    },
    # training config
    "train": {
        "n_epochs": int(4e3),
        "n_cycles": 10,
        "n_batches": 40,
        "shaping_n_epochs": int(1e4),
        "save_interval": 10,
    },
    "seed": tuple(range(2)),
}
