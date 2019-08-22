# Params from the original td3 paper
params_config = {
    # config summary
    "config": "default",
    # environment config
    "env_name": "FetchPickAndPlace-v1",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
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
        "layer_sizes": [400, 300],
        "initializer_type": "glorot",  # ["zero", "glorot"]
        "Q_lr": 0.001,
        "pi_lr": 0.001,
        "action_l2": 1.0,
        "batch_size": 100,
        # double q learning
        "polyak": 0.995,
        # use demonstrations
        "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
        "sample_demo_buffer": 0,
        "use_demo_reward": 0,
        "num_demo": 0,
        "batch_size_demo": 50,
        "q_filter": 1,
        "prm_loss_weight": 0.001,
        "aux_loss_weight": 0.0078,
        "shaping_params": {
            "batch_size": 128,
            "nf": {
                "num_ens": 1,
                "nf_type": "maf",  # ["maf", "realnvp"]
                "lr": 1e-4,
                "num_masked": 2,
                "num_bijectors": 6,
                "layer_sizes": [512, 512],
                "initializer_type": "glorot",  # ["zero", "glorot"]
                "prm_loss_weight": 1.0,
                "reg_loss_weight": 800.0,
                "potential_weight": 5.0,
            },
            "gan": {
                "num_ens": 1,
                "layer_sizes": [256, 256],
                "initializer_type": "glorot",  # ["zero", "glorot"]
                "latent_dim": 2,
                "gp_lambda": 0.1,
                "critic_iter": 5,
                "potential_weight": 3.0,
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
    "rollout": {"rollout_batch_size": 4, "noise_eps": 0.1, "polyak_noise": 0.0, "random_eps": 0.1, "compute_Q": False},
    "evaluator": {
        "rollout_batch_size": 20,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_Q": True,
    },
    # training config
    "train": {
        "n_epochs": 5000,
        "n_cycles": 10,
        "n_batches": 40,
        "shaping_n_epochs": 5000,
        "save_interval": 2,
        "shaping_policy": 0,
    },
    "seed": 0,
}
