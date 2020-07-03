parameters = {
    # config summary
    "algo": "sac",
    "config": "default",
    # environment config
    "env_name": "InvertedPendulum-v2",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 0,
    "env_args": {},
    "gamma": 0.99,
    "fix_T": False,
    # learner
    "random_expl_num_cycles": 0,
    "num_epochs": int(1e3),
    "num_cycles_per_epoch": int(1e3),
    "num_batches_per_cycle": 1,
    "expl_num_episodes_per_cycle": None,
    "expl_num_steps_per_cycle": 1,
    "eval_num_episodes_per_cycle": 4,
    "eval_num_steps_per_cycle": None,
    # agent config
    "agent": {
        "batch_size": 100,
        # use demonstrations
        "batch_size_demo": 128,
        "sample_demo_buffer": False,
        "initialize_with_bc": False,
        "initialize_num_epochs": 0,
        "use_demo_reward": False,
        "demo_strategy": "none",  # ["none", "bc", "nf", "gan"]
        # normalize observation
        "norm_eps": 0.01,
        "norm_clip": 5,
        # actor critic networks
        "layer_sizes": [256, 256],
        "auto_alpha": False,
        "alpha": 0.2,
        "q_lr": 3e-4,
        "vf_lr": 3e-4,
        "pi_lr": 3e-4,
        "action_l2": 0.0,
        # double q learning
        "polyak": 0.995,
        # multi step return
        "use_n_step_return": False,
        "bc_params": {
            "q_filter": False,
            "prm_loss_weight": 1.0,
            "aux_loss_weight": 1.0
        },
        "shaping_params": {
            # potential weight decay
            "potential_decay_scale": 1.0,
            "potential_decay_epoch": 0,
            "num_epochs": int(4e3),
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
            "orl": {
                "layer_sizes": [256, 256, 256],
                "q_lr": 1e-3,
                "pi_lr": 1e-3,
                "polyak": 0.995,
            },
        },
        # replay buffer setup
        "buffer_size": int(1e6),
    },
    "seed": 0,
}