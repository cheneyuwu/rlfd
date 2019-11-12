params_config = {
    # config summary
    "config": "default",
    # environment config
    "env_name": "YWFetchPegInHoleRandInit-v0",
    "r_scale": 1.0,
    "r_shift": 0.0,
    "eps_length": 40,
    "env_args": {},
    "fix_T": True,
    # GAIL config
    "gail": {
        "num_epochs": int(2.4e4),
        # replay buffer size (gail and demo)
        "buffer_size": int(1e6),
        # demonstrations
        "num_demo": 40,
        "scope": "gail",
        "policy_step": 2,
        "disc_step": 1,
        "gen_layer_sizes": [100, 100],
        "disc_layer_sizes": [100, 100],
        "max_kl": 0.01,
        "gen_ent_coeff": 0,
        "disc_ent_coeff": 1e-3,
        "lam": 0.97,
        "cg_damping": 3e-4,
        "cg_iters": 10,
        "vf_iters": 5,
        "vf_batch_size": 128,
        # normalizer
        "norm_eps": 0.01,
        "norm_clip": 5,
    },
    # rollouts config
    "rollout": {
        "rollout_batch_size": 4,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": False,
    },
    "evaluator": {
        "rollout_batch_size": 20,
        "noise_eps": 0.0,
        "polyak_noise": 0.0,
        "random_eps": 0.0,
        "compute_q": False,
    },
    "seed": 0,
}
