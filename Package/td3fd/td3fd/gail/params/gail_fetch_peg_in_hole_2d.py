# Best parameters found so far to be used for the Open AI fetch pick and place environment with a single goal and two objects.
params_config = {
    # config summary
    "config": "default",
    # environment config
    "env_name": "FetchPegInHole2DVersion-v1",
    "r_scale": 1.0,  # scale the reward of the environment down
    "r_shift": 0.0,  # shift the reward of the environment up
    "eps_length": 40,  # overwrite the default length of the episode provided in _max_episode_steps
    "env_args": {},  # extra arguments passed to the environment
    "fix_T": True,  # whether or not to fix episode length for all rollouts (if false, then use the ring buffer)
    # GAIL config
    "gail": {
        # replay buffer size (gail and demo)
        "buffer_size": int(1e6),
        # demonstrations
        "num_demo": 10,
        "scope": "gail",
        "policy_step": 3,
        "disc_step": 1,
        "gen_layer_sizes": [100, 100],
        "disc_layer_sizes": [100, 100],
        "max_kl": 0.01,
        "gen_ent_coeff": 0,
        "disc_ent_coeff": 1e-3,
        "lam": 0.97,  # value used by openai for their mujoco environments
        "cg_damping": 3e-4,
        "cg_iters": 10,
        "vf_iters": 5,
        "vf_batch_size": 20,
        # normalizer
        "norm_eps": 0.01,
        "norm_clip": 5,
    },
    # rollouts config
    "rollout": {
        "rollout_batch_size": 1,
        "noise_eps": 0.0,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
        "polyak_noise": 0.0,  # use polyak_noise * last_noise + (1 - polyak_noise) * curr_noise
        "random_eps": 0.0,  # percentage of time a random action is taken
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
