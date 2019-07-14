params_config = {
    # Config Summary
    "config": ["rldense"],  # change this for each customized params file
    "seed": 0,
    # Environment Config
    "env_name": "Reach2DFDense",
    "r_scale": 1.0,  # scale the reward of the environment down
    "r_shift": 0.0,  # shift the reward of the environment up
    "eps_length": 0,  # overwrite the default length of the episode
    "env_args": {},  # extra arguments passed to the environment.
    # DDPG Config
    "ddpg": {
        "buffer_size": int(1e6),
        "scope": "ddpg",
        "use_td3": 1,  # whether or not to use td3
        "layers": 3,  # number of layers in the critic/actor networks
        "hidden": 256,  # number of neurons in each hidden layers
        "Q_lr": 0.001,  # critic learning rate
        "pi_lr": 0.001,  # actor learning rate
        "action_l2": 1.0,  # quadratic penalty on actions (before rescaling by max_u)
        "batch_size": 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
        "batch_size_demo": 128,  # number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
        "demo_strategy": "none",  # choose between ["none", "bc", "norm", "manual", "maf"]
        "q_filter": 1,  # whether or not a Q value filter should be used on the actor outputs
        "num_demo": 0,  # number of expert demo episodes
        "prm_loss_weight": 0.001,  # weight corresponding to the primary loss
        "aux_loss_weight": 0.0078,  # weight corresponding to the auxilliary loss also called the cloning loss
        # double q learning
        "polyak": 0.95,  # polyak averaging coefficient for double q learning
        # normalization
        "norm_eps": 0.01,  # epsilon used for observation normalization
        "norm_clip": 5,  # normalized observations are cropped to this values
        # i/o clippings
        "clip_obs": 200.0,
        "clip_pos_returns": False,  # Whether or not this environment has positive return or not.
        "clip_return": False,
        # replay strategy to be used
        "replay_strategy": "none",  # choose between ["her", "none"] (her for hindsight exp replay)
    },
    # HER Config
    "her": {"k": 4},  # number of additional goals used for replay
    # Rollouts Config
    "rollout": {
        "rollout_batch_size": 4,  # per mpi thread
        "random_eps": 0.1,  # percentage of time a random action is taken
        "noise_eps": 0.1,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    },
    "evaluator": {
        "rollout_batch_size": 20,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
        "random_eps": 0.0,
        "noise_eps": 0.01,
        "compute_Q": True,
    },
    # Training Config
    "train": {
        "n_epochs": 10,
        "n_cycles": 10,  # per epoch
        "n_batches": 40,  # training batches per cycle
        "save_interval": 2,
        "shaping_policy": 0,  # whether or not to use a pretrained shaping policy
    },
}