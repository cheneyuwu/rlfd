import numpy as np

params_config = dict(
    # params required by td3fd for logging
    alg="rlkit-td3",
    config="default",
    env_name="HalfCheetah-v3",
    seed=0,
    # rlkit default params
    algorithm_kwargs=dict(
        num_epochs=3000,
        num_train_loops_per_epoch=1,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        demo_batch_size=128,
        prm_loss_weight=1.0,
        aux_loss_weight=1.0,
        q_filter=True,
    ),
    qf_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    policy_kwargs=dict(
        hidden_sizes=[400, 300],
    ),
    replay_buffer_size=int(1E6),
    demo_strategy="nf",
    shaping=dict(
        num_ensembles=1,
        num_epochs=int(3e3),
        batch_size=128,
        norm_obs=True,
        norm_eps=0.01,
        norm_clip=5,
        nf=dict(
            num_blocks=4,
            num_hidden=100,
            prm_loss_weight=1.0,
            reg_loss_weight=200.0,
            potential_weight=500.0,
        ),
        gan=dict(
            layer_sizes=[256, 256, 256], 
            potential_weight=3.0,
        ),
    ),
)