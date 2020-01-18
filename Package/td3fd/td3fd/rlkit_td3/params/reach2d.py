import numpy as np

params_config = dict(
    # params required by td3fd for logging
    alg="rlkit-td3",
    config="default",
    env_name="Reach2DF",
    # rlkit default params
    algorithm_kwargs=dict(
        num_epochs=100,
        num_train_loops_per_epoch=10,
        num_eval_steps_per_epoch=100,
        num_trains_per_train_loop=20,
        num_expl_steps_per_train_loop=100,
        min_num_steps_before_training=0,
        max_path_length=20,
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
        hidden_sizes=[256, 256],
    ),
    policy_kwargs=dict(
        hidden_sizes=[256, 256],
    ),
    replay_buffer_size=int(1E6),
    demo_strategy="nf",
    shaping=dict(
        num_ensembles=2,
        num_epochs=int(3e3),
        batch_size=128,
        norm_obs=True,
        norm_eps=0.01,
        norm_clip=5,
        nf=dict(
            num_blocks=4,
            num_hidden=64,
            prm_loss_weight=1.0,
            reg_loss_weight=1e3,
            potential_weight=5e2,
        ),
        gan=dict(
            layer_sizes=[256, 256], 
            potential_weight=0.5,
        ),
    ),
    seed=0,
)