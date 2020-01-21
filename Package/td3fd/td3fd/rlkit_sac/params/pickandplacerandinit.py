import numpy as np

params_config = dict(
    # params required by td3fd for logging
    alg="rlkit-sac",
    config="default",
    env_name="YWFetchPegInHoleRandInit-v0",
    # rlkit params
    algorithm="SAC",
    version="normal",
    layer_size=256,
    demo_strategy="none",
    algorithm_kwargs=dict(
        num_epochs=1000,
        num_train_loops_per_epoch=10,
        num_eval_steps_per_epoch=200,
        num_trains_per_train_loop=40,
        num_expl_steps_per_train_loop=160,
        min_num_steps_before_training=0,
        max_path_length=40,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
        demo_batch_size=128,
        prm_loss_weight=1e-2,
        aux_loss_weight=1.0,
        q_filter=False,
    ),
    replay_buffer_size=int(1E6),
    shaping=dict(
        num_ensembles=2,
        num_epochs=int(8e3),
        batch_size=128,
        norm_obs=True,
        norm_eps=0.01,
        norm_clip=5,
        nf=dict(
            num_blocks=4,
            num_hidden=64,
            prm_loss_weight=1.0,
            reg_loss_weight=60.0,
            potential_weight=2e3,
        ),
        gan=dict(
            latent_dim=29,
            lambda_term=0.1,            
            layer_sizes=[256, 256],
            potential_weight=1.0,
        ),
    ),
    seed=0,
)