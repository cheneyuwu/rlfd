import os

from train import Demo, Train, Display, Plot

if __name__ == "__main__":

    demo_exp = Demo()
    train_exp = Train()
    display_exp = Display()
    plot_exp = Plot()

    # Common result directory
    exp_dir = os.getenv("EXPERIMENT")
    result_dir = os.path.join(exp_dir, "Result/Temp")
    train_exp.result_dir = result_dir
    demo_exp.result_dir = result_dir
    display_exp.result_dir = result_dir
    plot_exp.result_dir = result_dir

    # Specify the environment and reward type.
    # If you want to use dense reward, add "Dense" to the reward name and make sure the env manager recognizes that.
    # Please follow this convention so that you can plot the same env with different reward types in the same graph.
    environment = "BlockReachFirstOrder"  # "BlockReachFirstOrderDense" for dense reward
    demo_data_size = 16
    seed = 0 # change seed value inside the for loop

    train_exp.set_shared_cmd(
        env=environment,
        rl_action_l2=0.5,
        n_cycles=10,
        rl_num_sample=1,
        rl_batch_size=256,
        train_rl_epochs=2,
    )

    demo_exp.set_shared_cmd(
        num_demo=demo_data_size,
    )

    for i in range(1):
        
        # Change seed value in each iteration
        seed += i * 100

        # Change the result directory so that different seed goes to different directory
        train_exp.result_dir = result_dir
        demo_exp.result_dir = result_dir

        # Train the RL without demonstration using dense or sparse reward.
        train_exp.rl_only(
            seed=seed + 10,
        )

        # Generate demonstration data
        demo_exp.generate_demo(
            policy_file=os.path.join(demo_exp.result_dir, "RL/rl/policy_latest.pkl"),
            seed=seed + 20,
        )


        # Train the RL with demonstration through BC
        train_exp.rl_with_bc(
            seed=seed + 40,
            rl_batch_size_demo=128,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment+".npz"),
        )

        # Train the RL with demonstration through shaping
        train_exp.rl_with_shaping(
            seed=seed + 30,
            num_demo=demo_data_size,
            demo_file=os.path.join(train_exp.result_dir, "DemoData", environment+".npz"),
            # shaping_policy=os.path.join(train_exp.result_dir, "RLDemoShaping", "shaping/shaping_latest.ckpt"),
        )

    # Plot the training result
    plot_exp.plot(
        dir=plot_exp.result_dir,
        xy=[
            "epoch:test/success_rate",
            "epoch:test/total_shaping_reward",
            "epoch:test/total_reward",
            "epoch:test/mean_Q",
        ],
    )

    # Display a policy result (calls run_agent).
    display_exp.display(
        policy_file=os.path.join(display_exp.result_dir, "RL/rl/policy_latest.pkl"), num_itr=3
    )
